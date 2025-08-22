import os
import sys
import math
import csv
import random
import argparse
from pathlib import Path
from typing import List, Optional, Tuple
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import cv2


try:
    import matplotlib.pyplot as plt
except:
    plt = None

# Optional video decoder (fast)
try:
    import decord
    decord.bridge.set_bridge("torch")
    USE_DECORD = True
except Exception:
    USE_DECORD = False

# Cap threads to avoid CPU stampedes and crashes 
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
try:
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass
# Utilities

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def contiguous_indices(total: int, num_frames: int, train: bool) -> np.ndarray:
    if total <= 0:
        return np.zeros((num_frames,), dtype=np.int64)
    clip = max(num_frames, min(total, num_frames * 4))
    if train:
        start = np.random.randint(0, max(1, total - clip + 1))
    else:
        start = max(0, (total - clip) // 2)
    window = np.arange(start, start + clip)
    take = np.linspace(0, clip - 1, num_frames).astype(np.int64)
    idx = window[take]
    return np.clip(idx, 0, total - 1)

def read_video_tchw(path: Path, num_frames: int, train: bool, img_size=128, crop_size=128) -> torch.Tensor:
    """Return RGB tensor [3, T, H, W] normalized to [-1, 1]."""
    if USE_DECORD:
        try:
            vr = decord.VideoReader(str(path))
            total = len(vr)
            idx = contiguous_indices(total, num_frames, train)
            frames = vr.get_batch(idx)  # [T,H,W,3] torch uint8 if torch bridge is set
            x = frames.float() / 255.0
        except Exception:
            x = None
    else:
        x = None

    if x is None:
        # Fallback to OpenCV
        cap = cv2.VideoCapture(str(path))
        all_frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            all_frames.append(torch.from_numpy(frame))
        cap.release()
        if len(all_frames) == 0:
            x = torch.zeros((num_frames, 224, 224, 3), dtype=torch.float32)
        else:
            total = len(all_frames)
            idx = contiguous_indices(total, num_frames, train)
            x = torch.stack([all_frames[i] for i in idx]).float() / 255.0  # [T,H,W,3]

    # Resize & crop (torch ops)
    x = x.permute(0, 3, 1, 2)  # [T,3,H,W]
    x = F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)
    H, W = x.shape[-2:]
    if train:
        top = 0 if H == crop_size else random.randint(0, H - crop_size)
        left = 0 if W == crop_size else random.randint(0, W - crop_size)
    else:
        top = (H - crop_size) // 2
        left = (W - crop_size) // 2
    x = x[:, :, top:top + crop_size, left:left + crop_size]  # [T,3,c,c]
    x = x.permute(1, 0, 2, 3).contiguous()  # [3,T,H,W]
    x = (x - 0.5) / 0.5  # normalize to [-1,1]
    return x

def compute_flow_tchw(rgb_tchw: torch.Tensor) -> torch.Tensor:
    """
    rgb_tchw: [3, T, H, W] in [-1,1] -> returns flow [2, T-1, H, W] (dx, dy) roughly in [-1,1]
    Uses classical Farnebäck optical flow (cv2.calcOpticalFlowFarneback).
    """
    x = ((rgb_tchw * 0.5) + 0.5).clamp(0, 1)  # back to [0,1]
    x = (x * 255).byte()
    T = x.shape[1]
    flows = []
    prev = x[:, 0].permute(1, 2, 0).cpu().numpy()  # H,W,3
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
    for t in range(1, T):
        cur = x[:, t].permute(1, 2, 0).cpu().numpy()
        cur_gray = cv2.cvtColor(cur, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, cur_gray, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0).astype(np.float32)  # H,W,2
        # Normalize a bit to keep tails from exploding
        flow = np.tanh(flow / 10.0)
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()  # [2,H,W]
        flows.append(flow)
        prev_gray = cur_gray
    if len(flows) == 0:
        return torch.zeros((2, 1, rgb_tchw.shape[2], rgb_tchw.shape[3]), dtype=torch.float32)
    flow_tchw = torch.stack(flows, dim=1)  # [2, T-1, H, W]
    return flow_tchw

# Datasets
class SignTrainSet(Dataset):
    def __init__(self, root: Path, csv_path: Path, keep: Optional[List[str]],
                 num_frames=24, img=128, crop=128, train=True):
        df = pd.read_csv(csv_path)
        if keep is not None:
            df = df[df['filename'].isin(keep)].reset_index(drop=True)
        self.paths = [root / fn for fn in df['filename'].tolist()]
        self.labels = df['label'].astype(int).tolist()
        self.num_frames = num_frames
        self.img = img
        self.crop = crop
        self.train = train

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        y = self.labels[i]
        rgb = read_video_tchw(p, self.num_frames, train=self.train, img_size=self.img, crop_size=self.crop)  # [3,T,H,W]
        flow = compute_flow_tchw(rgb)  # [2,T-1,H,W]
        # Pad flow to T for convenience
        if flow.shape[1] < rgb.shape[1]:
            pad_last = flow[:, -1:].repeat(1, rgb.shape[1] - flow.shape[1], 1, 1)
            flow = torch.cat([flow, pad_last], dim=1)
        return rgb, flow, y

class SignTestSet(Dataset):
    def __init__(self, root: Path, num_frames=24, img=128, crop=128):
        exts = {".mp4", ".avi", ".mov", ".mkv"}
        self.paths = sorted([p for p in root.glob("**/*") if p.suffix.lower() in exts])
        self.num_frames = num_frames
        self.img = img
        self.crop = crop

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        rgb = read_video_tchw(p, self.num_frames, train=False, img_size=self.img, crop_size=self.crop)
        flow = compute_flow_tchw(rgb)
        if flow.shape[1] < rgb.shape[1]:
            pad_last = flow[:, -1:].repeat(1, rgb.shape[1] - flow.shape[1], 1, 1)
            flow = torch.cat([flow, pad_last], dim=1)
        return rgb, flow, p.name

# Models (from scratch)

class Conv3DBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=(3, 3, 3), s=(1, 1, 1), p=(1, 1, 1)):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm3d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Small3D(nn.Module):
    def __init__(self, in_ch: int, num_classes: int, feat_out=512):
        super().__init__()
        self.stem = nn.Sequential(
            Conv3DBlock(in_ch, 64, k=(3, 7, 7), s=(1, 2, 2), p=(1, 3, 3)),
            nn.MaxPool3d((2, 2, 2))
        )
        self.l1 = nn.Sequential(Conv3DBlock(64, 128), Conv3DBlock(128, 128), nn.MaxPool3d((2, 2, 2)))
        self.l2 = nn.Sequential(Conv3DBlock(128, 256), Conv3DBlock(256, 256), nn.MaxPool3d((2, 2, 2)))
        self.l3 = nn.Sequential(Conv3DBlock(256, 512), Conv3DBlock(512, 512), nn.AdaptiveMaxPool3d((1, 1, 1)))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.Linear(512, num_classes))
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):  # [B,C,T,H,W]
        x = self.stem(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return self.head(x)

class TwoStreamFusion(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.rgb = Small3D(in_ch=3, num_classes=num_classes)
        self.flow = Small3D(in_ch=2, num_classes=num_classes)

    def forward(self, rgb, flow):
        logits_rgb = self.rgb(rgb)
        logits_flow = self.flow(flow)
        logits = (logits_rgb + logits_flow) / 2.0
        return logits, logits_rgb, logits_flow


# Training / Eval

# --- Metrics helpers (no sklearn required) ---
import numpy as _np

def build_confusion(ys, ps, num_classes: int):
    cm = _np.zeros((num_classes, num_classes), dtype=_np.int64)
    ys = _np.asarray(ys, dtype=_np.int64)
    ps = _np.asarray(ps, dtype=_np.int64)
    _np.add.at(cm, (ys, ps), 1)
    return cm


def f1_macro_from_confusion(cm):
    C = cm.shape[0]
    f1s = []
    for c in range(C):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        denom = (2 * tp + fp + fn)
        f1s.append((2 * tp) / denom if denom > 0 else 0.0)
    return float(_np.mean(f1s))

def stratified_split(csv_path: Path, val_ratio=0.15, seed=42):
    df = pd.read_csv(csv_path)
    g = defaultdict(list)
    for fn, lb in zip(df['filename'], df['label'].astype(int)):
        g[int(lb)].append(fn)
    rng = random.Random(seed)
    train_keep, val_keep = [], []
    for lb, files in g.items():
        files = files[:]
        rng.shuffle(files)
        k = max(1, int(len(files) * val_ratio))
        val_keep += files[:k]
        train_keep += files[k:]
    return train_keep, val_keep

def class_weights_from(csv_path: Path, keep: List[str], num_classes: int):
    df = pd.read_csv(csv_path)
    df = df[df['filename'].isin(keep)]
    labels = df['label'].astype(int).tolist()
    cnt = Counter(labels)
    w = torch.tensor([1.0 / max(1, cnt.get(c, 1)) for c in range(num_classes)], dtype=torch.float32)
    w = (w / w.sum()) * num_classes
    return w

def accuracy_from_logits(logits, y):
    return (logits.argmax(1) == y).float().mean().item()

@torch.no_grad()
def evaluate(model, val_ld, device, loss_fn, aux_lambda, num_classes):
    model.eval()
    tot_l = tot_a = n = 0.0
    ys, ps = [], []
    for (rgb, flow, y) in val_ld:
        rgb = rgb.to(device, non_blocking=True)
        flow = flow.to(device, non_blocking=True)
        y = torch.as_tensor(y, device=device)
        logits, l_rgb, l_flow = model(rgb, flow)
        loss = loss_fn(logits, y) + aux_lambda * (loss_fn(l_rgb, y) + loss_fn(l_flow, y))
        pred = logits.argmax(1)
        bs = y.size(0)
        tot_l += loss.item() * bs
        tot_a += (pred == y).float().mean().item() * bs
        n += bs
        ys += y.detach().cpu().tolist()
        ps += pred.detach().cpu().tolist()
    cm = build_confusion(ys, ps, num_classes)
    f1 = f1_macro_from_confusion(cm)
    return tot_l / max(1, n), tot_a / max(1, n), f1

def train_one_epoch(model, train_ld, optimizer, scaler, device, loss_fn, aux_lambda, grad_clip=1.0):
    model.train()
    tot_l = tot_a = n = 0.0
    pbar = tqdm(train_ld, leave=False, desc="train")
    use_amp = (device.type == 'cuda')
    for (rgb, flow, y) in pbar:
        rgb = rgb.to(device, non_blocking=True)
        flow = flow.to(device, non_blocking=True)
        y = torch.as_tensor(y, device=device)
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.cuda.amp.autocast():
                logits, l_rgb, l_flow = model(rgb, flow)
                loss = loss_fn(logits, y) + aux_lambda * (loss_fn(l_rgb, y) + loss_fn(l_flow, y))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, l_rgb, l_flow = model(rgb, flow)
            loss = loss_fn(logits, y) + aux_lambda * (loss_fn(l_rgb, y) + loss_fn(l_flow, y))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        acc = accuracy_from_logits(logits.detach(), y)
        bs = y.size(0)
        tot_l += loss.item() * bs
        tot_a += acc * bs
        n += bs
        pbar.set_postfix(loss=f"{tot_l/max(1,n):.4f}", acc=f"{tot_a/max(1,n):.4f}")
    return tot_l / max(1, n), tot_a / max(1, n)

@torch.no_grad()
def collect_preds(model, loader, device):
    model.eval()
    ys, ps = [], []
    for (rgb, flow, y) in loader:
        rgb = rgb.to(device, non_blocking=True)
        flow = flow.to(device, non_blocking=True)
        y = torch.as_tensor(y, device=device)
        logits, _, _ = model(rgb, flow)
        pred = logits.argmax(1)
        ys += y.cpu().tolist()
        ps += pred.cpu().tolist()
    return np.array(ys), np.array(ps)

@torch.no_grad()
def predict_submission(model, test_ld, out_csv: Path, device):
    model.eval()
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "label"])
        for (rgb, flow, names) in tqdm(test_ld, desc="inference"):
            rgb = rgb.to(device, non_blocking=True)
            flow = flow.to(device, non_blocking=True)
            logits, _, _ = model(rgb, flow)
            preds = logits.argmax(1).cpu().tolist()
            for n, p in zip(names, preds):
                w.writerow([n, int(p)])

# Main

def main():
    parser = argparse.ArgumentParser(description="model")
    parser.add_argument("--train_dir", type=str, default="data/public_train")
    parser.add_argument("--test_dir", type=str, default="data/public_test")
    parser.add_argument("--label_csv", type=str, default="public_train_label.csv")
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_frames", type=int, default=24)
    parser.add_argument("--img_size", type=int, default=320)
    parser.add_argument("--crop_size", type=int, default=320)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smooth", type=float, default=0.1)
    parser.add_argument("--aux_lambda", type=float, default=0.3)
    parser.add_argument("--workers", type=int, default=4)

    args = parser.parse_args()

    # Setup
    seed_everything(args.seed)
    device = get_device()

    ROOT = Path(".").resolve()
    TRAIN_DIR = ROOT / args.train_dir
    TEST_DIR = ROOT / args.test_dir
    LABEL_CSV = ROOT / args.label_csv
    OUT_DIR = ROOT / args.out_dir
    OUT_DIR.mkdir(exist_ok=True, parents=True)

    assert LABEL_CSV.exists(), f"Label CSV not found at {LABEL_CSV}"
    df_all = pd.read_csv(LABEL_CSV)
    assert {"filename", "label"}.issubset(df_all.columns), "CSV must have columns: filename,label"

    num_classes = int(df_all['label'].astype(int).max()) + 1

    print(f"[device] {device.type} | decord={USE_DECORD}")
    print(f"[data] #classes={num_classes}, #videos={len(df_all)}")

    # Split
    train_keep, val_keep = stratified_split(LABEL_CSV, val_ratio=args.val_ratio, seed=args.seed)

    # Datasets / Loaders
    train_ds = SignTrainSet(TRAIN_DIR, LABEL_CSV, train_keep,
                            num_frames=args.num_frames, img=args.img_size, crop=args.crop_size, train=True)
    val_ds = SignTrainSet(TRAIN_DIR, LABEL_CSV, val_keep,
                          num_frames=args.num_frames, img=args.img_size, crop=args.crop_size, train=False)

    pin = (device.type == "cuda")
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, pin_memory=pin, drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=pin)

    print(f"[split] train videos: {len(train_ds)} | val videos: {len(val_ds)}")

    # Class weights
    cls_w = class_weights_from(LABEL_CSV, train_keep, num_classes).to(device)

    # Model / Opt / Sched / Loss
    model = TwoStreamFusion(num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(weight=cls_w, label_smoothing=args.label_smooth)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    grad_clip = 1.0

    ckpt_path = OUT_DIR / "best.pt"
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}
    best_acc = 0.0

    # Train
    for epoch in range(1, args.epochs + 1):
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{args.epochs} (lr={lr_now:.2e})")
        tr_l, tr_a = train_one_epoch(model, train_ld, optimizer, scaler, device, loss_fn, args.aux_lambda, grad_clip)
        va_l, va_a, va_f1 = evaluate(model, val_ld, device, loss_fn, args.aux_lambda, num_classes)
        sched.step()

        history["train_loss"].append(tr_l); history["val_loss"].append(va_l)
        history["train_acc"].append(tr_a); history["val_acc"].append(va_a)
        history["val_f1"].append(va_f1)
        print(f"  train loss={tr_l:.4f} acc={tr_a:.4f} | val loss={va_l:.4f} acc={va_a:.4f} f1={va_f1:.4f}")

        if va_a >= best_acc:
            best_acc = va_a
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_acc": best_acc}, ckpt_path)
            print(f"  ↳ saved {ckpt_path} (val_acc={best_acc:.4f})")

    # Save history & curves
    with open(OUT_DIR / "history.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss", "train_acc", "val_acc", "val_f1"])
        for i in range(len(history["train_loss"])):
            w.writerow([i + 1, history["train_loss"][i], history["val_loss"][i], history["train_acc"][i], history["val_acc"][i], history["val_f1"][i]])

    if plt is not None:
        epochs = range(1, len(history["train_loss"]) + 1)
        plt.figure(figsize=(7, 5))
        plt.plot(epochs, history["train_loss"], label="train")
        plt.plot(epochs, history["val_loss"], label="val")
        plt.legend(); plt.xlabel("epoch"); plt.ylabel("loss"); plt.tight_layout()
        plt.savefig(OUT_DIR / "loss_curve.png", dpi=160); plt.close()

        plt.figure(figsize=(7, 5))
        plt.plot(epochs, history["train_acc"], label="train")
        plt.plot(epochs, history["val_acc"], label="val")
        plt.legend(); plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.tight_layout()
        plt.savefig(OUT_DIR / "acc_curve.png", dpi=160); plt.close()

    print(f"[train] complete. best val acc={best_acc:.4f}")

    # Confusion Matrix + macro-F1 on validation (load best)
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        # Collect preds and build confusion matrix without sklearn
        y_true, y_pred = collect_preds(model, val_ld, device)
        cm = build_confusion(y_true, y_pred, num_classes)
        np.savetxt(OUT_DIR / "confusion_val.csv", cm, fmt="%d", delimiter=",")

        val_f1 = f1_macro_from_confusion(cm)
        val_acc_from_cm = (cm.trace() / cm.sum()) if cm.sum() > 0 else 0.0
        with open(OUT_DIR / "val_metrics.txt", "w") as f:
            f.write(f"macro_f1={val_f1:.6f}val_acc={val_acc_from_cm:.6f}")
        print(f"[eval] macro-F1={val_f1:.4f}, val_acc={val_acc_from_cm:.4f}")

        if plt is not None:
            fig = plt.figure(figsize=(7, 6)); plt.imshow(cm, interpolation="nearest")
            plt.title("confusion_val"); plt.colorbar(); plt.xlabel("Predicted"); plt.ylabel("True")
            plt.tight_layout(); plt.savefig(OUT_DIR / "confusion_val.png", dpi=160); plt.close(fig)

            row = cm.sum(axis=1, keepdims=True).clip(min=1)
            cmn = cm / row
            fig = plt.figure(figsize=(7, 6)); plt.imshow(cmn, interpolation="nearest")
            plt.title("confusion_val_norm"); plt.colorbar(); plt.xlabel("Predicted"); plt.ylabel("True")
            plt.tight_layout(); plt.savefig(OUT_DIR / "confusion_val_norm.png", dpi=160); plt.close(fig)
        print("[eval] saved confusion matrices to out_dir")

    # Test-time inference & submission

    test_ds = SignTestSet(TEST_DIR, num_frames=args.num_frames, img=args.img_size, crop=args.crop_size)
    test_ld = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.workers, pin_memory=(device.type == 'cuda'))

    out_csv = OUT_DIR / "predictions.csv"
    predict_submission(model, test_ld, out_csv, device)
    print(f"[inference] wrote {out_csv}")


if __name__ == "__main__":
    main()
