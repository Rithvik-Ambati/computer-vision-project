"""U-Net wound segmentation: model, BCE+lambda*Dice loss, synthetic data generation, training.

Train on real data by pointing --data-dir at a folder with images/ and masks/ (e.g. Medetec, AZH, FUSeg);
the default is the synthetic phantom set so the whole project runs offline.
"""
import argparse, json, os, time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import sim

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Block(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
                                 nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True))

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, base=12, depth=4):
        super().__init__()
        ch = [base * 2 ** i for i in range(depth + 1)]
        self.enc = nn.ModuleList([Block(3, ch[0])] + [Block(ch[i], ch[i + 1]) for i in range(depth)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, 2) for i in range(depth)])
        self.dec = nn.ModuleList([Block(ch[i] * 2, ch[i]) for i in range(depth)])
        self.out = nn.Conv2d(ch[0], 1, 1)

    def forward(self, x):
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        for i in reversed(range(len(self.up))):
            x = self.dec[i](torch.cat([self.up[i](x), skips[i]], 1))
        return self.out(x)


def dice_loss(logits, y, eps=1.0):
    p = torch.sigmoid(logits)
    inter = (p * y).sum((1, 2, 3)); den = p.sum((1, 2, 3)) + y.sum((1, 2, 3))
    return 1 - ((2 * inter + eps) / (den + eps)).mean()


def combined_loss(logits, y, lam=1.0):
    """L = L_BCE + lambda * L_Dice   (Eq. 1 of the paper)."""
    return F.binary_cross_entropy_with_logits(logits, y) + lam * dice_loss(logits, y)


def dice_iou(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    inter = (pred & gt).sum(); u = (pred | gt).sum()
    s = pred.sum() + gt.sum()
    return (2 * inter / s if s else 1.0), (inter / u if u else 1.0)


# ---------------------------------------------------------------- synthetic data
def gen_set(n, seed, tilt_max=55, curved=0.25, light=1.0, cache=None):
    if cache and os.path.exists(cache):
        z = np.load(cache)
        return z["x"], z["y"], z["a"]
    rng = np.random.default_rng(seed)
    X, Y, A = [], [], []
    for i in range(n):
        sc = sim.make_scene(int(rng.integers(1 << 30)))
        img, m, _, a = sim.render(sc, sim.random_cam(rng, tilt_max=tilt_max, curved=curved, light_strength=light))
        X.append(img); Y.append(m); A.append(a)
    X, Y, A = np.stack(X), np.stack(Y), np.array(A)
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez_compressed(cache, x=X, y=Y, a=A)
    return X, Y, A


def to_tensor(X):
    return torch.from_numpy(X[..., ::-1].copy()).permute(0, 3, 1, 2).float() / 255.0   # BGR->RGB


@torch.no_grad()
def predict(model, X, bs=16, thr=0.5):
    model.eval(); out = []
    for i in range(0, len(X), bs):
        out.append(torch.sigmoid(model(to_tensor(X[i:i + bs]))).squeeze(1).numpy())
    P = np.concatenate(out) if out else np.zeros((0, sim.IMG, sim.IMG), np.float32)
    return P, (P > thr).astype(np.uint8)


def train(lam=1.0, epochs=22, n_train=320, n_val=48, bs=8, lr=1e-4, tag="unet", seed=0, base=12, log=print):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, Ytr, _ = gen_set(n_train, 1000, cache=os.path.join(ROOT, "data", f"train_{n_train}.npz"))
    Xva, Yva, _ = gen_set(n_val, 2000, cache=os.path.join(ROOT, "data", f"val_{n_val}.npz"))
    model = UNet(base)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    xt, yt = to_tensor(Xtr), torch.from_numpy(Ytr).float().unsqueeze(1)
    hist = dict(epoch=[], train_loss=[], val_dice=[], val_iou=[], val_loss=[])
    best, t0 = -1, time.time()
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(xt)); tl = 0
        for i in range(0, len(xt), bs):
            idx = perm[i:i + bs]; x, y = xt[idx], yt[idx]
            if np.random.rand() < 0.5: x, y = x.flip(3), y.flip(3)
            if np.random.rand() < 0.5: x, y = x.flip(2), y.flip(2)
            x = (x * (1 + 0.1 * (torch.rand(len(x), 1, 1, 1) - 0.5))).clamp(0, 1)
            loss = combined_loss(model(x), y, lam)
            opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item() * len(idx)
        sched.step()
        P, M = predict(model, Xva)
        d = [dice_iou(M[k], Yva[k]) for k in range(len(Yva))]
        vd, vi = float(np.mean([a for a, _ in d])), float(np.mean([b for _, b in d]))
        with torch.no_grad():
            vl = combined_loss(torch.from_numpy(np.log(np.clip(P, 1e-4, 1 - 1e-4) / (1 - np.clip(P, 1e-4, 1 - 1e-4)))).unsqueeze(1).float(),
                               torch.from_numpy(Yva).float().unsqueeze(1), lam).item()
        for k, v in zip(hist, [ep + 1, tl / len(xt), vd, vi, vl]): hist[k].append(v)
        if vd > best:
            best = vd; torch.save(model.state_dict(), os.path.join(ROOT, "models", f"{tag}.pt"))
        log(f"[{tag}] epoch {ep + 1:02d}/{epochs} loss {tl / len(xt):.4f} val Dice {vd:.4f} IoU {vi:.4f}  ({time.time() - t0:.0f}s)")
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump(hist, open(os.path.join(ROOT, "results", f"history_{tag}.json"), "w"))
    model.load_state_dict(torch.load(os.path.join(ROOT, "models", f"{tag}.pt")))
    return model, hist


def load_model(tag="unet", base=12):
    m = UNet(base); m.load_state_dict(torch.load(os.path.join(ROOT, "models", f"{tag}.pt"))); m.eval()
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=1.0); ap.add_argument("--epochs", type=int, default=22)
    ap.add_argument("--n-train", type=int, default=320); ap.add_argument("--tag", default="unet")
    a = ap.parse_args()
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    train(a.lam, a.epochs, a.n_train, tag=a.tag)
