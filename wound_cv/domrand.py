"""Domain-randomised training of the U-Net on simulated wounds, tested zero-shot on real photos (extension).

Idea: the simulator's skin colour, wound size and lighting are narrow, so the network over-fits to them.  We keep the
simulator but randomise, on the fly, the appearance factors that differ between simulation and real photographs:
random zoom / rotation / flip (real wounds fill the frame), hue rotation, saturation, per-channel gain, brightness,
contrast, gamma, grayscale, blur and noise.  NO real label is used for training or for model selection
(the final epoch is kept), so the real-photo score stays a clean zero-shot test.

    python -m wound_cv.domrand train      # trains models/dr_unet.pt  (GPU is used automatically if available)
    python -m wound_cv.domrand eval       # zero-shot Dice/IoU on the 537 real test photos -> results/ext_domrand.json
"""
import json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

from . import segment as S, sim

ROOT = S.ROOT
RES = os.path.join(ROOT, "results")
DEV = S.DEVICE


def augment(x, y, strength=1.0):
    """x: B,3,H,W in [0,1] (RGB); y: B,1,H,W.  Returns augmented (x, y) on the same device."""
    B, _, H, W = x.shape
    dev = x.device
    # --- geometric: zoom toward the wound, rotate, flip
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, H, device=dev), torch.linspace(-1, 1, W, device=dev), indexing="ij")
    area = y.sum((1, 2, 3)).clamp(min=1)
    cx = (y[:, 0] * xs).sum((1, 2)) / area
    cy = (y[:, 0] * ys).sum((1, 2)) / area
    sc = torch.empty(B, device=dev).uniform_(0.20, 1.0)                  # fraction of the image side that is kept
    ang = torch.rand(B, device=dev) * 2 * math.pi
    flip = (torch.rand(B, device=dev) < 0.5).float() * 2 - 1
    jit = (1 - sc).unsqueeze(1) * (torch.rand(B, 2, device=dev) - 0.5) * 1.2
    ca, sa = torch.cos(ang), torch.sin(ang)
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0] = sc * ca * flip; theta[:, 0, 1] = -sc * sa; theta[:, 1, 0] = sc * sa * flip; theta[:, 1, 1] = sc * ca
    zoomed = sc < 0.999
    tx = torch.where(zoomed, cx * (1 - sc) + jit[:, 0], torch.zeros_like(cx)).clamp(-1, 1)
    ty = torch.where(zoomed, cy * (1 - sc) + jit[:, 1], torch.zeros_like(cy)).clamp(-1, 1)
    theta[:, 0, 2] = tx; theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    y = (F.grid_sample(y, grid, mode="bilinear", padding_mode="zeros", align_corners=False) > 0.5).float()
    # --- photometric
    k = strength
    hue = (torch.rand(B, device=dev) - 0.5) * 2 * math.radians(35) * k
    c, s = torch.cos(hue), torch.sin(hue)
    one3 = 1.0 / 3.0
    sq = math.sqrt(one3)
    Rm = torch.zeros(B, 3, 3, device=dev)                                # rotation about the grey axis (YIQ-free hue shift)
    for i in range(3):
        for j in range(3):
            Rm[:, i, j] = c * (1.0 if i == j else 0.0) + (1 - c) * one3 + (s * sq * (1 if (j - i) % 3 == 1 else -1) if i != j else 0)
    x = torch.einsum("bij,bjhw->bihw", Rm, x)
    gray = x.mean(1, keepdim=True)
    sat = 1 + (torch.rand(B, 1, 1, 1, device=dev) - 0.5) * 1.2 * k
    x = gray + (x - gray) * sat
    x = x * (1 + (torch.rand(B, 3, 1, 1, device=dev) - 0.5) * 0.5 * k)              # per-channel gain (white balance)
    x = (x - 0.5) * (1 + (torch.rand(B, 1, 1, 1, device=dev) - 0.5) * 0.6 * k) + 0.5 + (torch.rand(B, 1, 1, 1, device=dev) - 0.5) * 0.3 * k
    x = x.clamp(1e-3, 1) ** (1 + (torch.rand(B, 1, 1, 1, device=dev) - 0.5) * 0.7 * k)
    g = torch.rand(B, 1, 1, 1, device=dev) < 0.08
    x = torch.where(g, x.mean(1, keepdim=True).expand_as(x), x)
    blur = (torch.rand(B, 1, 1, 1, device=dev) < 0.35).float()
    x = blur * F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), 3, 1) + (1 - blur) * x
    x = x + torch.randn_like(x) * 0.02 * k * torch.rand(B, 1, 1, 1, device=dev)
    return x.clamp(0, 1), y


def train(n_train=480, epochs=14, bs=8, lr=2e-4, seed=0, tag="dr_unet"):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, Ytr, _ = S.gen_set(n_train, 3000, cache=os.path.join(ROOT, "data", f"dr_train_{n_train}.npz"))
    model = S.UNet(12).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    xt, yt = S.to_tensor(Xtr), torch.from_numpy(Ytr).float().unsqueeze(1)
    hist = dict(epoch=[], train_loss=[]); t0 = time.time()
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(xt)); tl = 0
        for i in range(0, len(xt), bs):
            idx = perm[i:i + bs]
            x, y = augment(xt[idx].to(DEV), yt[idx].to(DEV))
            loss = S.combined_loss(model(x), y, 0.0)                        # lambda = 0 (chosen on validation area bias)
            opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item() * len(idx)
        sched.step()
        hist["epoch"].append(ep + 1); hist["train_loss"].append(tl / len(xt))
        print(f"[{tag}] epoch {ep + 1:02d}/{epochs} loss {tl / len(xt):.4f} ({time.time() - t0:.0f}s)", flush=True)
    torch.save(model.state_dict(), os.path.join(ROOT, "models", f"{tag}.pt"))
    json.dump(hist, open(os.path.join(RES, f"history_{tag}.json"), "w"))
    return model


def real_test_split():
    from .realdata import load_all
    X, Y = load_all(os.path.join(ROOT, "data", "mendeley_all.npz"))
    rng = np.random.default_rng(0); perm = rng.permutation(len(X))
    te = perm[:int(.2 * len(X))]
    return X[te], Y[te], te


def evaluate(tags=("dr_unet",)):
    Xt, Yt, _ = real_test_split()
    out = {}
    for tag in tags:
        m = S.load_model(tag)
        P, M = S.predict(m, Xt)
        d = np.array([S.dice_iou(M[k], Yt[k]) for k in range(len(Yt))])
        ratio = M.sum((1, 2)) / np.maximum(Yt.sum((1, 2)), 1)
        out[tag] = dict(dice=float(d[:, 0].mean()), iou=float(d[:, 1].mean()), dice_median=float(np.median(d[:, 0])),
                        area_ratio_median=float(np.median(ratio)), n=int(len(Yt)))
        print(tag, out[tag])
    json.dump(out, open(os.path.join(RES, "ext_domrand.json"), "w"), indent=1)


if __name__ == "__main__":
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    if cmd == "train":
        train()
    elif cmd == "eval":
        evaluate(tuple(sys.argv[2:]) or ("dr_unet",))
