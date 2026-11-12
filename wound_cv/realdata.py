"""Real-photo segmentation test on the Mendeley "Lower Limb and Feet Wound Image Dataset" (CC BY 4.0).

2,686 wound photos (331x331) with binary masks.  No patient IDs are published, so the random split below
cannot rule out near-duplicate photos of one patient across splits -- treat the held-out score as optimistic.
Protocol (seeded random split, no test leakage):
  70 % fit  |  10 % checkpoint selection  |  20 % HELD-OUT test (never used for training or selection)
Three models are compared on the held-out photos:
  synthetic-only  : the U-Net trained purely on phantoms (domain-gap probe)
  real-scratch    : trained from random init on real photos
  synth->real     : synthetic weights fine-tuned on real photos
These photos have no scale marker, so only segmentation (Dice/IoU/area ratio) is tested, not mm^2.
"""
import json, os, sys, time
import cv2
import numpy as np
import torch

from . import segment as S

ROOT = S.ROOT
DATA = os.path.join(ROOT, "real_data", "mendeley")
RES, FIGS = os.path.join(ROOT, "results"), os.path.join(ROOT, "figs")
SZ = 256


def load_all(cache):
    if os.path.exists(cache):
        z = np.load(cache); return z["x"], z["y"]
    X, Y = [], []
    for f in sorted(os.listdir(os.path.join(DATA, "wound_main"))):
        num = f.split("-")[1]
        im = cv2.imread(os.path.join(DATA, "wound_main", f)); lb = cv2.imread(os.path.join(DATA, "wound_mask", "wound_mask-" + num), 0)
        if im is None or lb is None: continue
        if lb.shape != im.shape[:2]: lb = cv2.resize(lb, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST)
        X.append(cv2.resize(im, (SZ, SZ), interpolation=cv2.INTER_AREA))
        Y.append((cv2.resize(lb, (SZ, SZ), interpolation=cv2.INTER_LINEAR) > 127).astype(np.uint8))
    X, Y = np.stack(X), np.stack(Y)
    np.savez_compressed(cache, x=X, y=Y)
    return X, Y


def evaluate(model, X, Y):
    P, M = S.predict(model, X)
    d = np.array([S.dice_iou(M[k], Y[k]) for k in range(len(Y))])
    ratio = M.sum((1, 2)) / np.maximum(Y.sum((1, 2)), 1)
    return dict(dice=d[:, 0], iou=d[:, 1], ratio=ratio), P, M


def train_real(Xtr, Ytr, Xva, Yva, tag, init=None, epochs=14, lr=2e-4, bs=8, lam=0.0, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    model = S.UNet(12)
    if init: model.load_state_dict(torch.load(os.path.join(ROOT, "models", f"{init}.pt")))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    xt, yt = S.to_tensor(Xtr), torch.from_numpy(Ytr).float().unsqueeze(1)
    hist = dict(epoch=[], train_loss=[], val_dice=[], val_iou=[]); best, t0 = -1, time.time()
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(xt)); tl = 0
        for i in range(0, len(xt), bs):
            idx = perm[i:i + bs]; x, y = xt[idx], yt[idx]
            if np.random.rand() < .5: x, y = x.flip(3), y.flip(3)
            if np.random.rand() < .5: x, y = x.flip(2), y.flip(2)
            k = int(np.random.randint(4)); x, y = torch.rot90(x, k, (2, 3)), torch.rot90(y, k, (2, 3))
            x = (x * (1 + .3 * (torch.rand(len(x), 1, 1, 1) - .5)) + .08 * (torch.rand(len(x), 3, 1, 1) - .5)).clamp(0, 1)
            loss = S.combined_loss(model(x), y, lam)
            opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item() * len(idx)
        sched.step()
        r, _, _ = evaluate(model, Xva, Yva)
        vd, vi = float(r["dice"].mean()), float(r["iou"].mean())
        for k, v in zip(hist, [ep + 1, tl / len(xt), vd, vi]): hist[k].append(v)
        if vd > best: best = vd; torch.save(model.state_dict(), os.path.join(ROOT, "models", f"{tag}.pt"))
        print(f"[{tag}] epoch {ep + 1:02d}/{epochs} loss {tl / len(xt):.4f} sel-val Dice {vd:.4f} IoU {vi:.4f} ({time.time() - t0:.0f}s)", flush=True)
    json.dump(hist, open(os.path.join(RES, f"history_{tag}.json"), "w"))
    model.load_state_dict(torch.load(os.path.join(ROOT, "models", f"{tag}.pt"))); model.eval()
    return model


def main(epochs=12):
    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    X, Y = load_all(os.path.join(ROOT, "data", "mendeley_all.npz"))
    rng = np.random.default_rng(0); perm = rng.permutation(len(X))
    n_te, n_sel = int(.2 * len(X)), int(.1 * len(X))
    te, sel, fit = perm[:n_te], perm[n_te:n_te + n_sel], perm[n_te + n_sel:]
    Xf, Yf, Xs, Ys, Xt, Yt = X[fit], Y[fit], X[sel], Y[sel], X[te], Y[te]
    print(f"photos {len(X)} | fit {len(Xf)} | checkpoint-selection {len(Xs)} | held-out test {len(Xt)}", flush=True)
    json.dump(dict(total=int(len(X)), fit=int(len(Xf)), select=int(len(Xs)), test=int(len(Xt))), open(os.path.join(RES, "real_split.json"), "w"))
    out = {}
    syn = S.load_model("unet_bce")
    out["synthetic-only"] = evaluate(syn, Xt, Yt)
    print("synthetic-only on real: Dice %.3f" % out["synthetic-only"][0]["dice"].mean(), flush=True)
    m1 = train_real(Xf, Yf, Xs, Ys, "real_scratch", init=None, epochs=epochs)
    out["real-scratch"] = evaluate(m1, Xt, Yt)
    m2 = train_real(Xf, Yf, Xs, Ys, "real_finetune", init="unet_bce", epochs=max(8, epochs - 4), lr=1e-4)
    out["synth→real"] = evaluate(m2, Xt, Yt)
    rows = {k: dict(dice=float(v[0]["dice"].mean()), iou=float(v[0]["iou"].mean()), dice_median=float(np.median(v[0]["dice"])),
                    area_ratio_median=float(np.median(v[0]["ratio"])), n=int(len(Xt))) for k, v in out.items()}
    json.dump(rows, open(os.path.join(RES, "real_data.json"), "w"), indent=1)
    np.savez(os.path.join(RES, "real_data_dice.npz"), **{k.replace("→", "_to_"): v[0]["dice"] for k, v in out.items()})
    print(json.dumps(rows, indent=1), flush=True)
    figure(out, Xt, Yt)


def figure(out, Xt, Yt):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    best = "synth→real"
    d = out[best][0]["dice"]; order = np.argsort(d)
    pick = [order[len(order) // 20], order[len(order) // 4], order[len(order) // 2], order[3 * len(order) // 4], order[-3]]
    fig = plt.figure(figsize=(15, 8.2)); gs = fig.add_gridspec(3, 5, height_ratios=[1, 1, 1.15])
    for k, i in enumerate(pick):
        a = fig.add_subplot(gs[0, k]); a.imshow(cv2.cvtColor(Xt[i], cv2.COLOR_BGR2RGB)); a.axis("off"); a.set_title(f"photo #{i}", fontsize=8)
        a = fig.add_subplot(gs[1, k]); o = cv2.cvtColor(Xt[i], cv2.COLOR_BGR2RGB).copy()
        for msk, col in ((Yt[i], (0, 255, 0)), (out[best][2][i], (255, 255, 0))):
            cs, _ = cv2.findContours(msk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(o, cs, -1, col, 1)
        a.imshow(o); a.axis("off"); a.set_title(f"Dice {d[i]:.2f}", fontsize=8)
    a = fig.add_subplot(gs[2, 0:3]); cols = {"synthetic-only": "#c0392b", "real-scratch": "#e08a1e", "synth→real": "#1f9d6b"}
    parts = a.violinplot([out[k][0]["dice"] for k in out], showmedians=True)
    for b, k in zip(parts["bodies"], out): b.set_facecolor(cols[k]); b.set_alpha(.8)
    a.set_xticks(range(1, 4)); a.set_xticklabels(list(out)); a.set_ylabel("Dice per photo")
    for i, k in enumerate(out): a.text(i + 1, 1.02, f"mean {out[k][0]['dice'].mean():.3f}", ha="center", fontsize=8)
    a.set_title("Real wound photos (held-out set): the synthetic→real domain gap"); a.set_ylim(0, 1.1)
    a = fig.add_subplot(gs[2, 3:5])
    for k in out: a.hist(np.clip(out[k][0]["ratio"], 0, 4), bins=30, alpha=.55, color=cols[k], label=k)
    a.axvline(1, color="k", ls="--"); a.set_xlabel("predicted / true mask area"); a.set_title("Area bias on real photos"); a.legend(fontsize=7)
    fig.text(.5, .0, "green = expert mask · yellow = prediction (synth→real) · these photos have no scale marker, so only segmentation is evaluated",
             ha="center", fontsize=8)
    fig.savefig(os.path.join(FIGS, "17_real_data.png"), bbox_inches="tight", facecolor="white"); plt.close(fig)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 12)
