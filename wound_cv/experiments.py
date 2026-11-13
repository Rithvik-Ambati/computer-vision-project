"""All experiments + figures.  Run:  python -m wound_cv.experiments [stage ...]

Stages: calib gallery main sweeps repeat healing flow train worked   (default: all)
Results -> results/*.csv|json   Figures -> figs/*.png
"""
import json, os, sys
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import sim, geometry as G, segment as S

ROOT = S.ROOT
FIGS, RES = os.path.join(ROOT, "figs"), os.path.join(ROOT, "results")
os.makedirs(FIGS, exist_ok=True); os.makedirs(RES, exist_ok=True)

MODES = ["pixel", "scale", "homog", "full"]
MODE_NAME = {"pixel": "Pixels only (nominal scale)", "scale": "Marker scale only",
             "homog": "+ Perspective (homography)", "full": "+ Lens undistortion (proposed)"}
COL = {"pixel": "#8a8a8a", "scale": "#e08a1e", "homog": "#3b7dd8", "full": "#1f9d6b"}
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold"})


def save(fig, name):
    fig.savefig(os.path.join(FIGS, name), bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("  fig", name)


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def contour_of(mask):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(cs, key=cv2.contourArea).reshape(-1, 2) if cs else np.zeros((0, 2))


_cache = {}


def calib():
    if "c" not in _cache:
        _cache["c"] = G.calibrate_camera()
    return _cache["c"]


def model():
    if "m" not in _cache:
        _cache["m"] = S.load_model("unet_bce")   # chosen on validation area bias
    return _cache["m"]


# ------------------------------------------------------------------ evaluation core
def true_perimeter(scene):
    pts, _ = scene["wound"].poly(scene["scale"])
    return float(np.sum(np.linalg.norm(pts - np.roll(pts, -1, 0), axis=1)))


def run_eval(items, label):
    """items: list of dict(scene, cam, meta).  Returns DataFrame (one row per photo)."""
    imgs, masks, scenes = [], [], []
    for it in items:
        img, m, _, a = sim.render(it["scene"], it["cam"])
        imgs.append(img); masks.append(m)
    P, M = S.predict(model(), np.stack(imgs))
    rows = []
    for k, it in enumerate(items):
        sc = it["scene"]
        row = dict(it["meta"]); row["gt_area"] = sc["area"]; row["gt_per"] = true_perimeter(sc)
        row["dice"], row["iou"] = S.dice_iou(M[k], masks[k])
        det = G.detect_marker(imgs[k]) is not None
        row["marker_found"] = det
        for src, mk in (("gt", masks[k]), ("net", M[k])):
            for mode in MODES:
                r = G.pipeline_measure(imgs[k], mk, calib(), mode)
                row[f"{src}_{mode}"] = r["area"]; row[f"{src}_{mode}_per"] = r["perimeter"]
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, f"{label}.csv"), index=False)
    return df, imgs, masks, M


def pct_err(df, src="net"):
    out = df.copy()
    for m in MODES:
        out[f"e_{src}_{m}"] = (df[f"{src}_{m}"] - df.gt_area) / df.gt_area * 100
    return out


def build_items(n, seed, **cam_kw):
    rng = np.random.default_rng(seed)
    items = []
    for i in range(n):
        sc = sim.make_scene(int(rng.integers(1 << 30)))
        cam = sim.random_cam(rng, **{k: v for k, v in cam_kw.items() if k in ("tilt_max", "d_range", "curved", "light_strength")})
        for k, v in cam_kw.items():
            if k not in ("tilt_max", "d_range", "curved", "light_strength"):
                cam[k] = v
        items.append(dict(scene=sc, cam=cam, meta=dict(idx=i, tilt=cam["tilt"], D=cam["D"], R=cam["R"] if cam["R"] else np.inf)))
    return items


# ------------------------------------------------------------------ stages
def stage_calib():
    print("[calib]")
    c = calib()
    img, corners = c["example"]
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    vis = img.copy(); cv2.drawChessboardCorners(vis, (9, 6), corners, True)
    ax[0].imshow(rgb(vis)); ax[0].set_title("1. Detect checkerboard corners"); ax[0].axis("off")
    ax[1].bar(range(len(c["per_view"])), c["per_view"], color="#3b7dd8")
    ax[1].axhline(c["rms"], color="k", ls="--", lw=1); ax[1].set_xlabel("calibration view"); ax[1].set_ylabel("reprojection RMS (px)")
    ax[1].set_title(f"2. Reprojection error\n(overall RMS {c['rms']:.2f} px)")
    # undistortion grid demo
    g = np.full((sim.IMG, sim.IMG, 3), 255, np.uint8)
    for t in range(0, sim.IMG, 16):
        cv2.line(g, (t, 0), (t, sim.IMG), (0, 0, 0), 1); cv2.line(g, (0, t), (sim.IMG, t), (0, 0, 0), 1)
    K, d = c["K"], c["dist"]
    mx, my = cv2.initUndistortRectifyMap(K, np.array([-d[0], 0, 0, 0, 0.]), None, K, (sim.IMG, sim.IMG), cv2.CV_32FC1)
    warped = cv2.remap(g, mx, my, cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    ax[2].imshow(warped); ax[2].axis("off"); ax[2].set_title("3. Barrel distortion this lens produces\n(grid straight → curved)")
    txt = (f"focal length  f = {K[0, 0]:.1f} px  (true {sim.F_PX:.0f})\nk1 = {d[0]:.3f}  (true {sim.K1_TRUE:.2f})\n"
           f"views used {c['n_used']}/{c['n_views']}")
    fig.text(0.5, -0.16, txt, ha="center", fontsize=9, family="monospace")
    save(fig, "01_calibration.png")
    json.dump(dict(f=float(K[0, 0]), cx=float(K[0, 2]), cy=float(K[1, 2]), k1=float(d[0]), rms=c["rms"],
                   f_true=sim.F_PX, k1_true=sim.K1_TRUE, used=c["n_used"]), open(os.path.join(RES, "calibration.json"), "w"), indent=1)


def stage_gallery():
    print("[gallery]")
    sc = sim.make_scene(11)
    confs = [("frontal", sim.default_cam()), ("tilt 30°", sim.default_cam(tilt=30, az=20)), ("tilt 55°", sim.default_cam(tilt=55, az=200)),
             ("limb R=60 mm", sim.default_cam(tilt=15, az=80, R=60)), ("dim light", sim.default_cam(gain=.65, gamma=1.3, grad=.4, noise=5)),
             ("bright / glare", sim.default_cam(gain=1.25, contrast=1.3, gamma=.8, grad=.35))]
    fig, ax = plt.subplots(2, 6, figsize=(15, 5.3))
    for k, (name, cam) in enumerate(confs):
        img, m, _, a = sim.render(sc, cam)
        ax[0, k].imshow(rgb(img)); ax[0, k].set_title(name); ax[0, k].axis("off")
        o = rgb(img).copy(); cv2.drawContours(o, [contour_of(m).reshape(-1, 1, 2)], -1, (0, 255, 255), 1)
        ax[1, k].imshow(o); ax[1, k].axis("off"); ax[1, k].set_title(f"GT outline · {m.sum()} px", fontsize=8)
    fig.suptitle(f"Phantom simulator: one wound (true area {sc['area']:.0f} mm²) photographed under different conditions", y=1.0)
    save(fig, "02_phantom_gallery.png")


def stage_main():
    print("[main]")
    df, imgs, masks, M = run_eval(build_items(150, 1, tilt_max=45, curved=0.15), "main")
    d = pct_err(df); d["ok"] = d.marker_found
    # ---- ablation boxplots (oracle vs U-Net masks)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for a, src, ttl in zip(ax, ["gt", "net"], ["Perfect segmentation (isolates geometry)", "U-Net segmentation (end-to-end)"]):
        dd = pct_err(df, src)
        data = [np.abs(dd.loc[dd.marker_found | (m == "pixel"), f"e_{src}_{m}"]) for m in MODES]
        bp = a.boxplot(data, patch_artist=True, showfliers=False, medianprops=dict(color="k"))
        for p, m in zip(bp["boxes"], MODES): p.set_facecolor(COL[m]); p.set_alpha(.85)
        a.set_xticks(range(1, 5)); a.set_xticklabels(["pixels\nonly", "marker\nscale", "+homog-\nraphy", "+lens\n(ours)"])
        for i, m in enumerate(MODES):
            a.text(i + 1, a.get_ylim()[1] * .02 + np.nanpercentile(data[i], 75) if False else 0, "", ha="center")
            a.text(i + 1, -0.12, f"MAPE {np.nanmean(data[i]):.1f}%", transform=a.get_xaxis_transform(), ha="center", va="top", fontsize=8, color=COL[m])
        a.set_title(ttl)
    ax[0].set_ylabel("absolute area error (%)")
    fig.suptitle("Ablation: perspective correction is the big win; lens correction only shows once segmentation is accurate", y=1.03)
    save(fig, "03_ablation.png")
    # ---- pred vs GT scatter + Bland-Altman for the proposed pipeline
    ok = df.marker_found
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    mx = df.gt_area.max() / 100 * 1.05
    for m in ["pixel", "scale", "full"]:
        ax[0].scatter(df.gt_area[ok] / 100, df[f"net_{m}"][ok] / 100, s=12, alpha=.7, color=COL[m], label=MODE_NAME[m])
    ax[0].plot([0, mx], [0, mx], "k--", lw=1); ax[0].set_xlabel("true area (cm²)"); ax[0].set_ylabel("measured area (cm²)")
    ax[0].set_title("Measured vs true area"); ax[0].legend(fontsize=7)
    for a, m in zip(ax[1:], ["pixel", "full"]):
        diff = (df[f"net_{m}"][ok] - df.gt_area[ok]) / 100; avg = (df[f"net_{m}"][ok] + df.gt_area[ok]) / 200
        bias, sd = diff.mean(), diff.std()
        a.scatter(avg, diff, s=12, color=COL[m], alpha=.7)
        a.axhline(bias, color="k"); a.axhline(bias + 1.96 * sd, color="k", ls="--"); a.axhline(bias - 1.96 * sd, color="k", ls="--")
        a.set_title(f"Bland–Altman · {MODE_NAME[m]}"); a.set_xlabel("mean area (cm²)"); a.set_ylabel("measured − true (cm²)")
        a.text(.02, .96, f"bias {bias:+.2f} cm²\n95% LoA ±{1.96 * sd:.2f} cm²", transform=a.transAxes, va="top", fontsize=8,
               bbox=dict(fc="white", ec="#ccc"))
        a.set_ylim(-4.2, 4.2) if False else None
    save(fig, "04_agreement.png")
    # ---- segmentation quality + overlays
    fig = plt.figure(figsize=(14, 5.6))
    gs = fig.add_gridspec(2, 6)
    ax0 = fig.add_subplot(gs[:, 0:2]); ax0.hist(df.dice, bins=22, color="#3b7dd8", alpha=.9)
    ax0.axvline(df.dice.mean(), color="k", ls="--"); ax0.set_title(f"Test Dice  (mean {df.dice.mean():.3f}, IoU {df.iou.mean():.3f})")
    ax0.set_xlabel("Dice per image"); ax0.set_ylabel("count")
    order = np.argsort(df.dice.values); pick = [order[0], order[len(order) // 4], order[len(order) // 2], order[-1]]
    pick = [order[len(order) // 10], order[len(order) // 3], order[len(order) // 2], order[-5]]
    for k, idx in enumerate(pick):
        a = fig.add_subplot(gs[k // 2, 2 + 2 * (k % 2) // 1 if False else 2 + (k % 2) * 2: 4 + (k % 2) * 2])
        o = rgb(imgs[idx]).copy()
        cv2.drawContours(o, [contour_of(masks[idx]).reshape(-1, 1, 2)], -1, (0, 255, 0), 1)
        if M[idx].sum() > 0:
            cv2.drawContours(o, [contour_of(M[idx]).reshape(-1, 1, 2)], -1, (255, 255, 0), 1)
        a.imshow(o); a.axis("off"); a.set_title(f"Dice {df.dice[idx]:.2f} · tilt {df.tilt[idx]:.0f}°", fontsize=8)
    fig.text(0.5, 0.0, "green = ground truth outline · yellow = U-Net prediction", ha="center", fontsize=8)
    save(fig, "05_segmentation.png")
    # ---- perimeter
    fig, ax = plt.subplots(figsize=(4.8, 4))
    for m in ["pixel", "full"]:
        ax.scatter(df.gt_per[ok] / 10, df[f"net_{m}_per"][ok] / 10, s=12, color=COL[m], alpha=.7, label=MODE_NAME[m])
    lim = df.gt_per.max() / 10 * 1.05; ax.plot([0, lim], [0, lim], "k--", lw=1)
    ax.set_xlabel("true perimeter (cm)"); ax.set_ylabel("measured perimeter (cm)"); ax.set_title("Perimeter"); ax.legend(fontsize=7)
    save(fig, "06_perimeter.png")
    return df


def sweep_plot(df, key, xlabel, name, title, xs=None, log=False, extra_dice=False):
    d = pct_err(df)
    g = d.groupby(key)
    xs = sorted(d[key].unique()) if xs is None else xs
    fig, ax = plt.subplots(1, 2 if extra_dice else 1, figsize=(10.5 if extra_dice else 6, 4))
    a0 = ax[0] if extra_dice else ax
    for m in MODES:
        ys = []
        for x in xs:
            s = d[d[key] == x]; s = s[s.marker_found | (m == "pixel")]
            ys.append(np.nanmean(np.abs(s[f"e_net_{m}"])) if len(s) else np.nan)
        a0.plot(range(len(xs)), ys, "o-", color=COL[m], label=MODE_NAME[m], lw=2, ms=5)
    a0.set_xticks(range(len(xs))); a0.set_xticklabels([f"{x:g}" for x in xs]); a0.set_xlabel(xlabel); a0.set_ylabel("MAPE (%)")
    a0.set_title(title); a0.legend(fontsize=7)
    if log: a0.set_yscale("log")
    det = [d[d[key] == x].marker_found.mean() * 100 for x in xs]
    if extra_dice:
        a1 = ax[1]; a1.plot(range(len(xs)), [d[d[key] == x].dice.mean() for x in xs], "s-", color="#7a4fd6", lw=2, label="Dice")
        a1b = a1.twinx(); a1b.plot(range(len(xs)), det, "^--", color="#c0392b", label="marker detected %"); a1b.set_ylim(0, 105)
        a1b.grid(False); a1b.set_ylabel("marker detection rate (%)", color="#c0392b")
        a1.set_xticks(range(len(xs))); a1.set_xticklabels([f"{x:g}" for x in xs]); a1.set_xlabel(xlabel); a1.set_ylabel("Dice", color="#7a4fd6")
        a1.set_title("Segmentation & marker detection")
    save(fig, name)
    return det


def stage_sweeps():
    print("[sweeps]")
    out = {}
    # angle
    items = []
    for t in range(0, 70, 10):
        items += [dict(it, meta=dict(it["meta"], cond=t)) for it in build_items(18, 100 + t, tilt_max=0, curved=0, tilt=float(t))]
    df = run_eval(items, "sweep_tilt")[0]
    out["tilt_det"] = sweep_plot(df, "cond", "camera tilt from perpendicular (°)", "07_sweep_tilt.png", "Error vs camera tilt", extra_dice=True)
    # distance
    items = []
    for D in [170, 200, 230, 260, 290, 320]:
        items += [dict(it, meta=dict(it["meta"], cond=D)) for it in build_items(18, 200 + D, tilt_max=0, curved=0, tilt=20.0, D=float(D))]
    df = run_eval(items, "sweep_dist")[0]
    out["dist_det"] = sweep_plot(df, "cond", "camera distance (mm)", "08_sweep_distance.png", "Error vs shooting distance", extra_dice=True)
    # curvature (limb radius)
    items = []
    Rs = [1e4, 200, 120, 80, 55, 40]
    for R in Rs:
        items += [dict(it, meta=dict(it["meta"], cond=R)) for it in build_items(18, 300 + int(R), tilt_max=0, curved=0, tilt=15.0, R=(None if R == 1e4 else float(R)))]
    df = run_eval(items, "sweep_curv")[0]
    out["curv_det"] = sweep_plot(df, "cond", "limb radius of curvature (mm)   [flat → thin forearm]", "09_sweep_curvature.png",
                                 "Error vs surface curvature (planar-assumption test)", xs=Rs, extra_dice=True)
    # lighting
    items = []
    for L in [0.0, 0.5, 1.0, 1.5, 2.0]:
        items += [dict(it, meta=dict(it["meta"], cond=L)) for it in build_items(18, 400 + int(L * 10), tilt_max=30, curved=0, light_strength=L)]
    df = run_eval(items, "sweep_light")[0]
    out["light_det"] = sweep_plot(df, "cond", "lighting variation strength (1 = training range)", "10_sweep_lighting.png", "Error vs lighting variation", extra_dice=True)
    json.dump(out, open(os.path.join(RES, "sweep_extra.json"), "w"))


def stage_repeat():
    print("[repeat]")
    rng = np.random.default_rng(7)
    sc = sim.make_scene(4242, area_mm2=700)
    items = [dict(scene=sc, cam=sim.random_cam(rng, tilt_max=45), meta=dict(idx=i)) for i in range(60)]
    df = run_eval(items, "repeat")[0]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    cvs = {}
    data = []
    for m in MODES:
        v = df[f"net_{m}"][df.marker_found | (m == "pixel")] / 100; data.append(v.values)
        cvs[m] = float(v.std() / v.mean() * 100)
    vp = ax[0].violinplot(data, showmedians=True)
    for b, m in zip(vp["bodies"], MODES): b.set_facecolor(COL[m]); b.set_alpha(.8)
    ax[0].axhline(sc["area"] / 100, color="k", ls="--", lw=1.2, label=f"true {sc['area'] / 100:.2f} cm²")
    ax[0].set_xticks(range(1, 5)); ax[0].set_xticklabels(["pixels", "scale", "homog.", "ours"]); ax[0].set_ylabel("measured area (cm²)")
    ax[0].set_title("Same wound, 60 different photos"); ax[0].legend()
    ax[1].bar(range(4), [cvs[m] for m in MODES], color=[COL[m] for m in MODES])
    for i, m in enumerate(MODES): ax[1].text(i, cvs[m] + .3, f"{cvs[m]:.1f}%", ha="center")
    ax[1].set_xticks(range(4)); ax[1].set_xticklabels(["pixels", "scale", "homog.", "ours"]); ax[1].set_ylabel("coefficient of variation (%)")
    ax[1].set_title("Repeatability (lower = more reproducible)")
    save(fig, "11_repeatability.png")
    json.dump(cvs, open(os.path.join(RES, "repeatability.json"), "w"))


# ------------------------------------------------------------------ longitudinal healing + registration
def to_frame(H, arr, lo, size, ppm, interp=cv2.INTER_LINEAR):
    T = np.array([[ppm, 0, -lo[0] * ppm], [0, ppm, -lo[1] * ppm], [0, 0, 1]]) @ H
    return cv2.warpPerspective(arr, T, size, flags=interp)


def stage_healing():
    print("[healing]")
    days = [0, 7, 14, 21, 28, 35]
    ratios = np.array([1, .78, .58, .42, .28, .17])
    K, dist = calib()["K"], calib()["dist"]
    wounds = [(901, 900), (902, 1200), (903, 600)]
    rows, show = [], None
    rng = np.random.default_rng(55)
    for w, (seed, a0) in enumerate(wounds):
        skin = sim.make_skin(seed)
        r0 = np.random.default_rng(seed + 12345)
        wound = sim.Wound(r0, a0)
        base = sim.make_scene(seed, a0, skin=skin, wound=wound)
        imgs, masks, hs, trues = [], [], [], []
        for t, rat in enumerate(ratios * (1 + rng.normal(0, .02, len(ratios)))):
            sc = sim.make_scene(seed, a0, scale=np.sqrt(rat), healing=1 - rat, skin=skin, wound=wound)
            cam = sim.random_cam(rng, tilt_max=35)
            img, m, txy, a = sim.render(sc, cam)
            imgs.append(img); masks.append(m); trues.append(a)
        P, M = S.predict(model(), np.stack(imgs))
        meas = []
        for t in range(len(days)):
            r = G.pipeline_measure(imgs[t], M[t], calib(), "full")
            meas.append(r)
            rows.append(dict(wound=w, day=days[t], truth=trues[t], full=r["area"], ok=r["ok"],
                             pixel=G.pipeline_measure(imgs[t], M[t], calib(), "pixel")["area"]))
        if w == 0:
            show = (imgs, M, meas, trues, base)
    df = pd.DataFrame(rows)
    df["truth_dA"] = df.groupby("wound").truth.transform(lambda s: (s.iloc[0] - s) / s.iloc[0] * 100)
    df["full_dA"] = df.groupby("wound").full.transform(lambda s: (s.iloc[0] - s) / s.iloc[0] * 100)
    df["pixel_dA"] = df.groupby("wound").pixel.transform(lambda s: (s.iloc[0] - s) / s.iloc[0] * 100)
    df.to_csv(os.path.join(RES, "healing.csv"), index=False)
    imgs, M, meas, trues, base = show
    # ---------- figure: row of raw photos + common-frame overlay + curves
    fig = plt.figure(figsize=(15, 9.5))
    gs = fig.add_gridspec(3, 6, height_ratios=[1, 1.5, 1.3], hspace=.6)
    cmap = plt.cm.viridis(np.linspace(0, .9, len(days)))
    for t in range(6):
        a = fig.add_subplot(gs[0, t]); o = rgb(imgs[t]).copy()
        cv2.drawContours(o, [contour_of(M[t]).reshape(-1, 1, 2)], -1, (255, 255, 0), 1)
        if meas[t]["corners"] is not None:
            c = G.detect_marker(imgs[t]).astype(np.int32); cv2.polylines(o, [c], True, (0, 255, 0), 1)
        a.imshow(o); a.axis("off"); a.set_title(f"day {days[t]}\ntrue {trues[t] / 100:.2f} cm²  meas {meas[t]['area'] / 100:.2f}", fontsize=8)
    # common frame overlay: rectify each visit (undistorted) to the marker frame
    lo = np.array([35.0, -62.0]); size = (int(90 * 5), int(85 * 5)); ppm = 5
    ax = fig.add_subplot(gs[1, 0:3])
    bg = None
    for t in range(6):
        iu = cv2.undistort(imgs[t], K, dist)
        c = G.undistort_points(G.detect_marker(imgs[t]), K, dist); H = G.homography_to_mm(c)
        fr = to_frame(H, iu, lo, size, ppm)
        if t == 0: bg = fr
        mk = to_frame(H, G.undistort_mask(M[t], K, dist).astype(np.float32), lo, size, ppm) > .5
        cs = contour_of(mk.astype(np.uint8))
        if t == 0:
            ax.imshow(rgb(bg), extent=[lo[0], lo[0] + size[0] / ppm, lo[1] + size[1] / ppm, lo[1]])
        if len(cs):
            ax.plot(cs[:, 0] / ppm + lo[0], cs[:, 1] / ppm + lo[1], color=cmap[t], lw=2, label=f"day {days[t]}")
    ax.set_xlabel("mm (marker frame)"); ax.set_ylabel("mm"); ax.set_title("All visits rectified into ONE metric frame → boundaries contract concentrically")
    ax.legend(ncol=3, fontsize=7); ax.grid(False)
    ax = fig.add_subplot(gs[1, 3:6])
    for t in range(6):
        iu = cv2.undistort(imgs[t], K, dist)
        c = G.undistort_points(G.detect_marker(imgs[t]), K, dist); H = G.homography_to_mm(c)
        fr = to_frame(H, iu, lo, size, ppm)
        if t in (0, 5):
            ax.imshow(rgb(fr), extent=[lo[0], lo[0] + size[0] / ppm, lo[1] + size[1] / ppm, lo[1]], alpha=.55 if t == 5 else 1)
    ax.set_title("Day 0 (opaque) vs day 35 (blended): skin texture aligns, wound shrinks"); ax.grid(False)
    ax.set_xlabel("mm"); ax.set_ylabel("mm")
    # curves
    ax = fig.add_subplot(gs[2, 0:3])
    for w in range(3):
        d = df[df.wound == w]
        ax.plot(d.day, d.truth / 100, "k-", lw=1.5, alpha=.5, label="true" if w == 0 else None)
        ax.plot(d.day, d.full / 100, "o--", color=COL["full"], label="proposed" if w == 0 else None)
        ax.plot(d.day, d.pixel / 100, "x:", color=COL["pixel"], label="pixels only" if w == 0 else None)
    ax.set_xlabel("day"); ax.set_ylabel("area (cm²)"); ax.set_title("Wound area over time (3 simulated patients)"); ax.legend(fontsize=7)
    ax = fig.add_subplot(gs[2, 3:6])
    w0 = np.arange(6); wd = .27
    ax.bar(w0 - wd, df[df.wound == 0].truth_dA.values, wd, color="#444", label="true")
    ax.bar(w0, df[df.wound == 0].full_dA.values, wd, color=COL["full"], label="proposed")
    ax.bar(w0 + wd, df[df.wound == 0].pixel_dA.values, wd, color=COL["pixel"], label="pixels only")
    ax.set_xticks(w0); ax.set_xticklabels([f"d{d}" for d in days]); ax.set_ylabel("ΔA healing (%)"); ax.set_title("Eq. (2)  ΔA = (A₀ − Aₜ)/A₀ × 100%  (patient 1)"); ax.legend(fontsize=7)
    save(fig, "12_healing.png")
    mae = {m: float(np.nanmean(np.abs(df[df.day > 0][f"{m}_dA"] - df[df.day > 0].truth_dA))) for m in ["full", "pixel"]}
    json.dump(mae, open(os.path.join(RES, "healing_mae.json"), "w"))
    print("   healing ΔA MAE (pp):", mae)
    stage_registration()


def stage_registration():
    """SIFT+RANSAC raw-image registration vs marker-frame registration, measured against exact ground truth."""
    print("[registration]")
    rng = np.random.default_rng(31)
    seed, a0 = 777, 900
    skin = sim.make_skin(seed); wound = sim.Wound(np.random.default_rng(seed + 12345), a0)
    ratios = [1, .78, .58, .42, .28, .17]
    recs, ims, txys, masks = [], [], [], []
    for t, rat in enumerate(ratios):
        sc = sim.make_scene(seed, a0, scale=np.sqrt(rat), healing=1 - rat, skin=skin, wound=wound)
        cam = sim.random_cam(rng, tilt_max=40); cam["k1"] = 0.0     # images assumed already undistorted
        img, m, txy, a = sim.render(sc, cam)
        ims.append(img); txys.append(txy); masks.append(m)
    scene0 = sim.make_scene(seed, a0, skin=skin, wound=wound)
    ctr, mk = scene0["centre"], scene0["marker_tl"]

    def gt_tex2pix(txy):
        v, u = np.nonzero(~np.isnan(txy[..., 0]))
        sel = np.random.default_rng(0).choice(len(u), min(4000, len(u)), replace=False)
        H, _ = cv2.findHomography(txy[v[sel], u[sel]].astype(np.float32), np.stack([u[sel], v[sel]], 1).astype(np.float32), 0)
        return H
    Ht = [gt_tex2pix(x) for x in txys]
    grid = np.array([[ctr[0] + dx, ctr[1] + dy] for dx in np.linspace(-30, 30, 7) for dy in np.linspace(-30, 30, 7)], np.float32)
    sift = cv2.SIFT_create(2000)
    g0 = cv2.cvtColor(ims[0], cv2.COLOR_BGR2GRAY)
    rows, vis = [], None
    for t in range(1, 6):
        gt_ = cv2.cvtColor(ims[t], cv2.COLOR_BGR2GRAY)
        kmask0 = (1 - cv2.dilate(masks[0], np.ones((15, 15)))).astype(np.uint8) * 255
        kmaskt = (1 - cv2.dilate(masks[t], np.ones((15, 15)))).astype(np.uint8) * 255
        k0, d0 = sift.detectAndCompute(g0, kmask0); kt, dt = sift.detectAndCompute(gt_, kmaskt)
        bf = cv2.BFMatcher().knnMatch(dt, d0, k=2)
        good = [m for m, n in bf if m.distance < .75 * n.distance]
        src = np.float32([kt[m.queryIdx].pt for m in good]); dst = np.float32([k0[m.trainIdx].pt for m in good])
        Hs, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        # ground truth transfer (img_t -> img_0) from grid points
        pt_t = cv2.perspectiveTransform(grid.reshape(-1, 1, 2), Ht[t]).reshape(-1, 2)
        true0 = cv2.perspectiveTransform(grid.reshape(-1, 1, 2), Ht[0]).reshape(-1, 2)
        Hinv0 = np.linalg.inv(Ht[0])

        def err_mm(p0):
            back = cv2.perspectiveTransform(p0.reshape(-1, 1, 2).astype(np.float32), Hinv0).reshape(-1, 2)
            return float(np.mean(np.linalg.norm(back - grid, axis=1)))
        e_sift = err_mm(cv2.perspectiveTransform(pt_t.reshape(-1, 1, 2), Hs).reshape(-1, 2)) if Hs is not None else np.nan
        c_t, c_0 = G.detect_marker(ims[t]), G.detect_marker(ims[0])
        e_mark = np.nan
        if c_t is not None and c_0 is not None:
            Hm = np.linalg.inv(G.homography_to_mm(c_0)) @ G.homography_to_mm(c_t)
            e_mark = err_mm(cv2.perspectiveTransform(pt_t.reshape(-1, 1, 2), Hm).reshape(-1, 2))
        rows.append(dict(visit=t, matches=len(good), inliers=int(inl.sum()) if inl is not None else 0, sift_err_mm=e_sift, marker_err_mm=e_mark))
        if t == 5:
            vis = cv2.drawMatches(ims[t], kt, ims[0], k0, [m for m, i in zip(good, inl.ravel()) if i][:60], None,
                                  matchColor=(0, 255, 0), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            warp = cv2.warpPerspective(ims[t], Hs, (sim.IMG, sim.IMG))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, "registration.csv"), index=False)
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1])
    a = fig.add_subplot(gs[0, :]); a.imshow(rgb(vis)); a.axis("off")
    a.set_title(f"SIFT keypoints (wound masked out) matched day 35 → day 0 · RANSAC inliers shown ({df.inliers.iloc[-1]}/{df.matches.iloc[-1]} matches)")
    a = fig.add_subplot(gs[1, 0]); a.imshow(rgb(cv2.addWeighted(ims[0], .5, warp, .5, 0))); a.axis("off"); a.set_title("Day 35 warped onto day 0 (blend)")
    a = fig.add_subplot(gs[1, 1]); w = .38; x = np.arange(len(df))
    a.bar(x - w / 2, df.sift_err_mm, w, color="#7a4fd6", label="SIFT+RANSAC (raw photos)"); a.bar(x + w / 2, df.marker_err_mm, w, color=COL["full"], label="Marker homographies")
    a.set_xticks(x); a.set_xticklabels([f"d{d}" for d in [7, 14, 21, 28, 35]]); a.set_ylabel("registration error around wound (mm)"); a.set_title("Alignment error vs ground truth"); a.legend(fontsize=7)
    a = fig.add_subplot(gs[1, 2]); a.bar(x, df.inliers / df.matches * 100, color="#3b7dd8"); a.set_xticks(x); a.set_xticklabels([f"d{d}" for d in [7, 14, 21, 28, 35]])
    a.set_ylabel("RANSAC inlier ratio (%)"); a.set_title("Match quality")
    save(fig, "13_registration.png")


def stage_flow():
    print("[flow]")
    n = 6
    seed = 515
    sc = sim.make_scene(seed, 800)
    rng = np.random.default_rng(3)
    base = sim.default_cam(tilt=15, az=40, k1=0.0, noise=1.5, blur=.6)
    frames, gts = [], []
    for i in range(n):
        cam = dict(base, jx=rng.normal(0, 1.2), jy=rng.normal(0, 1.2), tilt=15 + rng.normal(0, 1.0), az=40 + rng.normal(0, 4), D=240 + rng.normal(0, 3), seed=i,
                   gain=1 + rng.normal(0, .03))
        field = sim.smooth_field(100 + i, amp_mm=1.6) if i else None
        img, m, _, _ = sim.render(sc, cam, field=field)
        frames.append(img); gts.append(m)
    g0 = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    ious = {"none": [], "homography (sparse)": [], "optical flow (dense)": []}
    sift = cv2.SIFT_create(1500)
    yy, xx = np.mgrid[0:sim.IMG, 0:sim.IMG].astype(np.float32)
    keep = None
    for i in range(1, n):
        gi = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        iou = lambda a, b: (a.astype(bool) & b.astype(bool)).sum() / max(1, (a.astype(bool) | b.astype(bool)).sum())
        ious["none"].append(iou(gts[0], gts[i]))
        # homography frame i -> 0 then propagate mask0 into frame i with inverse
        k0, d0 = sift.detectAndCompute(g0, None); ki, di = sift.detectAndCompute(gi, None)
        mt = [m for m, nn in cv2.BFMatcher().knnMatch(di, d0, k=2) if m.distance < .75 * nn.distance]
        Hs, _ = cv2.findHomography(np.float32([ki[m.queryIdx].pt for m in mt]), np.float32([k0[m.trainIdx].pt for m in mt]), cv2.RANSAC, 3)
        ph = cv2.warpPerspective(gts[0].astype(np.float32), np.linalg.inv(Hs), (sim.IMG, sim.IMG)) > .5
        ious["homography (sparse)"].append(iou(ph, gts[i]))
        fl = cv2.calcOpticalFlowFarneback(gi, g0, None, 0.5, 4, 21, 5, 7, 1.5, 0)    # frame i -> frame 0
        pf = cv2.remap(gts[0].astype(np.float32), xx + fl[..., 0], yy + fl[..., 1], cv2.INTER_LINEAR) > .5
        ious["optical flow (dense)"].append(iou(pf, gts[i]))
        if i == 3: keep = (frames[i], fl, gts[i], ph, pf)
    fi, fl, gt_i, ph, pf = keep
    hsv = np.zeros((sim.IMG, sim.IMG, 3), np.uint8)
    mag, ang = cv2.cartToPolar(fl[..., 0], fl[..., 1]); hsv[..., 0] = ang * 90 / np.pi; hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag * 60, 0, 255)
    fig, ax = plt.subplots(1, 5, figsize=(16, 3.6))
    ax[0].imshow(rgb(frames[0])); ax[0].set_title("frame 0 (reference)")
    ax[1].imshow(rgb(fi)); ax[1].set_title("frame 3 (hand shake + tissue shift)")
    ax[2].imshow(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)); ax[2].set_title(f"Farnebäck flow (max {mag.max():.1f} px)")
    o = rgb(fi).copy()
    for msk, colr in ((gt_i, (0, 255, 0)), (ph, (255, 80, 80)), (pf, (255, 255, 0))):
        cs = contour_of(msk.astype(np.uint8))
        if len(cs): cv2.drawContours(o, [cs.reshape(-1, 1, 2)], -1, colr, 1)
    ax[3].imshow(o); ax[3].set_title("mask propagation\ngreen GT · red homography · yellow flow", fontsize=8)
    for k, c in zip(ious, ["#999", "#7a4fd6", "#1f9d6b"]):
        ax[4].plot(range(1, n), ious[k], "o-", color=c, label=k, lw=2)
    ax[4].set_xlabel("frame index"); ax[4].set_ylabel("IoU of propagated frame-0 mask"); ax[4].set_ylim(0, 1.02); ax[4].legend(fontsize=7); ax[4].set_title("Why flow: propagate the mask in a burst")
    for a in ax[:4]: a.axis("off")
    save(fig, "14_optical_flow.png")
    json.dump({k: float(np.mean(v)) for k, v in ious.items()}, open(os.path.join(RES, "flow_iou.json"), "w"), indent=1)


def stage_train():
    print("[train curves]")
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for tag, c in (("unet", "#1f9d6b"), ("unet_bce", "#c0392b")):
        p = os.path.join(RES, f"history_{tag}.json")
        if not os.path.exists(p): continue
        h = json.load(open(p)); lab = "BCE + λ·Dice (λ=1)" if tag == "unet" else "BCE only (λ=0)"
        ax[0].plot(h["epoch"], h["train_loss"], color=c, label=lab); ax[1].plot(h["epoch"], h["val_dice"], color=c, label=lab)
        ax[2].plot(h["epoch"], h["val_iou"], color=c, label=lab)
    ax[0].set_title("Training loss"); ax[1].set_title("Validation Dice"); ax[2].set_title("Validation IoU")
    for a in ax: a.set_xlabel("epoch"); a.legend(fontsize=7)
    save(fig, "15_training_curves.png")


def stage_worked():
    print("[worked examples]")
    K, dist = calib()["K"], calib()["dist"]
    sc = sim.make_scene(2024, area_mm2=820)
    exs = [("A", "Frontal, flat skin", sim.default_cam(tilt=0, D=250, seed=1)),
           ("B", "Tilted 45° camera, flat skin", sim.default_cam(tilt=45, az=210, D=230, seed=2)),
           ("C", "Tilt 30° on a curved limb (R = 60 mm)", sim.default_cam(tilt=30, az=60, D=230, R=60, seed=3))]
    md = ["# Worked examples\n", f"True wound area (analytic polygon): **{sc['area']:.1f} mm² = {sc['area'] / 100:.2f} cm²**, "
          f"true perimeter {true_perimeter(sc):.1f} mm. Marker: {sim.MARKER_MM:.0f} mm ArUco (DICT_4X4_50, id {sim.MARKER_ID}). "
          f"Lens: f={K[0, 0]:.1f} px, k1={dist[0]:.3f} (from checkerboard calibration).\n"]
    for tag, name, cam in exs:
        img, gt, _, area = sim.render(sc, cam)
        P, M = S.predict(model(), img[None])
        mask = M[0]
        res = {m: G.pipeline_measure(img, mask, calib(), m) for m in MODES}
        full = res["full"]
        corners_raw = G.detect_marker(img)
        corners_ud = full["corners"]; H = full["H"]
        n_px = int(mask.sum()); d_, i_ = S.dice_iou(mask, gt)
        fig = plt.figure(figsize=(16, 8.4))
        gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05])
        a = fig.add_subplot(gs[0, 0]); o = rgb(img).copy()
        cv2.polylines(o, [corners_raw.astype(np.int32)], True, (0, 255, 0), 1)
        for k, p in enumerate(corners_raw): cv2.putText(o, str(k), tuple(int(v) for v in p), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 0), 1)
        a.imshow(o); a.axis("off"); a.set_title("① Photo + detected 40 mm marker\n(corners 0–3 in image pixels)")
        a = fig.add_subplot(gs[0, 1]); o = rgb(img).copy()
        cv2.drawContours(o, [contour_of(gt).reshape(-1, 1, 2)], -1, (0, 255, 0), 1)
        cv2.drawContours(o, [contour_of(mask).reshape(-1, 1, 2)], -1, (255, 255, 0), 1)
        a.imshow(o); a.axis("off"); a.set_title(f"② U-Net mask: {n_px} px (Dice {d_:.2f})\ngreen GT · yellow prediction")
        lo = np.array([-20.0, -20.0]); size = (int(130 * 4), int(110 * 4)); ppm = 4
        a = fig.add_subplot(gs[0, 2])
        iu = cv2.undistort(img, K, dist)
        fr = to_frame(H, iu, lo, size, ppm)
        a.imshow(rgb(fr), extent=[lo[0], lo[0] + size[0] / ppm, lo[1] + size[1] / ppm, lo[1]])
        a.add_patch(plt.Rectangle((0, 0), sim.MARKER_MM, sim.MARKER_MM, fill=False, ec="#00ff66", lw=1.5))
        a.set_title("③ Undistort + homography → metric view\nmarker becomes a true 40×40 mm square"); a.set_xlabel("mm"); a.set_ylabel("mm"); a.grid(False)
        a = fig.add_subplot(gs[0, 3])
        can = full["canvas"]; a.imshow(can, cmap="gray", extent=[full["lo"][0], full["lo"][0] + can.shape[1] / 8, full["lo"][1] + can.shape[0] / 8, full["lo"][1]])
        a.set_title(f"④ Mask in metric frame (8 px/mm)\n{int(can.sum())} canvas px ÷ 8² = {full['area']:.1f} mm²"); a.set_xlabel("mm"); a.grid(False)
        a = fig.add_subplot(gs[1, 0:2])
        vals = [res[m]["area"] for m in MODES]
        b = a.bar(range(4), vals, color=[COL[m] for m in MODES])
        a.axhline(area, color="k", ls="--"); a.text(3.45, area, f" true {area:.0f}", va="bottom", ha="right", fontsize=8)
        for i, m in enumerate(MODES):
            e = (vals[i] - area) / area * 100
            a.text(i, vals[i] + area * .02, f"{vals[i]:.0f} mm²\n{e:+.1f}%", ha="center", fontsize=8)
        a.set_xticks(range(4)); a.set_xticklabels(["pixels only", "marker scale", "+ homography", "+ lens (ours)"]); a.set_ylabel("area (mm²)")
        a.set_ylim(0, max(vals + [area]) * 1.3); a.set_title("⑤ Same photo, four ways to turn pixels into mm²")
        a = fig.add_subplot(gs[1, 2:4]); a.axis("off")
        Hs = "\n".join("   [" + "  ".join(f"{v:9.4f}" for v in row) + "]" for row in H)
        sc_px = G.scale_from_marker(corners_raw)
        corner_txt = "  ".join(f"({x:.1f},{y:.1f})" for x, y in corners_raw)
        txt = (f"{tag}. {name}\n\n"
               f"marker corners (px):\n  {corner_txt}\n"
               f"mean side {sc_px * sim.MARKER_MM:.1f} px  →  scale = {sc_px:.3f} px/mm\n\n"
               f"homography H (undistorted px → mm):\n{Hs}\n\n"
               f"mask pixels: {n_px}  →  naive: {n_px}/{sc_px:.3f}² = {n_px / sc_px ** 2:.0f} mm²\n"
               f"with H:   {full['area']:.1f} mm²   (true {area:.1f})  error {((full['area'] - area) / area * 100):+.1f}%\n"
               f"perimeter: {full['perimeter']:.1f} mm   (true {true_perimeter(sc):.1f})")
        a.text(0, 1, txt, va="top", family="monospace", fontsize=8.2)
        fig.suptitle(f"Worked example {tag}: {name}", y=.99, fontsize=12, fontweight="bold")
        save(fig, f"16_worked_example_{tag}.png")
        md.append(f"\n## Example {tag} — {name}\n\n| step | value |\n|---|---|\n"
                  f"| marker corners (px) | `{np.round(corners_raw, 1).tolist()}` |\n| marker scale (mean side) | {sc_px:.3f} px/mm |\n"
                  f"| homography H (image→mm) | `{np.round(H, 4).tolist()}` |\n| U-Net mask | {n_px} px, Dice {d_:.3f}, IoU {i_:.3f} |\n")
        md.append("\n| method | area (mm²) | error vs true |\n|---|---|---|\n" + "\n".join(
            f"| {MODE_NAME[m]} | {res[m]['area']:.1f} | {(res[m]['area'] - area) / area * 100:+.1f}% |" for m in MODES) + f"\n\ntrue area {area:.1f} mm²\n")
    open(os.path.join(ROOT, "worked_examples.md"), "w", encoding="utf-8").write("\n".join(md))


STAGES = dict(calib=stage_calib, gallery=stage_gallery, main=stage_main, sweeps=stage_sweeps, repeat=stage_repeat,
              healing=stage_healing, flow=stage_flow, train=stage_train, worked=stage_worked)

if __name__ == "__main__":
    todo = sys.argv[1:] or list(STAGES)
    for s in todo:
        STAGES[s]()
