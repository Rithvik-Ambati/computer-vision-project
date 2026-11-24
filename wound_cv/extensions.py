"""Extension experiments: new metrics, curvature-aware area, uncertainty, error maps and extra figures.

    python -m wound_cv.extensions metrics   # HD95, ASSD, precision/recall, CCC, R2, within-tolerance, Wilcoxon, runtime + figs 20-24
    python -m wound_cv.extensions curv      # curvature-aware area vs planar, limb-radius sensitivity, detector ablation -> fig 25
    python -m wound_cv.extensions unc       # uncertainty intervals (MC / conformal) + reliability flag -> fig 26
    python -m wound_cv.extensions grid      # tilt x distance error heat-map -> fig 27
    python -m wound_cv.extensions curves    # training curves (fig 28) and real-photo gallery (fig 29) and domain-randomisation bar (fig 30)

All outputs go to results/ext_* and figs/20+ ; the original results/ and figs/01-19 are never touched.
"""
import json, os, sys, time
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage, stats

from . import sim, geometry as G, segment as S, curvature as CV
from .experiments import (build_items, calib, model, save, rgb, RES, FIGS, MODES, MODE_NAME, COL)

TEST_SEED, TEST_N = 1, 150          # identical to stage_main
SMALL = 250.0                       # mm^2: below this boundary error dominates (see paper)


# ----------------------------------------------------------------------------- helpers
def render_items(items):
    imgs, gts = [], []
    for it in items:
        img, m, _, _ = sim.render(it["scene"], it["cam"])
        imgs.append(img); gts.append(m)
    P, M = S.predict(model(), np.stack(imgs))
    return imgs, gts, M, P


def test_items():
    return build_items(TEST_N, TEST_SEED, tilt_max=45, curved=0.15)


def boundary(mask):
    m = mask.astype(bool)
    return m & ~ndimage.binary_erosion(m)


def surface_distances(a, b):
    """Boundary distances (px) from mask a to mask b and back.  Returns (hd95, assd) or (nan, nan)."""
    if a.sum() == 0 or b.sum() == 0:
        return np.nan, np.nan
    ba, bb = boundary(a), boundary(b)
    d_ab = ndimage.distance_transform_edt(~bb)[ba]
    d_ba = ndimage.distance_transform_edt(~ba)[bb]
    d = np.concatenate([d_ab, d_ba])
    return float(np.percentile(d, 95)), float(d.mean())


def prec_rec(pred, gt):
    p, g = pred.astype(bool), gt.astype(bool)
    tp = (p & g).sum()
    return (tp / p.sum() if p.sum() else 1.0), (tp / g.sum() if g.sum() else 1.0)


def ccc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(2 * np.cov(x, y, bias=True)[0, 1] / (x.var() + y.var() + (x.mean() - y.mean()) ** 2))


def measure_full_from(corners, mask, cal):
    """'full' pipeline given already-detected corners (so uncertainty runs can jitter them)."""
    c = G.undistort_points(corners, cal["K"], cal["dist"])
    m = G.undistort_mask(mask, cal["K"], cal["dist"])
    return G.measure(m, G.homography_to_mm(c))["area"]


def est_tilt(corners, cal):
    """Camera tilt (deg) recovered from the marker alone: angle between plane normal and optical axis."""
    cu = G.undistort_points(corners, cal["K"], cal["dist"])
    _, rv, _ = cv2.solvePnP(np.c_[CV._SQ, np.zeros(4)].astype(np.float32), cu.astype(np.float32), cal["K"], None, flags=cv2.SOLVEPNP_IPPE)
    return float(np.degrees(np.arccos(np.clip(abs(cv2.Rodrigues(rv)[0][2, 2]), 0, 1))))


# ----------------------------------------------------------------------------- 1. new metrics
def stage_metrics():
    print("[metrics]")
    cal = calib()
    items = test_items()
    t0 = time.time(); imgs, gts, M, P = render_items(items); t_seg = (time.time() - t0)
    main = pd.read_csv(os.path.join(RES, "main.csv"))
    rows = []
    for k, it in enumerate(items):
        hd, assd = surface_distances(M[k], gts[k]); pr, rc = prec_rec(M[k], gts[k])
        d, i = S.dice_iou(M[k], gts[k])
        rows.append(dict(idx=k, dice=d, iou=i, precision=pr, recall=rc, hd95_px=hd, assd_px=assd, gt_area=it["scene"]["area"],
                         tilt=it["cam"]["tilt"], D=it["cam"]["D"], marker_found=main.marker_found[k]))
    df = pd.DataFrame(rows)
    assert np.allclose(df.dice, main.dice, atol=1e-9), "re-render does not reproduce main.csv"
    print("   re-render reproduces main.csv Dice exactly (max diff %.2e)" % np.abs(df.dice - main.dice).max())
    for m in MODES:
        df[f"net_{m}"] = main[f"net_{m}"]; df[f"gt_{m}"] = main[f"gt_{m}"]
        df[f"err_{m}"] = (main[f"net_{m}"] - main.gt_area) / main.gt_area * 100
        df[f"gterr_{m}"] = (main[f"gt_{m}"] - main.gt_area) / main.gt_area * 100
    df.to_csv(os.path.join(RES, "ext_metrics_per_image.csv"), index=False)

    # runtime (CPU/GPU as installed)
    t0 = time.time()
    for k in range(30):
        G.pipeline_measure(imgs[k], M[k], cal, "full")
    t_geo = (time.time() - t0) / 30
    out = dict(device=str(S.DEVICE), n=len(df), n_marker=int(df.marker_found.sum()),
               segmentation=dict(dice_mean=float(df.dice.mean()), dice_median=float(df.dice.median()), iou_mean=float(df.iou.mean()),
                                 precision_mean=float(df.precision.mean()), recall_mean=float(df.recall.mean()),
                                 hd95_px_median=float(df.hd95_px.median()), hd95_px_mean=float(df.hd95_px.mean()),
                                 assd_px_mean=float(df.assd_px.mean())),
               runtime_ms=dict(segmentation_per_image=1000 * t_seg / len(items), geometry_per_image=1000 * t_geo))
    sel = df[df.marker_found.astype(bool)]
    agree = {}
    for m in MODES:
        s = df if m == "pixel" else sel
        e = s[f"err_{m}"]; ae = e.abs()
        agree[m] = dict(n=int(len(s)), mape=float(ae.mean()), median_abs=float(ae.median()), p95_abs=float(np.percentile(ae, 95)),
                        bias=float(e.mean()), within10=float((ae <= 10).mean() * 100), within20=float((ae <= 20).mean() * 100),
                        ccc=ccc(s.gt_area, s[f"net_{m}"]), r2=float(np.corrcoef(s.gt_area, s[f"net_{m}"])[0, 1] ** 2),
                        mape_gtmask=float(s[f"gterr_{m}"].abs().mean()))
    out["area_agreement_netmask"] = agree
    tests = {}
    for a, b in [("homog", "pixel"), ("full", "homog"), ("full", "scale"), ("full", "pixel"), ("homog", "scale")]:
        for src, pref in (("net", "err"), ("gt", "gterr")):
            ea, eb = sel[f"{pref}_{a}"].abs().values, sel[f"{pref}_{b}"].abs().values
            w = stats.wilcoxon(ea, eb)
            tests[f"{a}_vs_{b}_{src}mask"] = dict(median_abs_diff=float(np.median(ea - eb)), p=float(w.pvalue))
    out["wilcoxon_paired_abs_error"] = tests
    small = sel[sel.gt_area < SMALL]; large = sel[sel.gt_area >= SMALL]
    out["by_size"] = dict(small_n=int(len(small)), large_n=int(len(large)),
                          small_mape_full=float(small.err_full.abs().mean()), large_mape_full=float(large.err_full.abs().mean()),
                          small_hd95=float(small.hd95_px.mean()), large_hd95=float(large.hd95_px.mean()))
    json.dump(out, open(os.path.join(RES, "ext_metrics.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "wilcoxon_paired_abs_error"}, indent=1)[:2500])

    # ---------------- figures
    # fig 20: estimated vs true area
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.9), sharex=True, sharey=True)
    lim = [0, max(df.gt_area.max(), 1) * 1.15]
    for a, m in zip(ax, MODES):
        s = df if m == "pixel" else sel
        a.plot(lim, lim, "k--", lw=1); a.scatter(s.gt_area, s[f"net_{m}"], s=14, color=COL[m], alpha=.8)
        a.set_xlim(lim); a.set_ylim(lim); a.set_xlabel("true area (mm²)")
        a.set_title(MODE_NAME[m], fontsize=8.5)
        a.text(.04, .95, f"CCC {agree[m]['ccc']:.2f}   R² {agree[m]['r2']:.2f}\nMAPE {agree[m]['mape']:.1f}%", transform=a.transAxes, va="top", fontsize=8)
    ax[0].set_ylabel("estimated area (mm²)")
    fig.suptitle("Estimated vs. true wound area (U-Net masks; dashed = perfect agreement)", y=1.03, fontweight="bold")
    save(fig, "20_est_vs_true.png")
    # fig 21: cumulative error
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for m in MODES:
        s = df if m == "pixel" else sel
        v = np.sort(s[f"err_{m}"].abs().values); ax.plot(v, np.arange(1, len(v) + 1) / len(v) * 100, color=COL[m], lw=2, label=MODE_NAME[m])
    for t in (10, 20):
        ax.axvline(t, color="k", ls=":", lw=1)
    ax.set_xlim(0, 60); ax.set_xlabel("absolute area error (%)"); ax.set_ylabel("photos with error below x (%)"); ax.legend(fontsize=7.5, loc="lower right")
    ax.set_title("Cumulative error distribution")
    save(fig, "21_error_cdf.png")
    # fig 22: error vs wound size
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
    ax[0].axvspan(0, SMALL, color="#f6d9d9", alpha=.6, label=f"small wounds (< {SMALL:.0f} mm²)")
    ax[0].scatter(sel.gt_area, sel.err_full.abs(), s=15, color=COL["full"]); ax[0].set_xlabel("true area (mm²)"); ax[0].set_ylabel("absolute error, proposed (%)")
    bins = np.array([0, 250, 500, 800, 1100, 1600]); idx = np.digitize(sel.gt_area, bins) - 1
    med = [sel.err_full.abs()[idx == b].median() for b in range(len(bins) - 1)]
    ax[0].plot((bins[:-1] + bins[1:]) / 2, med, "ko-", label="bin median"); ax[0].legend(fontsize=7.5); ax[0].set_title("Error is largest for small wounds")
    ax[1].scatter(sel.gt_area, sel.hd95_px, s=15, color="#7a4fd6"); ax[1].axvspan(0, SMALL, color="#f6d9d9", alpha=.6)
    ax[1].set_xlabel("true area (mm²)"); ax[1].set_ylabel("boundary error HD95 (px)"); ax[1].set_title("Boundary error vs wound size")
    save(fig, "22_error_vs_size.png")
    # fig 23: boundary error overlays (best / median / 90th pct / worst by HD95)
    order = np.argsort(df.hd95_px.fillna(1e9).values)
    pick = [order[0], order[len(order) // 2], order[int(len(order) * .9)], order[-1]]
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.9))
    for a, k in zip(ax, pick):
        a.imshow(rgb(imgs[k])); a.axis("off")
        bg, bp = boundary(gts[k]), boundary(M[k])
        if bp.any() and bg.any():
            dt = ndimage.distance_transform_edt(~bg); yy, xx = np.nonzero(bp)
            sc = a.scatter(xx, yy, c=dt[yy, xx], s=2, cmap="turbo", vmin=0, vmax=8)
            yy, xx = np.nonzero(bg); a.scatter(xx, yy, s=1, color="white")
        a.set_title(f"HD95 {df.hd95_px[k]:.1f}px · Dice {df.dice[k]:.2f}", fontsize=8.5)
    fig.colorbar(sc, ax=ax, fraction=.015, pad=.01, label="distance of predicted edge to true edge (px); white = true edge")
    save(fig, "23_boundary_error.png")
    # fig 24: rectified metric canvas
    pick = [k for k in order[:60] if df.marker_found[k]][:1] + [k for k in order if df.marker_found[k]][len(order) // 3:len(order) // 3 + 1] + \
           [k for k in order if df.marker_found[k] and df.tilt[k] > 30][:2]
    fig, ax = plt.subplots(2, len(pick), figsize=(3.3 * len(pick), 6.6))
    for j, k in enumerate(pick):
        ax[0, j].imshow(rgb(imgs[k])); ax[0, j].axis("off"); ax[0, j].set_title(f"photo (tilt {df.tilt[k]:.0f}°)", fontsize=8.5)
        r = G.pipeline_measure(imgs[k], M[k], cal, "full")
        und = cv2.undistort(imgs[k], cal["K"], cal["dist"]); T = r["T"]
        w = 8 * 150; hh = 8 * 150
        rect = cv2.warpPerspective(und, T @ np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), (hh, w), borderValue=(110, 110, 110)) if T is not None else und
        can = r["canvas"]
        ax[1, j].imshow(rgb(rect[:can.shape[0], :can.shape[1]]) if can is not None else rgb(und)); ax[1, j].axis("off")
        ax[1, j].set_title(f"metric view: {r['area']:.0f} mm² (true {df.gt_area[k]:.0f})", fontsize=8.5)
        if can is not None:
            ax[1, j].contour(can, [.5], colors="lime", linewidths=1)
            ax[1, j].plot([10, 10 + 80], [can.shape[0] - 12] * 2, color="w", lw=3); ax[1, j].text(10, can.shape[0] - 22, "10 mm", color="w", fontsize=8)
    fig.suptitle("Marker homography turns the tilted photo into a top-down view in millimetres", y=1.0, fontweight="bold")
    save(fig, "24_rectified_view.png")


# ----------------------------------------------------------------------------- 2. curvature
def stage_curv():
    print("[curv]")
    cal = calib()
    Rs, facs = [1e4, 200, 120, 80, 55], [1.0, 0.9, 1.1, 0.8, 1.2]
    rows = []
    for R in Rs:
        for tilt in (15.0, 35.0):
            items = build_items(16, 5000 + int(R) + int(tilt), tilt_max=0, curved=0, tilt=tilt, R=(None if R == 1e4 else float(R)))
            imgs, gts, M, _ = render_items(items)
            for k, it in enumerate(items):
                a = it["scene"]["area"]
                row = dict(R=R, tilt=tilt, gt_area=a, dice=S.dice_iou(M[k], gts[k])[0])
                row["found_default"] = G.detect_marker(imgs[k]) is not None
                c = CV.detect_marker_tolerant(imgs[k]); row["found_tol"] = c is not None
                if c is not None:
                    for src, mk in (("gt", gts[k]), ("net", M[k])):
                        row[f"{src}_planar"] = measure_full_from(c, mk, cal)
                        for f in (facs if src == "net" else [1.0]):
                            r = CV.measure_cylinder(imgs[k], mk, cal, R * f, corners=c)
                            row[f"{src}_cyl_{f:g}"] = r["area"]
                            if f == 1.0:
                                row[f"{src}_cyl_rms"] = r["rms"]
                rows.append(row)
        print("   R", R, "done", flush=True)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, "ext_curvature.csv"), index=False)
    summ = {}
    for R in Rs:
        d = df[df.R == R]; f = d[d.found_tol.astype(bool)]
        e = lambda col: ((f[col] - f.gt_area) / f.gt_area * 100)
        summ[str(R)] = dict(n=int(len(d)), found_default=int(d.found_default.sum()), found_tolerant=int(d.found_tol.sum()),
                            planar_mape_gt=float(e("gt_planar").abs().mean()) if len(f) else None,
                            cyl_mape_gt=float(e("gt_cyl_1").abs().mean()) if len(f) else None,
                            planar_mape_net=float(e("net_planar").abs().mean()) if len(f) else None,
                            planar_bias_net=float(e("net_planar").mean()) if len(f) else None,
                            **{f"cyl_mape_net_R{fc:g}": (float(e(f"net_cyl_{fc:g}").abs().mean()) if len(f) else None) for fc in facs},
                            cyl_bias_net_R1=float(e("net_cyl_1").mean()) if len(f) else None)
    json.dump(summ, open(os.path.join(RES, "ext_curvature.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))
    # fig 25
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    xs = np.arange(len(Rs)); labs = ["flat" if R == 1e4 else f"{R:g}" for R in Rs]
    ax[0].bar(xs - .2, [summ[str(R)]["found_default"] / summ[str(R)]["n"] * 100 for R in Rs], .4, color="#8a8a8a", label="default detector")
    ax[0].bar(xs + .2, [summ[str(R)]["found_tolerant"] / summ[str(R)]["n"] * 100 for R in Rs], .4, color=COL["full"], label="curvature-tolerant")
    ax[0].set_xticks(xs); ax[0].set_xticklabels(labs); ax[0].set_xlabel("limb radius (mm)"); ax[0].set_ylabel("marker found (%)"); ax[0].legend(fontsize=8); ax[0].set_title("Marker detection on curved skin")
    ax[1].plot(xs, [summ[str(R)]["planar_mape_net"] or np.nan for R in Rs], "o-", color=COL["homog"], lw=2, label="planar homography (paper)")
    ax[1].plot(xs, [summ[str(R)]["cyl_mape_net_R1"] or np.nan for R in Rs], "s-", color=COL["full"], lw=2, label="cylinder model, R known")
    ax[1].set_xticks(xs); ax[1].set_xticklabels(labs); ax[1].set_xlabel("limb radius (mm)"); ax[1].set_ylabel("MAPE (%) on photos with marker"); ax[1].legend(fontsize=8)
    ax[1].set_title("Curvature-aware correction removes the curvature failure")
    for fc, ls in zip([0.8, 0.9, 1.0, 1.1, 1.2], [":", "--", "-", "--", ":"]):
        ax[2].plot(xs, [summ[str(R)][f"cyl_mape_net_R{fc:g}"] or np.nan for R in Rs], ls, marker="o", ms=3, label=f"R estimate x{fc:g}")
    ax[2].set_xticks(xs); ax[2].set_xticklabels(labs); ax[2].set_xlabel("true limb radius (mm)"); ax[2].set_ylabel("MAPE (%)"); ax[2].legend(fontsize=7.5)
    ax[2].set_title("Sensitivity to an error in the supplied radius")
    save(fig, "25_curvature_correction.png")


# ----------------------------------------------------------------------------- 3. uncertainty
def mc_samples(img, P4, corners, cal, rng, n=30):
    """Monte-Carlo area samples: corner jitter (0.3 px) x mask variation (flip-TTA, threshold 0.45-0.55, 1 px dilate/erode)."""
    out = []
    k3 = np.ones((3, 3), np.uint8)
    for _ in range(n):
        p = P4[rng.integers(len(P4))]
        m = (p > rng.uniform(0.45, 0.55)).astype(np.uint8)   # 0.35-0.65 was tried first: on low-confidence images it swallows the whole skin
        r = rng.random()
        if r < .25:
            m = cv2.dilate(m, k3)
        elif r < .5:
            m = cv2.erode(m, k3)
        if m.sum() < 5:
            continue
        out.append(measure_full_from(corners + rng.normal(0, 0.3, corners.shape).astype(np.float32), m, cal))
    return np.array(out)


def tta4(model_, X):
    """Probabilities for the 4 flip variants (undone), each (N,H,W)."""
    outs = []
    for fl in [(), (2,), (1,), (1, 2)]:       # dims of NHW (1 = H, 2 = W)
        Xf = np.flip(X, axis=[d for d in fl]) if fl else X       # X is N,H,W,3 -> same axes as H=1, W=2
        P, _ = S.predict(model_, np.ascontiguousarray(Xf))
        outs.append(np.flip(P, axis=[d for d in fl]) if fl else P)
    return np.stack(outs, 1)


def unc_table(items, label, cal, n_mc=30, seed=0):
    imgs, gts, M, P = render_items(items)
    P4 = tta4(model(), np.stack(imgs))
    rng = np.random.default_rng(seed); rows = []
    for k, it in enumerate(items):
        c = G.detect_marker(imgs[k])
        row = dict(idx=k, gt_area=it["scene"]["area"], tilt_true=it["cam"]["tilt"], found=c is not None)
        if c is not None:
            row["a_hat"] = measure_full_from(c, M[k], cal)
            s = mc_samples(imgs[k], P4[k], c, cal, rng, n_mc)
            row["mc_mean"], row["mc_sd"] = float(s.mean()), float(s.std(ddof=1))
            row["tilt_est"] = est_tilt(c, cal)
        rows.append(row)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, f"ext_unc_{label}.csv"), index=False)
    return df


def conformal_q(scores, alpha):
    n = len(scores); k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(np.sort(scores)[min(k, n) - 1])


def stage_unc():
    print("[unc]")
    cal = calib()
    t0 = time.time()
    cal_df = unc_table(build_items(150, 7001, tilt_max=45, curved=0.15), "calibration", cal, seed=11)
    test_df = unc_table(test_items(), "test", cal, seed=12)
    print(f"   MC tables done in {time.time() - t0:.0f}s")
    cf, tf = cal_df[cal_df.found].copy(), test_df[test_df.found].copy()
    main = pd.read_csv(os.path.join(RES, "main.csv"))
    assert np.allclose(tf.a_hat.values, main.net_full[main.marker_found].values, rtol=1e-9), "A_hat should equal main.csv net_full"
    out = {"n_calibration": int(len(cf)), "n_test": int(len(tf))}
    for alpha in (0.10, 0.05):
        res = {}
        # (a) raw MC interval: a_hat +/- z sigma
        z = 1.645 if alpha == 0.10 else 1.96
        lo, hi = tf.a_hat - z * tf.mc_sd, tf.a_hat + z * tf.mc_sd
        res["mc_raw"] = dict(coverage=float(((tf.gt_area >= lo) & (tf.gt_area <= hi)).mean() * 100), mean_rel_width=float(((hi - lo) / tf.a_hat).mean() * 100))
        # (b) split-conformal, global relative score |A_hat - A| / A_hat
        sc = (cf.a_hat - cf.gt_area).abs() / cf.a_hat; q = conformal_q(sc.values, alpha)
        lo, hi = tf.a_hat * (1 - q), tf.a_hat * (1 + q)
        res["conformal_global"] = dict(q=q, coverage=float(((tf.gt_area >= lo) & (tf.gt_area <= hi)).mean() * 100), mean_rel_width=float(2 * q * 100))
        # (c) Mondrian conformal: separate quantile for small / large estimated area
        cov, wid = [], []
        for sm in (True, False):
            cs = cf[(cf.a_hat < SMALL) == sm]; ts = tf[(tf.a_hat < SMALL) == sm]
            if len(cs) < 5 or len(ts) == 0:
                continue
            qq = conformal_q(((cs.a_hat - cs.gt_area).abs() / cs.a_hat).values, alpha)
            cov += list(((ts.gt_area >= ts.a_hat * (1 - qq)) & (ts.gt_area <= ts.a_hat * (1 + qq))).values); wid += [2 * qq * 100] * len(ts)
            res[f"mondrian_{'small' if sm else 'large'}_q"] = qq
        res["conformal_mondrian"] = dict(coverage=float(np.mean(cov) * 100), mean_rel_width=float(np.mean(wid)))
        # (d) MC-normalised conformal: score |A_hat - A| / sigma_MC  (adaptive width)
        sn = ((cf.a_hat - cf.gt_area).abs() / cf.mc_sd.clip(lower=1e-6)).values; qn = conformal_q(sn, alpha)
        lo, hi = tf.a_hat - qn * tf.mc_sd, tf.a_hat + qn * tf.mc_sd
        res["conformal_mc_normalised"] = dict(q=qn, coverage=float(((tf.gt_area >= lo) & (tf.gt_area <= hi)).mean() * 100), mean_rel_width=float(((hi - lo) / tf.a_hat).mean() * 100))
        out[f"alpha_{alpha}"] = res
    # reliability flag from information available at deployment time
    def flag(d):
        return (d.tilt_est > 50) | (d.a_hat < SMALL)
    tf["flag"] = flag(tf); tf["abs_err"] = (tf.a_hat - tf.gt_area).abs() / tf.gt_area * 100
    out["flag"] = dict(rule=f"estimated tilt > 50 deg OR estimated area < {SMALL:.0f} mm2 (marker missing = cannot measure)",
                       n_flagged=int(tf.flag.sum()), mape_flagged=float(tf.abs_err[tf.flag].mean()) if tf.flag.any() else None,
                       mape_unflagged=float(tf.abs_err[~tf.flag].mean()), mape_all=float(tf.abs_err.mean()),
                       marker_missing=int((~test_df.found).sum()), tilt_est_vs_true_mae=float((tf.tilt_est - tf.tilt_true).abs().mean()))
    out["mc_sd_vs_abs_error_spearman"] = float(stats.spearmanr(tf.mc_sd / tf.a_hat, tf.abs_err)[0])
    json.dump(out, open(os.path.join(RES, "ext_uncertainty.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
    # fig 26
    a10 = out["alpha_0.1"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    names = {"mc_raw": "MC only", "conformal_global": "conformal\n(global)", "conformal_mondrian": "conformal\n(size classes)", "conformal_mc_normalised": "conformal\n(MC-normalised)"}
    xs = np.arange(len(names))
    ax[0].bar(xs, [a10[k]["coverage"] for k in names], color=["#8a8a8a", "#3b7dd8", "#7a4fd6", COL["full"]]); ax[0].axhline(90, color="k", ls="--")
    ax[0].set_xticks(xs); ax[0].set_xticklabels(names.values(), fontsize=8); ax[0].set_ylabel("coverage of the true area (%)"); ax[0].set_ylim(0, 105)
    ax[0].set_title("Interval coverage (target 90%)")
    for i, k in enumerate(names):
        ax[0].text(i, a10[k]["coverage"] + 1.5, f"{a10[k]['coverage']:.0f}%\nwidth ±{a10[k]['mean_rel_width'] / 2:.0f}%", ha="center", fontsize=7.5)
    qn = a10["conformal_mc_normalised"]["q"]
    o = np.argsort(tf.gt_area.values)[::2][:45]; sub = tf.iloc[o]
    ax[1].errorbar(np.arange(len(sub)), sub.a_hat, yerr=qn * sub.mc_sd, fmt="o", ms=3, color=COL["full"], ecolor="#9ad8bd", capsize=2, label="estimate ± 90% interval")
    ax[1].plot(np.arange(len(sub)), sub.gt_area, "k_", ms=7, label="true area"); ax[1].set_xlabel("test photos (sorted by true area)"); ax[1].set_ylabel("area (mm²)")
    ax[1].legend(fontsize=8); ax[1].set_title("Reported area with uncertainty")
    ax[2].scatter(tf.mc_sd / tf.a_hat * 100, tf.abs_err, s=14, c=np.where(tf.flag, "#c0392b", COL["full"]))
    ax[2].set_xlabel("Monte-Carlo spread (% of estimate)"); ax[2].set_ylabel("actual absolute error (%)")
    ax[2].set_title(f"Spread tracks error (Spearman {out['mc_sd_vs_abs_error_spearman']:.2f}); red = flagged")
    save(fig, "26_uncertainty.png")


# ----------------------------------------------------------------------------- 4. tilt x distance grid
def stage_grid():
    print("[grid]")
    tilts, Ds, n = [0, 15, 30, 45, 60], [170, 200, 260, 320], 12
    rows = []
    for t in tilts:
        for D in Ds:
            items = build_items(n, 9000 + 10 * t + D, tilt_max=0, curved=0, tilt=float(t), D=float(D))
            imgs, gts, M, _ = render_items(items)
            for k, it in enumerate(items):
                c = G.detect_marker(imgs[k]); row = dict(tilt=t, D=D, found=c is not None, gt_area=it["scene"]["area"], dice=S.dice_iou(M[k], gts[k])[0])
                row["pixel"] = G.pipeline_measure(imgs[k], M[k], calib(), "pixel")["area"]
                row["full"] = G.pipeline_measure(imgs[k], M[k], calib(), "full")["area"] if c is not None else np.nan
                rows.append(row)
        print("   tilt", t, flush=True)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, "ext_grid.csv"), index=False)
    for m in ("pixel", "full"):
        df[f"e_{m}"] = (df[m] - df.gt_area).abs() / df.gt_area * 100
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.9))
    for a, (col, ttl, cm, vmax) in zip(ax, [("e_pixel", "Pixels only: MAPE (%)", "magma_r", 80), ("e_full", "Proposed: MAPE (%)", "magma_r", 80), ("found", "Marker found (%)", "viridis", 100)]):
        Z = np.array([[df[(df.tilt == t) & (df.D == D)][col].mean() * (100 if col == "found" else 1) for D in Ds] for t in tilts])
        im = a.imshow(Z, cmap=cm, vmin=0, vmax=vmax, origin="lower", aspect="auto")
        a.set_xticks(range(len(Ds))); a.set_xticklabels(Ds); a.set_yticks(range(len(tilts))); a.set_yticklabels(tilts)
        a.set_xlabel("camera distance (mm)"); a.set_ylabel("camera tilt (°)"); a.set_title(ttl); a.grid(False)
        for i in range(len(tilts)):
            for j in range(len(Ds)):
                a.text(j, i, f"{Z[i, j]:.0f}", ha="center", va="center", fontsize=8, color="w" if (Z[i, j] > vmax * .55) != (col == "found") else "k")
        fig.colorbar(im, ax=a, fraction=.046)
    save(fig, "27_tilt_distance_heatmap.png")


# ----------------------------------------------------------------------------- 5. curves, real gallery, domain randomisation
def stage_curves():
    print("[curves]")
    h = {t: json.load(open(os.path.join(RES, f"history_{t}.json"))) for t in ("unet", "unet_bce", "real_scratch", "real_finetune") if os.path.exists(os.path.join(RES, f"history_{t}.json"))}
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.9))
    for t, c in (("unet", "#e08a1e"), ("unet_bce", COL["full"])):
        if t in h:
            ax[0].plot(h[t]["epoch"], h[t]["train_loss"], color=c, label=f"{t} (train)"); ax[1].plot(h[t]["epoch"], h[t]["val_dice"], color=c, label=t)
    ax[0].set_title("Simulated data: training loss"); ax[1].set_title("Simulated data: validation Dice"); ax[0].set_xlabel("epoch"); ax[1].set_xlabel("epoch"); ax[0].legend(fontsize=8); ax[1].legend(fontsize=8)
    for t, c in (("real_scratch", "#3b7dd8"), ("real_finetune", "#7a4fd6")):
        if t in h:
            ax[2].plot(h[t]["epoch"], h[t]["val_dice"], color=c, marker="o", ms=3, label=t)
    ax[2].set_title("Real photos: selection-set Dice"); ax[2].set_xlabel("epoch"); ax[2].legend(fontsize=8)
    save(fig, "28_training_curves.png")
    # real gallery
    from . import domrand
    Xt, Yt, te = domrand.real_test_split()
    m = S.load_model("real_scratch"); P, Mk = S.predict(m, Xt)
    d = np.array([S.dice_iou(Mk[k], Yt[k])[0] for k in range(len(Yt))]); o = np.argsort(-d)
    pick = list(o[:3]) + list(o[len(o) // 2 - 1:len(o) // 2 + 2]) + list(o[-3:])
    fig, ax = plt.subplots(3, 3, figsize=(9.6, 9.6))
    for a, k in zip(ax.ravel(), pick):
        im = rgb(Xt[k]).copy()
        for msk, col in ((Yt[k], (0, 255, 0)), (Mk[k], (255, 0, 0))):
            cs, _ = cv2.findContours(msk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(im, cs, -1, col, 2)
        a.imshow(im); a.axis("off"); a.set_title(f"Dice {d[k]:.2f}", fontsize=9)
    fig.suptitle("Real held-out photos, best / median / worst (green = true outline, red = model)", y=.93, fontweight="bold")
    save(fig, "29_real_gallery.png")
    # domain-randomisation comparison
    rd = json.load(open(os.path.join(RES, "real_data.json")))
    dr = json.load(open(os.path.join(RES, "ext_domrand.json"))) if os.path.exists(os.path.join(RES, "ext_domrand.json")) else {}
    labels, vals = [], []
    for key, lab in (("synthetic-only", "simulated only\n(paper)"), ("DR", "simulated +\ndomain randomisation"), ("synth→real", "sim → fine-tune\non real"), ("real-scratch", "real only")):
        v = dr["dr_unet"]["dice"] if key == "DR" and "dr_unet" in dr else (rd[key]["dice"] if key in rd else None)
        if v is not None:
            labels.append(lab); vals.append(v)
    fig, ax = plt.subplots(figsize=(7.2, 4))
    ax.bar(range(len(vals)), vals, color=["#8a8a8a", COL["full"], "#7a4fd6", "#3b7dd8"][:len(vals)])
    for i, v in enumerate(vals):
        ax.text(i, v + .01, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels, fontsize=8.5); ax.set_ylabel("Dice on 537 real test photos"); ax.set_ylim(0, .7)
    ax.set_title("Zero-shot transfer to real photos")
    save(fig, "30_domain_randomisation.png")


def stage_estR():
    """Negative result kept on record: estimating the limb radius from the 4 marker corners alone is not identifiable."""
    print("[estR]")
    cal = calib(); out = {}
    for R in (200, 120, 80):
        items = build_items(16, 5000 + R + 15, tilt_max=0, curved=0, tilt=15.0, R=float(R))
        ea, Re = [], []
        for it in items:
            img, m, _, a = sim.render(it["scene"], it["cam"])
            c = CV.detect_marker_tolerant(img)
            if c is None:
                continue
            r = CV.measure_cylinder_estR(img, m, cal, corners=c)
            if r["ok"]:
                ea.append((r["area"] - a) / a * 100); Re.append(r["R_est"])
        out[str(R)] = dict(n=len(ea), mape=float(np.mean(np.abs(ea))), bias=float(np.mean(ea)), R_est_median=float(np.median(Re)),
                           R_est_q25=float(np.percentile(Re, 25)), R_est_q75=float(np.percentile(Re, 75)))
    json.dump(out, open(os.path.join(RES, "ext_curvature_estR.json"), "w"), indent=1); print(json.dumps(out, indent=1))


STAGES = dict(estR=stage_estR, metrics=stage_metrics, curv=stage_curv, unc=stage_unc, grid=stage_grid, curves=stage_curves)

if __name__ == "__main__":
    for s in (sys.argv[1:] or list(STAGES)):
        STAGES[s]()
