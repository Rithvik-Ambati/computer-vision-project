"""Browser demo (Gradio) of the same pipeline as wound_cv/demo_app.py.

    python -m wound_cv.gradio_app            # local only:  http://127.0.0.1:7860
    python -m wound_cv.gradio_app --share    # also creates a temporary public https://xxxx.gradio.live link (valid ~1 week
                                             # or until you stop the program; anyone with the link can use the demo)

Tab 1  Simulated wound   : sliders -> renders a photo with EXACT true area, measures it by every method, interval + flags
Tab 2  Real wound photo  : upload a real photo -> U-Net outline (+ Dice if a clinician mask is known)
Tab 4  Healing           : 3-6 real follow-up photos of one wound (LUTSeg style) -> relative size trend + verdict
Tab 3  Photo with marker : upload a photo that contains the 40 mm ArUco marker
                           (a) printed target page photographed with a real camera, or (b) a simulated wound photo
"""
import glob, json, os, re, sys
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

from . import sim, geometry as G, segment as S, curvature as CV, demo_app as D
from .experiments import ROOT, MODE_NAME
from .extensions import mc_samples, tta4, est_tilt
from .experiments import to_frame, contour_of

COL = {"pixel": "#8a8a8a", "scale": "#e08a1e", "homog": "#3b7dd8", "full": "#1f9d6b", "cyl": "#7a4fd6"}
LABEL = dict(MODE_NAME, cyl="Curvature-aware (radius given)")


def bgr2rgb(im):
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def bar_plot(modes, true=None):
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    names = [k for k in ("pixel", "scale", "homog", "full", "cyl") if k in modes and np.isfinite(modes[k])]
    vals = [modes[k] for k in names]
    ax.barh(range(len(names)), vals, color=[COL[k] for k in names])
    ax.set_yticks(range(len(names))); ax.set_yticklabels([LABEL[k] for k in names], fontsize=8); ax.invert_yaxis()
    if true is not None:
        ax.axvline(true, color="k", ls="--", lw=1.5); ax.text(true, -0.6, f"true {true:.0f} mm²", ha="center", fontsize=8)
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.0f}" + (f" ({(v - true) / true * 100:+.0f}%)" if true else ""), va="center", fontsize=8)
    ax.set_xlabel("estimated area (mm²)"); ax.set_xlim(0, max(vals + [true or 0]) * 1.3 if vals else 1)
    ax.spines[["top", "right"]].set_visible(False); fig.tight_layout()
    return fig


def summary_md(modes, true, dice, interval, flags, extra=""):
    lines = []
    if true is not None:
        lines.append(f"**True area: {true:.0f} mm²**" + (f" · U-Net Dice {dice:.3f}" if dice is not None else ""))
    lines.append("")
    lines.append("| method | area (mm²) | error |\n|---|---|---|")
    for k in ("pixel", "scale", "homog", "full", "cyl"):
        if k in modes:
            v = modes[k]
            lines.append(f"| {LABEL[k]} | " + (f"{v:.0f} | " + (f"{(v - true) / true * 100:+.1f}%" if true else "–") if np.isfinite(v) else "marker not found | –") + " |")
    if interval is not None:
        lo, hi = interval
        lines.append(f"\n**90% interval (proposed method): {lo:.0f} – {hi:.0f} mm²**" + (f" — {'contains' if lo <= true <= hi else 'MISSES'} the true area" if true else ""))
    for f in flags:
        lines.append(f"\n⚠ {f}")
    if extra:
        lines.append("\n" + extra)
    return "\n".join(lines)


# ---------------------------------------------------------------- tab 1
def tab_sim(tilt, dist, curved, radius, seed, new):
    seed = int(np.random.randint(1, 10 ** 6)) if new else int(seed)
    R = float(radius) if curved else None
    res = D.run_sim(seed, tilt, dist, R)
    return (bgr2rgb(D.annotate_sim(res)), bar_plot(res["modes"], res["true"]),
            summary_md(res["modes"], res["true"], res["dice"], res["interval"], res["flags"],
                       "white = true outline · red = U-Net · green = detected marker"), seed)


# ---------------------------------------------------------------- tab 2
def tab_real(path, tag, mask_img):
    if path is None:
        return None, "Upload a real wound photo."
    name = os.path.basename(path)
    img = cv2.imread(path)
    gt = None
    if mask_img is not None:
        g = cv2.cvtColor(mask_img, cv2.COLOR_RGB2GRAY) if mask_img.ndim == 3 else mask_img
        gt = (cv2.resize(g, (sim.IMG, sim.IMG), interpolation=cv2.INTER_LINEAR) > 127).astype(np.uint8)
    else:
        m = re.search(r"wound_main-(\d+)\.jpg", name)
        mp = os.path.join(ROOT, "real_data", "mendeley", "wound_mask", f"wound_mask-{m.group(1)}.jpg") if m else None
        if mp and os.path.exists(mp):
            gt = (cv2.resize(cv2.imread(mp, 0), (sim.IMG, sim.IMG), interpolation=cv2.INTER_LINEAR) > 127).astype(np.uint8)
    x = cv2.resize(img, (sim.IMG, sim.IMG), interpolation=cv2.INTER_AREA)
    P, M = S.predict(D.model_cached(tag), x[None])
    big = cv2.resize(x, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    def draw(mask, col):
        cs, _ = cv2.findContours(cv2.resize(mask, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(big, cs, -1, col, 2)
    draw(M[0], (60, 60, 255))
    if gt is not None:
        draw(gt, (0, 255, 0)); d, i = S.dice_iou(M[0], gt)
        txt = f"**Dice {d:.3f} · IoU {i:.3f}** (model `{tag}`)\n\nred = model, green = clinician mask."
    else:
        txt = f"Model `{tag}` outline in red. No clinician mask found for this file, so no Dice is shown."
    return bgr2rgb(big), txt + "\n\nThese photos contain no marker, so only the **outline** is evaluated (no mm²). Segmentation of real photos is the weak point of the project (mean Dice ≈ 0.50)."


# ---------------------------------------------------------------- tab 3
def tab_marker(img_rgb, source, radius):
    if img_rgb is None:
        return None, None, "Upload a photo that contains the 40 mm marker."
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    if source.startswith("Printed"):
        res, out = D.measure_target(img)
        if res is None:
            return bgr2rgb(out), None, "Marker not found. Make sure the black marker is fully visible and in focus."
        rows = ["**Marker homography, real camera** (no lens correction: the lens calibration belongs to the simulated camera)\n",
                "| shape | measured (mm²) | true (mm²) | error |\n|---|---|---|---|"]
        rows += [f"| {r['name']} | {r['est']:.0f} | {r['true']:.0f} | {r['err']:+.1f}% |" for r in res]
        return bgr2rgb(out), None, "\n".join(rows) if res else "Marker found, but no red shapes recognised."
    # simulated wound photo
    if img.shape[:2] != (sim.IMG, sim.IMG):
        img = cv2.resize(img, (sim.IMG, sim.IMG), interpolation=cv2.INTER_AREA)
    cal = D.calib_cached()
    P, M = S.predict(D.model_cached("unet_bce"), img[None]); mask = M[0]
    modes = {m: (lambda r: r["area"] if r["ok"] else np.nan)(G.pipeline_measure(img, mask, cal, m)) for m in ("pixel", "scale", "homog", "full")}
    corners = G.detect_marker(img)
    R = float(radius) if radius and radius > 0 else None
    if corners is None and R is not None:
        corners = CV.detect_marker_tolerant(img)
    if R is not None and corners is not None:
        modes["cyl"] = CV.measure_cylinder(img, mask, cal, R, corners=corners)["area"]
    interval, flags = None, []
    if corners is None:
        flags.append("marker not found: cannot measure")
    else:
        s = mc_samples(img, tta4(D.model_cached("unet_bce"), img[None])[0], corners, cal, np.random.default_rng(0), 30)
        q = D.interval_q(); a_hat = modes["full"] if np.isfinite(modes["full"]) else modes.get("cyl", np.nan)
        if q is not None and len(s) > 3 and np.isfinite(a_hat) and R is None:
            interval = (a_hat - q * s.std(ddof=1), a_hat + q * s.std(ddof=1))
        t_est = est_tilt(corners, cal)
        if t_est > 50: flags.append(f"steep angle ({t_est:.0f}°)")
        if np.isfinite(a_hat) and a_hat < 250: flags.append("small wound (<250 mm²)")
        if R is not None:
            flags.append("curved limb: use the curvature-aware row; the 90% interval is calibrated for flat skin only, so it is not shown")
    big = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    cs, _ = cv2.findContours(cv2.resize(mask, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(big, cs, -1, (60, 60, 255), 2)
    if corners is not None:
        cv2.polylines(big, [(corners * 2).astype(np.int32)], True, (0, 255, 0), 2)
    return bgr2rgb(big), bar_plot(modes), summary_md(modes, None, None, interval, flags, "red = U-Net outline · green = detected marker. The true area of an uploaded photo is not known to the app; for the supplied example photos it is written in the file name (e.g. `true907`).")


_LUT_CACHE = {}


def _lut_model(path):
    """Fine-tuned LUTSeg model. A photo from a known LUTSeg patient is scored by the fold model that never trained on that patient."""
    m = re.search(r"P(\d+)_?T", os.path.basename(path)) or re.search(r"Patient_(\d+)", path.replace("\\", "/"))
    tag = "lutseg_all"
    if m:
        fl = json.load(open(os.path.join(ROOT, "results", "ext_lutseg_folds.json")))["folds"]
        for k, g in enumerate(fl):
            if f"Patient_{m.group(1)}" in g: tag = f"lutseg_fold{k}"
    if tag not in _LUT_CACHE: _LUT_CACHE[tag] = S.load_model(tag)
    return tag, _LUT_CACHE[tag]


def _mask_lookup():
    if "mask" not in _LUT_CACHE:
        from . import lutseg as L
        _LUT_CACHE["mask"] = {os.path.basename(r["image"]): r["mask"] for _, r in L.load_index().iterrows()} if os.path.isdir(L.DATA) else {}
    return _LUT_CACHE["mask"]


def _visit_no(p):
    m = re.search(r"T_?(\d+)\.", os.path.basename(p))
    return (int(m.group(1)) if m else 0, os.path.basename(p))


def tab_visits_real(files):
    """Real follow-up photos of one wound (no simulation, no scale marker): outline each photo, follow the RELATIVE size, give a verdict."""
    from . import lutseg as L
    paths = [getattr(f, "name", f) for f in (files or [])]
    if len(paths) < 3:
        return None, None, "Upload at least 3 real photos of the same wound (visits in time order; files named ...T1, T2, ... are sorted by visit number)."
    paths = sorted(paths, key=_visit_no); n = len(paths)
    gal, frac, gtf, shapes = [], [], [], set()
    for i, p in enumerate(paths):
        im = cv2.imread(p); shapes.add(im.shape[:2]); tag, model = _lut_model(p)
        x = cv2.resize(im, (sim.IMG, sim.IMG), interpolation=cv2.INTER_AREA)
        _, M = S.predict(model, x[None]); m = M[0]
        frac.append(float(m.mean() * 100)); o = bgr2rgb(x).copy()
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(o, cs, -1, (255, 255, 0), 2)
        cap = f"visit {i + 1}: model {frac[-1]:.1f}% of frame"
        mp = _mask_lookup().get(os.path.basename(p))
        if mp and os.path.exists(mp):
            g = (cv2.imread(mp, cv2.IMREAD_UNCHANGED) > 0).astype(np.uint8)
            if g.ndim == 3: g = g[..., 0]
            g = cv2.resize(g, (sim.IMG, sim.IMG), interpolation=cv2.INTER_NEAREST); gtf.append(float(g.mean() * 100))
            cs, _ = cv2.findContours(g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(o, cs, -1, (0, 255, 0), 2)
            cap += f" · clinician {gtf[-1]:.1f}%"
        gal.append((o, cap))
    frac = np.array(frac); gtf = np.array(gtf) if len(gtf) == n else None
    if frac[0] <= 0: return gal, None, "No wound was found in the first photo."
    dA = (frac[0] - frac) / frac[0] * 100; x = np.arange(n); labels = [f"visit {i + 1}" for i in range(n)]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(x, frac, "o--", color=COL["full"], lw=2, label="model outline")
    if gtf is not None: ax[0].plot(x, gtf, "k-", lw=2, alpha=.5, label="clinician outline")
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels); ax[0].set_ylabel("wound / frame (%)"); ax[0].set_title("Relative wound size"); ax[0].legend(fontsize=8)
    wd = .35
    if gtf is not None and gtf[0] > 0: ax[1].bar(x - wd / 2, (gtf[0] - gtf) / gtf[0] * 100, wd, color="#444", label="clinician")
    ax[1].bar(x + (wd / 2 if gtf is not None else 0), dA, wd, color=COL["full"], label="model")
    ax[1].axhline(0, color="k", lw=.8); ax[1].set_xticks(x); ax[1].set_xticklabels(labels); ax[1].set_title("Change vs visit 1 (+ = smaller)"); ax[1].legend(fontsize=8); fig.tight_layout()
    md = [f"### Interpretation: **{L.verdict(dA[-1])}** (relative size change {dA[-1]:+.0f} % between visit 1 and visit {n})"]
    st = np.diff(frac); md.append(f"Visit-to-visit: {int((st < 0).sum())} of {len(st)} steps got smaller, {int((st > 0).sum())} got larger.")
    if gtf is not None and gtf[0] > 0:
        dg = (gtf[0] - gtf[-1]) / gtf[0] * 100; md.append(f"Clinician outlines say: **{L.verdict(dg)}** ({dg:+.0f} %). " + ("The model agrees." if L.verdict(dg) == L.verdict(dA[-1]) else "**The model disagrees here.**"))
    if len(shapes) > 1: md.append("**Warning: the photos have different sizes/crops, so wound-as-fraction-of-frame is not comparable. Treat the verdict as unreliable.**")
    md.append("\nThere is **no scale marker** in these photos, so only the size relative to the frame is available (not mm²), and it is valid only if the camera distance stayed about the same. "
              "A change under ±10 % is called *stalled*. Yellow = model outline, green = clinician outline. Real smartphone photos of leprosy ulcers (LUTSeg, CC BY 4.0). "
              "Photos of patients in the dataset are scored by a model fold that never trained on that patient.")
    return gal, fig, "\n".join(md)


def build():
    tags = [os.path.splitext(os.path.basename(p))[0] for p in sorted(glob.glob(os.path.join(ROOT, "models", "*.pt")))]
    with gr.Blocks(title="Wound measurement demo") as demo:
        gr.Markdown("# Geometry-aware wound measurement\nMarker homography + lens calibration + U-Net. Tab 1 and the simulated part of tab 3 use a **simulator with exact ground truth**; real photos (tab 2) have outlines only.")
        with gr.Tab("1 · Simulated wound (exact truth)"):
            with gr.Row():
                with gr.Column(scale=1):
                    tilt = gr.Slider(0, 60, 25, step=1, label="camera tilt (°)")
                    dist = gr.Slider(170, 320, 240, step=5, label="camera distance (mm)")
                    curved = gr.Checkbox(False, label="curved limb")
                    rad = gr.Slider(55, 200, 90, step=5, label="limb radius (mm)")
                    seed = gr.Number(1, label="seed (same seed = same wound)", precision=0)
                    b1 = gr.Button("Measure with these settings", variant="primary"); b2 = gr.Button("New random wound")
                with gr.Column(scale=2):
                    im1 = gr.Image(label="photo", height=420)
                with gr.Column(scale=2):
                    pl1 = gr.Plot(label="area by method"); md1 = gr.Markdown()
            b1.click(lambda *a: tab_sim(*a, False), [tilt, dist, curved, rad, seed], [im1, pl1, md1, seed])
            b2.click(lambda *a: tab_sim(*a, True), [tilt, dist, curved, rad, seed], [im1, pl1, md1, seed])
        with gr.Tab("2 · Real wound photo (outline)"):
            with gr.Row():
                with gr.Column():
                    up2 = gr.Image(type="filepath", label="real wound photo (e.g. wound_main-0919.jpg)")
                    mk2 = gr.Image(label="optional: clinician mask", image_mode="L")
                    tg = gr.Dropdown(tags, value="real_scratch" if "real_scratch" in tags else tags[0], label="model")
                    b3 = gr.Button("Segment", variant="primary")
                with gr.Column():
                    im2 = gr.Image(label="result"); md2 = gr.Markdown()
            b3.click(tab_real, [up2, tg, mk2], [im2, md2])
        with gr.Tab("3 · Photo with the 40 mm marker"):
            with gr.Row():
                with gr.Column():
                    src = gr.Radio(["Printed target page (real camera, homography only)", "Simulated wound photo (U-Net + lens calibration)"], value="Simulated wound photo (U-Net + lens calibration)", label="what is in the photo?")
                    up3 = gr.Image(label="photo")
                    rad3 = gr.Number(0, label="limb radius in mm (0 = flat; simulated photos only)")
                    b4 = gr.Button("Measure", variant="primary")
                with gr.Column():
                    im3 = gr.Image(label="result"); pl3 = gr.Plot(); md3 = gr.Markdown()
            b4.click(tab_marker, [up3, src, rad3], [im3, pl3, md3])
        with gr.Tab("4 · Healing over visits (real photos)"):
            gr.Markdown("Is the wound healing? Upload **3-6 real photos of the same wound taken at different visits** (time order; `..._T1, _T2, ...` names are sorted automatically). "
                        "The model outlines the wound in each photo and follows its size relative to the frame. No simulation, no printed pages.")
            with gr.Row():
                with gr.Column(scale=1):
                    hup = gr.File(file_count="multiple", file_types=["image"], label="visit photos (3-6)")
                    h3 = gr.Button("Interpret the visits", variant="primary")
                with gr.Column(scale=3):
                    hmd = gr.Markdown(); hgal = gr.Gallery(label="visits (yellow = model, green = clinician)", columns=4, height=300); hpl = gr.Plot()
            h3.click(tab_visits_real, [hup], [hgal, hpl, hmd])
    return demo


if __name__ == "__main__":
    build().launch(share="--share" in sys.argv, server_name="127.0.0.1", server_port=7860)
