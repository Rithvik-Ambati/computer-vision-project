"""Assemble results/*.csv|json + figs/*.png into one self-contained report.html"""
import base64, json, os
import numpy as np
import pandas as pd

from . import sim
from .experiments import RES, FIGS, ROOT, MODES, MODE_NAME

def img(name, cap=""):
    p = os.path.join(FIGS, name)
    if not os.path.exists(p): return ""
    b = base64.b64encode(open(p, "rb").read()).decode()
    return f'<figure><img src="data:image/png;base64,{b}" alt="{name}"><figcaption>{cap}</figcaption></figure>'

def J(name, default=None):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else default

def table(rows, head):
    h = "".join(f"<th>{x}</th>" for x in head)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<div class='tw'><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>"

def err_stats(df, src, m):
    s = df if m == "pixel" else df[df.marker_found]
    e = (s[f"{src}_{m}"] - s.gt_area) / s.gt_area * 100
    return np.nanmean(np.abs(e)), np.nanmedian(np.abs(e)), np.nanmean(e), np.nanpercentile(np.abs(e), 95)

def build():
    main = pd.read_csv(os.path.join(RES, "main.csv"))
    cal = J("calibration.json", {})
    parts = []
    # ---------- headline numbers
    net = {m: err_stats(main, "net", m) for m in MODES}; orc = {m: err_stats(main, "gt", m) for m in MODES}
    det = main.marker_found.mean() * 100
    rep = J("repeatability.json", {}); heal = J("healing_mae.json", {}); flow = J("flow_iou.json", {})
    reg = pd.read_csv(os.path.join(RES, "registration.csv")) if os.path.exists(os.path.join(RES, "registration.csv")) else None
    hist = J("history_unet.json"); hist0 = J("history_unet_bce.json")
    kpis = [("U-Net test Dice", f"{main.dice.mean():.3f}"), ("Area MAPE · proposed", f"{net['full'][0]:.1f}%"),
            ("Area MAPE · pixels only", f"{net['pixel'][0]:.1f}%"), ("Marker detected", f"{det:.0f}%"),
            ("Repeatability CV · proposed", f"{rep.get('full', float('nan')):.1f}%"), ("Healing ΔA error", f"{heal.get('full', float('nan')):.1f} pp")]
    kp = "".join(f"<div class='kpi'><div class='v'>{v}</div><div class='l'>{k}</div></div>" for k, v in kpis)

    def sec(title, body, note=""):
        parts.append(f"<section><h2>{title}</h2>{body}{('<p class=note>' + note + '</p>') if note else ''}</section>")

    sec("What was built", f"""<p>The paper proposed a five-stage pipeline; this project implements and tests it end to end on a physically
    grounded phantom simulator (exact ground truth for area, masks and image-to-image geometry). Methodology changes after review:</p>
    <ol>
    <li><b>Planar assumption stated and stress-tested</b> on cylinder-wrapped skin (limb radius sweep).</li>
    <li><b>ArUco marker</b> (40 mm, auto-detected, same plane as the wound) replaces the vague “printed marker or ruler”.</li>
    <li><b>Calibration clarified</b>: checkerboard calibration estimates focal length and lens distortion; the marker homography handles pose and scale. They do different jobs.</li>
    <li><b>SIFT/RANSAC re-scoped</b>: not needed for area (the marker frame already gives metric area) — used only for cross-visit boundary overlay, and compared head-to-head with marker-frame alignment.</li>
    <li><b>Optical flow narrowed</b> to one concrete job — propagating a mask through a short burst — and compared against doing nothing and against a homography.</li>
    <li><b>Segmentation fully specified</b>: U-Net (4 levels, base 12), BCE + λ·Dice, Adam 1e-4, cosine schedule, flip/photometric augmentation, held-out test scenes, λ ablation. <b>Result: λ = 0 (BCE only) won on validation</b> — the Dice term made the network over-segment (predicted area 13 % too large), so the final model is BCE-only. This contradicts the paper's Eq. (1) and should be reported as such.</li>
    <li><b>Contour-free area</b>: mask warped into a metric canvas and pixels counted (no staircase bias); perimeter reported separately as a digital perimeter.</li>
    <li><b>Ablation + statistics</b>: pixels → marker scale → homography → lens correction; MAPE, bias, Bland–Altman, repeatability CV, sweeps over tilt / distance / curvature / lighting.</li>
    </ol>""")
    # data splits + leakage checks
    def leak():
        out = []
        try:
            tr = np.load(os.path.join(ROOT, "data", "train_320.npz"))["a"]; va = np.load(os.path.join(ROOT, "data", "val_48.npz"))["a"]
            te = main.gt_area.values
            r = lambda x: set(np.round(x, 6))
            out.append(("simulated: train ∩ validation", len(r(tr) & r(va))))
            out.append(("simulated: train ∩ test", len(r(tr) & r(te))))
            out.append(("simulated: validation ∩ test", len(r(va) & r(te))))
        except Exception as e:
            out.append((f"simulated check failed ({e})", -1))
        try:
            import cv2
            z = np.load(os.path.join(ROOT, "data", "mendeley_all.npz")); X = z["x"]
            def dh(im):
                g = cv2.cvtColor(cv2.resize(im, (9, 8), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
                return (g[:, 1:] > g[:, :-1]).tobytes()
            perm = np.random.default_rng(0).permutation(len(X)); n_te, n_sel = int(.2 * len(X)), int(.1 * len(X))
            te_i, fit_i = perm[:n_te], perm[n_te + n_sel:]
            fit_h = {dh(X[i]) for i in fit_i}
            dup = sum(dh(X[i]) in fit_h for i in te_i)
            out.append(("real: held-out photos whose perceptual hash equals a training photo", dup))
        except Exception:
            pass
        return out
    sp = J("real_split.json")
    rows = [["simulated · train", "320", "seed 1000, fits U-Net weights"], ["simulated · validation", "48", "seed 2000, checkpoint + loss-model selection"],
            ["simulated · test", f"{len(main)} + sweeps", "fresh seeds, never used for training or selection"]]
    if sp:
        rows += [["real · fit", str(sp["fit"]), "70 % of Mendeley wound photos"], ["real · checkpoint selection", str(sp["select"]), "10 %"],
                 ["real · held-out test", str(sp["test"]), "20 %, never used for training or selection"]]
    sec("Data splits & leakage checks", table(rows, ["split", "photos", "role"]) +
        table([[a, b] for a, b in leak()], ["check", "overlaps found (0 = clean)"]),
        "Caveats: simulated splits share one simulator, so they test generalisation to new simulated wounds only. The real dataset publishes no patient IDs, "
        "so near-duplicate photos of one patient could sit in different splits; the hash check only catches exact-looking duplicates. "
        "I saw the simulated test numbers for the first model before choosing the final loss; that choice was made on validation area bias alone.")
    # calibration
    sec("1 · Camera calibration", f"""{img('01_calibration.png', 'Checkerboard calibration recovers focal length and lens distortion.')}
    <p>Estimated f = <b>{cal.get('f', 0):.1f} px</b> (true {cal.get('f_true', 0):.0f}), k1 = <b>{cal.get('k1', 0):.3f}</b> (true {cal.get('k1_true', 0):.2f}),
    reprojection RMS {cal.get('rms', 0):.2f} px from {cal.get('used', 0)} views.</p>""")
    sec("2 · The phantom simulator", img("02_phantom_gallery.png", "Tilt, curvature and lighting are all rendered from exact geometry; the outline is the true wound boundary."))
    # worked examples
    ex = "".join(img(f"16_worked_example_{t}.png", f"Worked example {t}") for t in "ABC")
    sec("3 · Worked examples (one photo, step by step)", ex + "<p>Full numbers (marker corners, scale, homography matrix, pixel counts) are in <code>worked_examples.md</code>.</p>")
    # segmentation
    seg_rows = []
    if hist: seg_rows.append(["BCE + 1.0·Dice (proposed)", f"{max(hist['val_dice']):.3f}", f"{max(hist['val_iou']):.3f}"])
    if hist0: seg_rows.append(["BCE only (λ = 0)", f"{max(hist0['val_dice']):.3f}", f"{max(hist0['val_iou']):.3f}"])
    sec("4 · Segmentation (U-Net)", img("05_segmentation.png", "Held-out test photos: tilt up to 45°, 15 % on curved limbs, random lighting.") +
        img("15_training_curves.png", "Loss ablation from Eq. (1): adding the Dice term.") +
        (table(seg_rows, ["loss", "best val Dice", "best val IoU"]) if seg_rows else "") +
        f"<p>Test-set mean Dice <b>{main.dice.mean():.3f}</b>, IoU <b>{main.iou.mean():.3f}</b> (n = {len(main)}).</p>")
    real = J("real_data.json")
    if real:
        rr = [[k, f"{v['dice']:.3f}", f"{v['dice_median']:.3f}", f"{v['iou']:.3f}", f"{v['area_ratio_median']:.2f}"] for k, v in real.items()]
        sec("4b · Real wound photos (domain-gap test)", img("17_real_data.png", "Held-out real photos: expert mask (green) vs prediction (yellow).") +
            table(rr, ["model", "mean Dice", "median Dice", "mean IoU", "median pred/true area"]) +
            "<p>The model trained only on simulated wounds does not transfer to real photos. Training on real photos is needed for the segmentation stage; "
            "the geometric stages are unaffected by this. These photos carry no scale marker, so area in mm² is <b>not</b> evaluated on real data.</p>",
            "Dataset: Lower Limb and Feet Wound Image Dataset (Mendeley Data, DOI 10.17632/hsj38fwnvr, CC BY 4.0).")
    # ablation
    rows = [[MODE_NAME[m], f"{orc[m][0]:.1f}%", f"{net[m][0]:.1f}%", f"{net[m][1]:.1f}%", f"{net[m][2]:+.1f}%", f"{net[m][3]:.1f}%"] for m in MODES]
    sec("5 · Ablation — does each geometric stage matter?", img("03_ablation.png") +
        table(rows, ["method", "MAPE (perfect mask)", "MAPE (U-Net)", "median |err| (U-Net)", "bias (U-Net)", "95th pct |err|"]) +
        f"<p>Marker detected in {det:.0f}% of photos; error statistics for marker-based rows use detected photos only.</p>" +
        img("04_agreement.png", "Agreement with ground truth.") + img("06_perimeter.png"),
        "Pixel-only and scale-only are not strawmen: they are what you get from a photo if you ignore pose. The scale-only row can be worse than pixels-only because a tilted marker’s mean side is a biased scale estimate.")
    # sweeps
    sec("6 · Robustness sweeps", img("07_sweep_tilt.png", "Camera tilt") + img("08_sweep_distance.png", "Shooting distance") +
        img("09_sweep_curvature.png", "Surface curvature — the planar assumption.") + img("10_sweep_lighting.png", "Lighting (1.0 = training range)."))
    cv = "".join(f"<li>{MODE_NAME[m]}: CV {rep.get(m, float('nan')):.1f}%</li>" for m in MODES)
    sec("7 · Repeatability", img("11_repeatability.png") + f"<ul>{cv}</ul>")
    hrows = ""
    sec("8 · Healing progress (Eq. 2) & registration",
        img("12_healing.png") + f"<p>Mean absolute error of ΔA over visits: proposed <b>{heal.get('full', float('nan')):.1f}</b> percentage points; pixels-only <b>{heal.get('pixel', float('nan')):.1f}</b> pp.</p>" +
        img("13_registration.png") + "<p class=note>Note: most SIFT matches land on the marker's corners because the simulated skin has little texture, so SIFT's advantage over marker-only alignment is partly the marker itself; visit 3 had only 14 RANSAC inliers and a 2.5 mm error. Real skin texture is untested.</p>" +
        (table([[int(r.visit), int(r.matches), int(r.inliers), f"{r.sift_err_mm:.2f}", f"{r.marker_err_mm:.2f}"] for r in reg.itertuples()],
               ["visit", "SIFT matches", "RANSAC inliers", "SIFT error (mm)", "marker error (mm)"]) if reg is not None else ""))
    sec("9 · Optical flow, re-scoped", img("14_optical_flow.png") +
        "<p>Mean IoU of a propagated frame-0 mask over the burst: " + ", ".join(f"{k}: <b>{v:.2f}</b>" for k, v in flow.items()) + ".</p>")
    sec("10 · Honest limitations", """<ul>
    <li><b>Everything here is simulated.</b> The U-Net sees synthetic wounds; real data (Medetec, AZH, FUSeg) will have a domain gap. <code>segment.py</code> is written so real image/mask folders can be dropped in, but no real-wound accuracy is claimed.</li>
    <li><b>Planar homography breaks on curved limbs</b> — see the curvature sweep. A cylinder/mesh model or multi-marker approach is the next step.</li>
    <li>The marker must be visible and coplanar with the wound; detection drops at steep tilt.</li>
    <li>Perimeter is a digital perimeter and is noisier than area; it is reported but not the headline.</li>
    <li>Optical flow is not needed for area; its demonstrated value is mask propagation in a burst, under mild non-rigid motion.</li>
    <li>Lens is a single-parameter (k1) model; real phones also apply software distortion correction and autofocus breathing.</li></ul>""")
    html = f"""<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Wound Measurement Results</title><style>
:root{{--bg:#fff;--fg:#1a1d23;--mu:#5b6270;--card:#f5f6f8;--bd:#e1e4ea;--ac:#1f9d6b}}
@media (prefers-color-scheme:dark){{:root{{--bg:#14161a;--fg:#e8eaee;--mu:#9aa1ad;--card:#1d2026;--bd:#2c3038;--ac:#3ccf94}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,Segoe UI,sans-serif}}
main{{max-width:1080px;margin:0 auto;padding:0 16px 60px}}h1{{font-size:26px;margin:36px 0 4px}}h2{{margin:0 0 12px;font-size:19px}}
.sub{{color:var(--mu);margin-bottom:20px}}section{{background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:18px 20px;margin:18px 0}}
figure{{margin:12px 0}}img{{max-width:100%;height:auto;border-radius:8px;background:#fff}}figcaption{{color:var(--mu);font-size:13px;margin-top:4px}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:14px 0}}.kpi{{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:12px}}
.kpi .v{{font-size:24px;font-weight:700;color:var(--ac)}}.kpi .l{{color:var(--mu);font-size:12.5px}}
.tw{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:13.5px}}th,td{{padding:6px 10px;border-bottom:1px solid var(--bd);text-align:left;white-space:nowrap}}
.note{{color:var(--mu);font-size:13.5px}}code{{background:var(--bd);padding:1px 5px;border-radius:4px}}</style></head><body><main>
<h1>Geometry-aware wound measurement — results</h1><div class=sub>Simulated-phantom validation of the framework in <i>computer_vision_project_sem_7.pdf</i></div>
<div class=kpis>{kp}</div>{''.join(parts)}</main></body></html>"""
    open(os.path.join(ROOT, "report.html"), "w", encoding="utf-8").write(html)
    print("report.html written", len(html) // 1024, "KB")

if __name__ == "__main__":
    build()
