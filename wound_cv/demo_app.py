"""Live demo (desktop window, no extra installs: uses Tkinter + OpenCV already in the project).

    python -m wound_cv.demo_app                 # open the demo window
    python -m wound_cv.demo_app --make-target   # write the printable marker + known-area shapes page (demo_target.png)
    python -m wound_cv.demo_app --selftest      # headless check of every demo code path

Tab 1  Simulated photo   : renders a wound with the sliders' tilt / distance / curvature and measures it by every method
                           next to the EXACT true area, with uncertainty and a reliability flag.
Tab 2  Real wound photo  : runs the segmentation network on a real photo (e.g. from real_data/demo_candidates) and shows the
                           predicted outline against the clinician mask, with Dice / IoU.   (outline only: no marker, no mm^2)
Tab 3  Printed target    : photograph the printed page from demo_target.png with a phone/webcam; the marker homography
                           measures the printed shapes, whose true areas are known exactly.  (real camera => homography only,
                           the simulated lens calibration does not apply)
"""
import base64, glob, json, os, re, sys, time
import cv2
import numpy as np

from . import sim, geometry as G, segment as S, curvature as CV
from .experiments import RES, ROOT, MODE_NAME

SHAPES = None        # filled by target_shapes()
PX_MM = 300 / 25.4   # printable page resolution (300 dpi)


# ----------------------------------------------------------------------------- printable target
def target_shapes():
    """Known shapes, positions in mm in the marker frame (origin = marker top-left, x right, y down)."""
    global SHAPES
    if SHAPES is None:
        blob = sim.Wound(np.random.default_rng(5), 900.0)
        pts, area = blob.poly(1.0, 360)
        SHAPES = [dict(name="rectangle 30 x 20 mm", kind="rect", c=(85.0, 15.0), wh=(30.0, 20.0), area=600.0),
                  dict(name="ellipse 30 x 20 mm", kind="ellipse", c=(85.0, 55.0), ab=(15.0, 10.0), area=float(np.pi * 15 * 10)),
                  dict(name="irregular blob", kind="poly", c=(30.0, 80.0), pts=pts, area=float(area))]
    return SHAPES


def make_target(path="demo_target.png"):
    W, H = int(210 * PX_MM), int(297 * PX_MM)
    page = np.full((H, W, 3), 255, np.uint8)
    ox, oy = 25.0, 40.0                                  # marker top-left on the page (mm)
    P = lambda x, y: (int(round((ox + x) * PX_MM)), int(round((oy + y) * PX_MM)))
    m = int(round(sim.MARKER_MM * PX_MM))
    mk = cv2.aruco.generateImageMarker(sim.ARUCO, sim.MARKER_ID, m, borderBits=1)
    x0, y0 = P(0, 0)
    page[y0:y0 + m, x0:x0 + m] = mk[..., None]
    red = (30, 30, 215)
    for s in target_shapes():
        cx, cy = s["c"]
        if s["kind"] == "rect":
            a, b = P(cx - s["wh"][0] / 2, cy - s["wh"][1] / 2), P(cx + s["wh"][0] / 2, cy + s["wh"][1] / 2)
            cv2.rectangle(page, a, b, red, -1)
        elif s["kind"] == "ellipse":
            cv2.ellipse(page, P(cx, cy), (int(round(s["ab"][0] * PX_MM)), int(round(s["ab"][1] * PX_MM))), 0, 0, 360, red, -1, cv2.LINE_AA)
        else:
            cv2.fillPoly(page, [np.array([P(cx + x, cy + y) for x, y in s["pts"]], np.int32)], red, cv2.LINE_AA)
    cv2.putText(page, "PRINT AT 100% (no 'fit to page'). The black marker must measure 40 mm.", (int(12 * PX_MM), int(15 * PX_MM)), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(page, "Keep the whole sheet flat and in the photo.", (int(12 * PX_MM), int(24 * PX_MM)), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.line(page, P(0, 100), P(100, 100), (0, 0, 0), 3); cv2.putText(page, "100 mm ruler check", P(0, 108), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imwrite(path, page)
    json.dump([dict(name=s["name"], area_mm2=s["area"], centre_mm=s["c"]) for s in target_shapes()], open(os.path.splitext(path)[0] + "_truth.json", "w"), indent=1)
    return page


def measure_target(img):
    """Measure the red printed shapes in a photo of the target page.  Returns list of dict, annotated image."""
    corners = G.detect_marker(img)
    out = img.copy()
    if corners is None:
        return None, out
    H = G.homography_to_mm(corners)
    cv2.polylines(out, [corners.astype(np.int32)], True, (0, 255, 0), 2)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    red = (((hsv[..., 0] < 10) | (hsv[..., 0] > 168)) & (hsv[..., 1] > 110) & (hsv[..., 2] > 60)).astype(np.uint8)
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(red, 8)
    exp = target_shapes(); res = []
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < 150:
            continue
        mask = (lab == i).astype(np.uint8)
        c_mm = cv2.perspectiveTransform(cen[i].reshape(1, 1, 2).astype(np.float32), H).ravel()
        d = [np.hypot(c_mm[0] - s["c"][0], c_mm[1] - s["c"][1]) for s in exp]
        j = int(np.argmin(d))
        if d[j] > 20:
            continue
        a = G.measure(mask, H)["area"]
        res.append(dict(name=exp[j]["name"], est=a, true=exp[j]["area"], err=(a - exp[j]["area"]) / exp[j]["area"] * 100))
        cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(out, cs, -1, (255, 0, 0), 2)
        cv2.putText(out, f"{a:.0f} mm2", tuple(int(v) for v in cen[i]), cv2.FONT_HERSHEY_SIMPLEX, max(0.5, img.shape[1] / 1600), (255, 255, 255), 2, cv2.LINE_AA)
    return res, out


# ----------------------------------------------------------------------------- simulated tab backend
_state = {}


def calib_cached():
    if "cal" not in _state:
        j = json.load(open(os.path.join(RES, "calibration.json")))
        K = np.array([[j["f"], 0, j["cx"]], [0, j["f"], j["cy"]], [0, 0, 1]], np.float64)
        _state["cal"] = dict(K=K, dist=np.array([j["k1"], 0, 0, 0, 0], np.float64))
    return _state["cal"]


def model_cached(tag):
    if tag not in _state:
        _state[tag] = S.load_model(tag)
    return _state[tag]


def interval_q():
    p = os.path.join(RES, "ext_uncertainty.json")
    if os.path.exists(p):
        return json.load(open(p))["alpha_0.1"]["conformal_mc_normalised"]["q"]
    return None


def run_sim(seed, tilt, D, R, light=1.0):
    """Render + measure one simulated photo.  R=None for flat skin.  Returns dict with images and numbers."""
    cal = calib_cached()
    rng = np.random.default_rng(seed)
    sc = sim.make_scene(int(rng.integers(1 << 30)))
    cam = sim.random_cam(rng, tilt_max=0, curved=0, light_strength=light)
    cam.update(tilt=float(tilt), D=float(D), R=(None if R is None else float(R)))
    img, gt, _, true_a = sim.render(sc, cam)
    P, M = S.predict(model_cached("unet_bce"), img[None])
    mask = M[0]
    res = dict(img=img, gt=gt, mask=mask, true=true_a, dice=S.dice_iou(mask, gt)[0], modes={})
    corners = G.detect_marker(img)
    res["corners"] = corners
    for mode in ("pixel", "scale", "homog", "full"):
        r = G.pipeline_measure(img, mask, cal, mode)
        res["modes"][mode] = r["area"] if r["ok"] else np.nan
    if R is not None and corners is None:
        c2 = CV.detect_marker_tolerant(img)
        if c2 is not None:
            res["corners"] = c2; corners = c2
    if R is not None and corners is not None:
        res["modes"]["cyl"] = CV.measure_cylinder(img, mask, cal, float(R), corners=corners)["area"]
    # uncertainty (MC, conformal-normalised) + reliability flag
    res["interval"] = None; res["flags"] = []
    if corners is None:
        res["flags"].append("marker not found: cannot measure")
    else:
        from .extensions import mc_samples, tta4, est_tilt
        P4 = tta4(model_cached("unet_bce"), img[None])[0]
        s = mc_samples(img, P4, corners, cal, np.random.default_rng(0), 30)
        q = interval_q()
        a_hat = res["modes"]["full"]
        if q is not None and len(s) > 3 and R is None:      # interval is calibrated on flat skin only
            res["interval"] = (a_hat - q * s.std(ddof=1), a_hat + q * s.std(ddof=1))
        t_est = est_tilt(corners, cal)
        if t_est > 50: res["flags"].append(f"steep angle ({t_est:.0f} deg)")
        if a_hat < 250: res["flags"].append("small wound (<250 mm2)")
        if R is not None and R < 150: res["flags"].append("curved limb: planar methods unreliable, use curvature-aware (interval calibrated for flat skin only, not shown)")
    return res


def annotate_sim(res):
    im = res["img"].copy()
    big = cv2.resize(im, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    def draw(mask, col):
        cs, _ = cv2.findContours(cv2.resize(mask, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(big, cs, -1, col, 2)
    draw(res["gt"], (255, 255, 255)); draw(res["mask"], (60, 60, 255))
    if res["corners"] is not None:
        cv2.polylines(big, [(res["corners"] * 2).astype(np.int32)], True, (0, 255, 0), 2)
    return big


# ----------------------------------------------------------------------------- healing verdict
def heal_verdict(dA_last, tol=10.0):
    """Plain-language call from the last-visit healing percentage (threshold = about twice the 4.8 pp error measured)."""
    return "HEALING" if dA_last > tol else ("WORSENING" if dA_last < -tol else "STALLED / no clear change")


# ----------------------------------------------------------------------------- real tab backend
def run_real(path, tag):
    img = cv2.imread(path)
    m = re.search(r"wound_main-(\d+)\.jpg", os.path.basename(path))
    gt = None
    if m:
        mp = os.path.join(ROOT, "real_data", "mendeley", "wound_mask", f"wound_mask-{m.group(1)}.jpg")
        if os.path.exists(mp):
            g = cv2.imread(mp, 0); gt = (cv2.resize(g, (S.sim.IMG, S.sim.IMG), interpolation=cv2.INTER_LINEAR) > 127).astype(np.uint8)
    x = cv2.resize(img, (sim.IMG, sim.IMG), interpolation=cv2.INTER_AREA)
    P, M = S.predict(model_cached(tag), x[None])
    big = cv2.resize(x, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    def draw(mask, col):
        cs, _ = cv2.findContours(cv2.resize(mask, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(big, cs, -1, col, 2)
    draw(M[0], (60, 60, 255))
    d = i = None
    if gt is not None:
        draw(gt, (0, 255, 0)); d, i = S.dice_iou(M[0], gt)
    return big, d, i


# ----------------------------------------------------------------------------- GUI
def png_data(img, max_side=560):
    h, w = img.shape[:2]; s = max_side / max(h, w)
    if s != 1: img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    return base64.b64encode(cv2.imencode(".png", img)[1].tobytes())


def build_gui(auto_close_ms=None):
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    root = tk.Tk(); root.title("Wound measurement demo"); root.geometry("1280x760")
    nb = ttk.Notebook(root); nb.pack(fill="both", expand=True)
    COLS = {"pixel": "#8a8a8a", "scale": "#e08a1e", "homog": "#3b7dd8", "full": "#1f9d6b", "cyl": "#7a4fd6"}

    # ---------------- tab 1
    t1 = ttk.Frame(nb); nb.add(t1, text="1  Simulated photo (exact truth)")
    ctrl = ttk.Frame(t1); ctrl.pack(side="left", fill="y", padx=8, pady=8)
    v_tilt, v_dist, v_R, v_curved, v_seed = tk.DoubleVar(value=25), tk.DoubleVar(value=240), tk.DoubleVar(value=90), tk.BooleanVar(value=False), tk.IntVar(value=1)
    for lab, var, lo, hi in (("camera tilt (deg)", v_tilt, 0, 60), ("camera distance (mm)", v_dist, 170, 320), ("limb radius (mm)", v_R, 55, 200)):
        ttk.Label(ctrl, text=lab).pack(anchor="w"); tk.Scale(ctrl, variable=var, from_=lo, to=hi, orient="horizontal", length=220, resolution=1).pack()
    ttk.Checkbutton(ctrl, text="curved limb (use radius)", variable=v_curved).pack(anchor="w", pady=4)
    img_lab1 = ttk.Label(t1); img_lab1.pack(side="left", padx=6)
    cnv = tk.Canvas(t1, width=520, height=560, bg="white", highlightthickness=0); cnv.pack(side="left", padx=6)

    def go1(new=False):
        if new: v_seed.set(int(np.random.randint(1, 10**6)))
        t0 = time.time()
        R = float(v_R.get()) if v_curved.get() else None
        res = run_sim(v_seed.get(), v_tilt.get(), v_dist.get(), R)
        ph = tk.PhotoImage(data=png_data(annotate_sim(res))); img_lab1.configure(image=ph); img_lab1.image = ph
        cnv.delete("all"); cnv.create_text(10, 10, anchor="nw", font=("Segoe UI", 12, "bold"), text=f"True area {res['true']:.0f} mm²   ·   U-Net Dice {res['dice']:.3f}")
        names = ["pixel", "scale", "homog", "full"] + (["cyl"] if "cyl" in res["modes"] else [])
        mx = max([res["true"]] + [v for v in res["modes"].values() if np.isfinite(v)]) * 1.15
        y = 50; X0, W = 170, 330
        cnv.create_line(X0 + W * res["true"] / mx, 40, X0 + W * res["true"] / mx, 50 + 52 * len(names), fill="black", dash=(4, 3), width=2)
        for k in names:
            v = res["modes"][k]; lab = {"cyl": "curvature-aware"}.get(k, MODE_NAME.get(k, k)); lab = lab.replace("+ ", "")
            cnv.create_text(8, y + 12, anchor="w", text=lab, font=("Segoe UI", 9), width=155)
            if np.isfinite(v):
                cnv.create_rectangle(X0, y, X0 + W * v / mx, y + 24, fill=COLS[k], outline="")
                cnv.create_text(X0 + W * v / mx + 4, y + 12, anchor="w", text=f"{v:.0f}  ({(v - res['true']) / res['true'] * 100:+.0f}%)", font=("Segoe UI", 9, "bold"))
            else:
                cnv.create_text(X0 + 4, y + 12, anchor="w", text="marker not found", fill="#c0392b")
            y += 52
        y += 8
        if res["interval"] is not None:
            lo, hi = res["interval"]; ok = lo <= res["true"] <= hi
            cnv.create_text(10, y, anchor="nw", font=("Segoe UI", 10, "bold"), text=f"90% interval (proposed): {lo:.0f} – {hi:.0f} mm²   {'contains truth' if ok else 'MISSES truth'}", fill="#1f7a3f" if ok else "#c0392b"); y += 24
        for f in res["flags"]:
            cnv.create_text(10, y, anchor="nw", font=("Segoe UI", 10), text="⚠ " + f, fill="#c0392b"); y += 20
        cnv.create_text(10, 540, anchor="nw", font=("Segoe UI", 8), fill="#555", text=f"white = true outline, red = U-Net, green = marker · {time.time() - t0:.1f}s · device {S.DEVICE}")
    ttk.Button(ctrl, text="Measure with these settings", command=go1).pack(fill="x", pady=(10, 2))
    ttk.Button(ctrl, text="New random wound", command=lambda: go1(True)).pack(fill="x", pady=2)

    # ---------------- tab 2
    t2 = ttk.Frame(nb); nb.add(t2, text="2  Real wound photo (outline)")
    c2 = ttk.Frame(t2); c2.pack(side="left", fill="y", padx=8, pady=8)
    tags = [os.path.splitext(os.path.basename(p))[0] for p in sorted(glob.glob(os.path.join(ROOT, "models", "*.pt")))]
    v_tag = tk.StringVar(value="real_scratch" if "real_scratch" in tags else tags[0])
    ttk.Label(c2, text="model").pack(anchor="w"); ttk.Combobox(c2, textvariable=v_tag, values=tags, state="readonly", width=22).pack()
    lab2 = ttk.Label(t2); lab2.pack(side="left", padx=6)
    txt2 = tk.Label(t2, text="Open a real photo.\nIf it is an original wound_main-XXXX.jpg\n(or a demo_candidates file) its clinician mask\nis found automatically.", justify="left", font=("Segoe UI", 11)); txt2.pack(side="left", padx=10, anchor="n", pady=20)

    def go2(path=None):
        path = path or filedialog.askopenfilename(initialdir=os.path.join(ROOT, "real_data", "mendeley", "wound_main"), filetypes=[("images", "*.jpg *.jpeg *.png")])
        if not path: return
        base = os.path.basename(path); m = re.search(r"wound_main-\d+\.jpg", base)
        if m and m.group(0) != base:      # a demo_candidates composite (e.g. 0.93_wound_main-0919.jpg) -> the original dataset photo
            path = os.path.join(ROOT, "real_data", "mendeley", "wound_main", m.group(0))
        big, d, i = run_real(path, v_tag.get())
        ph = tk.PhotoImage(data=png_data(big)); lab2.configure(image=ph); lab2.image = ph
        txt2.configure(text=(f"{os.path.basename(path)}\n\nDice {d:.3f}\nIoU  {i:.3f}\n\nred = model, green = clinician mask\n\nThese photos have no marker,\nso only the outline is scored (no mm²)." if d is not None
                             else f"{os.path.basename(path)}\n\nred = model prediction\n(no ground-truth mask found)"))
    ttk.Button(c2, text="Open real photo…", command=go2).pack(fill="x", pady=10)

    # ---------------- tab 3
    t3 = ttk.Frame(nb); nb.add(t3, text="3  Printed target (real camera)")
    c3 = ttk.Frame(t3); c3.pack(side="left", fill="y", padx=8, pady=8)
    lab3 = ttk.Label(t3); lab3.pack(side="left", padx=6)
    txt3 = tk.Label(t3, text="1) python -m wound_cv.demo_app --make-target\n2) print demo_target.png at 100% scale\n3) photograph it at an angle, then open the photo.", justify="left", font=("Segoe UI", 11)); txt3.pack(side="left", padx=10, anchor="n", pady=20)

    def show3(img):
        res, out = measure_target(img)
        ph = tk.PhotoImage(data=png_data(out)); lab3.configure(image=ph); lab3.image = ph
        if res is None:
            txt3.configure(text="Marker not found.\nMake sure the black marker is fully visible and in focus."); return
        lines = ["Marker homography, real camera (no lens correction)\n"] + [f"{r['name']}:\n   measured {r['est']:.0f} mm²   true {r['true']:.0f} mm²   ({r['err']:+.1f}%)" for r in res]
        txt3.configure(text="\n".join(lines) if res else "Marker found, but no red shapes recognised.")

    def go3():
        p = filedialog.askopenfilename(filetypes=[("images", "*.jpg *.jpeg *.png")])
        if p: show3(cv2.imread(p))

    def cam3():
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        ok, frame = cap.read(); cap.release()
        if not ok: messagebox.showerror("Webcam", "Could not read the webcam."); return
        show3(frame)
    ttk.Button(c3, text="Open photo of the printed page…", command=go3).pack(fill="x", pady=4)
    ttk.Button(c3, text="Snapshot from webcam", command=cam3).pack(fill="x", pady=4)

    root.after(200, go1)
    if auto_close_ms: root.after(auto_close_ms, root.destroy)
    return root, dict(go1=go1, go2=go2, show3=show3)


# ----------------------------------------------------------------------------- self-test
def selftest():
    print("[selftest] simulated, flat"); r = run_sim(3, 25, 240, None); print("  true %.0f  modes %s  interval %s  flags %s" % (r["true"], {k: round(v) for k, v in r["modes"].items()}, r["interval"], r["flags"]))
    print("[selftest] simulated, curved R=90"); r = run_sim(4, 20, 240, 90); print("  true %.0f  modes %s  flags %s" % (r["true"], {k: round(v) for k, v in r["modes"].items() if np.isfinite(v)}, r["flags"]))
    page = make_target(os.path.join(ROOT, "demo_target.png"))
    src = np.array([[0, 0], [page.shape[1], 0], [page.shape[1], page.shape[0]], [0, page.shape[0]]], np.float32)
    for name, dst in (("frontal", [[60, 40], [900, 50], [905, 1230], [55, 1220]]), ("tilted ~35 deg", [[140, 80], [820, 40], [900, 1180], [40, 1240]])):
        ph = cv2.warpPerspective(page, cv2.getPerspectiveTransform(src, np.array(dst, np.float32)), (960, 1280), borderValue=(90, 90, 90))
        ph = cv2.GaussianBlur(ph, (0, 0), 1.0); ph = np.clip(ph.astype(np.float32) + np.random.default_rng(0).normal(0, 3, ph.shape), 0, 255).astype(np.uint8)
        res, _ = measure_target(ph)
        print(f"[selftest] printed target, {name}:", None if res is None else [(x["name"], round(x["est"]), round(x["true"]), f"{x['err']:+.1f}%") for x in res])
    cand = sorted(glob.glob(os.path.join(ROOT, "real_data", "demo_candidates", "0.9*_wound_main-*.jpg")))
    if cand:
        orig = os.path.join(ROOT, "real_data", "mendeley", "wound_main", re.search(r"wound_main-\d+\.jpg", cand[0]).group(0))
        for tag in ("real_scratch", "unet_bce"):
            big, d, i = run_real(orig, tag); print(f"[selftest] real photo {os.path.basename(orig)} {tag}: Dice {d:.3f}")
    root, fns = build_gui(); fns["go1"](True); root.update(); root.destroy(); print("[selftest] GUI built and tab 1 ran")


if __name__ == "__main__":
    if "--make-target" in sys.argv:
        make_target("demo_target.png"); print("wrote demo_target.png and demo_target_truth.json")
    elif "--selftest" in sys.argv:
        selftest()
    else:
        build_gui()[0].mainloop()
