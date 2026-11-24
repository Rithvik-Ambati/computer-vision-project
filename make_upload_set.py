"""Builds demo_upload/: ready-made files for the Gradio demo (real photos + masks, simulated marker photos, printed-target photos).
Selection rules are fixed in advance (not by error): real = 3 best, 2 median, 1 worst test photos by Dice; simulated = frontal, 40 deg tilt,
curved R=120, smallest wound among photos whose marker is found."""
import csv, os, shutil, numpy as np, cv2
from wound_cv import sim, geometry as G, demo_app as D
from wound_cv.experiments import build_items, ROOT
OUT = os.path.join(ROOT, "demo_upload"); shutil.rmtree(OUT, ignore_errors=True)
for d in ("real_photos", "real_photos/masks", "marker_photos"): os.makedirs(os.path.join(OUT, d))
# --- real photos (held-out test split), ranked by Dice (ranking.csv from make_demo_set.py)
rk = list(csv.DictReader(open(os.path.join(ROOT, "real_data", "demo_candidates", "ranking.csv"))))
n = len(rk); pick = [rk[0], rk[1], rk[2], rk[n // 2], rk[n // 2 + 1], rk[-1]]
for r in pick:
    f = r["file"]; num = f.split("-")[1]
    shutil.copy(os.path.join(ROOT, "real_data", "mendeley", "wound_main", f), os.path.join(OUT, "real_photos", f))
    shutil.copy(os.path.join(ROOT, "real_data", "mendeley", "wound_mask", "wound_mask-" + num), os.path.join(OUT, "real_photos", "masks", "wound_mask-" + num))
    print("real", f, "Dice", r["dice"])
# --- simulated photos with a marker (true area in the file name)
def save(name, it):
    img, m, _, a = sim.render(it["scene"], it["cam"]); fn = f"{name}_true{int(round(a))}.png"
    cv2.imwrite(os.path.join(OUT, "marker_photos", fn), img); print("sim", fn, "marker", G.detect_marker(img) is not None); return fn
items = build_items(150, 1, tilt_max=45, curved=0.15)
ok = [i for i, it in enumerate(items) if G.detect_marker(sim.render(it["scene"], it["cam"])[0]) is not None and it["cam"]["R"] is None]
save("sim_frontal", items[min(ok, key=lambda i: items[i]["cam"]["tilt"])])
save("sim_tilt40", items[min(ok, key=lambda i: abs(items[i]["cam"]["tilt"] - 40))])
save("sim_smallest_wound", items[min(ok, key=lambda i: items[i]["scene"]["area"])])
cur = build_items(40, 5120 + 15, tilt_max=0, curved=0, tilt=15.0, R=120.0)
okc = [it for it in cur if G.detect_marker(sim.render(it["scene"], it["cam"])[0]) is not None]
save("sim_curved_R120_enter120", okc[0])
# --- printed target (digitally warped picture of the page: a stand-in until you photograph a real print)
page = D.make_target(os.path.join(OUT, "marker_photos", "printed_target_PRINT_THIS_AT_100pct.png"))
src = np.array([[0, 0], [page.shape[1], 0], [page.shape[1], page.shape[0]], [0, page.shape[0]]], np.float32)
for name, dst in (("printed_target_frontal", [[60, 40], [900, 50], [905, 1230], [55, 1220]]), ("printed_target_tilted", [[140, 80], [820, 40], [900, 1180], [40, 1240]])):
    ph = cv2.warpPerspective(page, cv2.getPerspectiveTransform(src, np.array(dst, np.float32)), (960, 1280), borderValue=(90, 90, 90))
    cv2.imwrite(os.path.join(OUT, "marker_photos", name + "_digital_stand_in.jpg"), cv2.GaussianBlur(ph, (0, 0), 1.0))
