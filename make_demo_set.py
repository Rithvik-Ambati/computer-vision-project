"""Copies held-out TEST photos (same split as realdata.py) to real_data/demo_candidates/ with the model's overlay and Dice."""
import os, csv, shutil, numpy as np, cv2
from wound_cv import segment as S
from wound_cv.realdata import ROOT, DATA
X, Y = np.load(os.path.join(ROOT, "data", "mendeley_all.npz"))["x"], np.load(os.path.join(ROOT, "data", "mendeley_all.npz"))["y"]
files = sorted(os.listdir(os.path.join(DATA, "wound_main")))
te = np.random.default_rng(0).permutation(len(X))[:int(.2 * len(X))]
model = S.load_model("real_scratch")
P, M = S.predict(model, X[te])
out = os.path.join(ROOT, "real_data", "demo_candidates"); os.makedirs(out, exist_ok=True)
rows = []
for j, i in enumerate(te):
    d, iou = S.dice_iou(M[j], Y[i]); name = files[i]
    ov = X[i].copy(); ov[Y[i] > 0] = (0.5 * ov[Y[i] > 0] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
    cnt, _ = cv2.findContours(M[j].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(ov, cnt, -1, (0, 0, 255), 2)
    cv2.imwrite(os.path.join(out, f"{d:.2f}_{name}"), np.hstack([X[i], ov]))
    rows.append((name, round(float(d), 3), round(float(iou), 3)))
rows.sort(key=lambda r: -r[1])
csv.writer(open(os.path.join(out, "ranking.csv"), "w", newline="")).writerows([("file", "dice", "iou")] + rows)
print(len(rows), "test images; best", rows[0], "median", rows[len(rows)//2], "worst", rows[-1])
