"""Real longitudinal photos: LUTSeg (CC BY 4.0, 141 smartphone photos of leprosy ulcers, 39 patients, visits T_1..T_n).

No scale marker exists in these photos, so only the RELATIVE wound size (wound pixels / photo pixels) can be followed over visits, and only
when the camera distance stays about the same.  This module (1) scores our outline model against the clinician wound masks and
(2) checks how well the model's relative-size trend matches the clinicians' trend.

    python -m wound_cv.lutseg eval        # -> results/ext_lutseg_*.csv / .json
"""
import glob, json, os, re, sys
import cv2
import numpy as np
import pandas as pd

from . import segment as S, sim

ROOT = S.ROOT
DATA = os.path.join(ROOT, "data", "LUTSeg")
RES = os.path.join(ROOT, "results")
TOL = 10.0     # % : changes smaller than this are called "no clear change"


def verdict(dA, tol=TOL):
    return "healing" if dA > tol else ("worsening" if dA < -tol else "stalled")


def load_index():
    rows = [json.loads(l) for l in open(os.path.join(DATA, "metadata.jsonl"), encoding="utf-8")]
    out = []
    for r in rows:
        m = re.search(r"T_?(\d+)$", r["image_id"])
        out.append(dict(patient=r["patient_id"], visit=int(m.group(1)), image=os.path.join(DATA, r["image_file_name"]),
                        mask=os.path.join(DATA, r["wound_mask_file_name"]), split=r["split"]))
    return pd.DataFrame(out).sort_values(["patient", "visit"], key=lambda c: c.str.extract(r"(\d+)")[0].astype(int) if c.name == "patient" else c).reset_index(drop=True)


def load_pair(row, size=sim.IMG):
    im = cv2.imread(row["image"]); mk = cv2.imread(row["mask"], cv2.IMREAD_UNCHANGED)
    if mk.ndim == 3: mk = mk[..., 0]
    gt = (mk > 0).astype(np.uint8)
    if gt.shape != im.shape[:2]: gt = cv2.resize(gt, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST)
    return cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA), cv2.resize(gt, (size, size), interpolation=cv2.INTER_NEAREST), im.shape[:2]


def evaluate(tags=("real_scratch", "real_finetune", "unet_bce")):
    idx = load_index(); X, Y, shp = [], [], []
    for _, r in idx.iterrows():
        a, b, s = load_pair(r); X.append(a); Y.append(b); shp.append(s)
    X, Y = np.stack(X), np.stack(Y)
    idx["gt_frac"] = Y.reshape(len(Y), -1).mean(1) * 100
    idx["h"], idx["w"] = zip(*shp)
    out = {}
    for tag in tags:
        P, M = S.predict(S.load_model(tag), X)
        idx[f"{tag}_frac"] = M.reshape(len(M), -1).mean(1) * 100
        d = np.array([S.dice_iou(M[k], Y[k]) for k in range(len(Y))]); idx[f"{tag}_dice"] = d[:, 0]
        out[tag] = dict(dice_mean=float(d[:, 0].mean()), dice_median=float(np.median(d[:, 0])), iou_mean=float(d[:, 1].mean()))
    # relative-size trend per patient: change of the visit-1 area, in %, clinicians vs model
    rows = []
    for p, g in idx.groupby("patient"):
        g = g.sort_values("visit")
        for tag in ("gt",) + tuple(tags):
            f = g[f"{tag}_frac"].values
            if f[0] <= 0: continue
            for v, fr in zip(g.visit.values, f):
                rows.append(dict(patient=p, visit=v, who=tag, frac=fr, dA=(f[0] - fr) / f[0] * 100))
    tr = pd.DataFrame(rows)
    for tag in tags:
        a = tr[(tr.who == "gt") & (tr.visit > 1)].set_index(["patient", "visit"]).dA
        b = tr[(tr.who == tag) & (tr.visit > 1)].set_index(["patient", "visit"]).dA
        j = pd.concat([a, b], axis=1, keys=["gtv", "mv"]).dropna()
        last = j.groupby(level=0).tail(1)
        out[tag].update(n_comparisons=int(len(j)), corr_dA=float(np.corrcoef(j["gtv"], j["mv"])[0, 1]), mae_dA_pp=float((j["gtv"] - j["mv"]).abs().mean()),
                        verdict_agreement_last_visit=float((last["gtv"].map(verdict) == last["mv"].map(verdict)).mean() * 100), n_patients=int(len(last)))
    idx.to_csv(os.path.join(RES, "ext_lutseg_per_image.csv"), index=False); tr.to_csv(os.path.join(RES, "ext_lutseg_trends.csv"), index=False)
    json.dump(out, open(os.path.join(RES, "ext_lutseg_eval.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
    return idx, tr, out


if __name__ == "__main__":
    if (sys.argv[1:] or ["eval"])[0] == "eval":
        evaluate()
