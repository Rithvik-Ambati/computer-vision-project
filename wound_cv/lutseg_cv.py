"""Patient-level 5-fold cross-validation on LUTSeg (real follow-up photos).

For each fold the outline model (initialised from the Mendeley-trained real_scratch weights) is fine-tuned on the patients NOT in the fold,
with 4 of those patients held back for epoch selection, and then predicts the fold's patients.  Every LUTSeg photo is therefore scored
by a model that never saw that patient (out-of-fold).  A final model on all patients (minus 4 selection patients) is also saved for the app.

    python -m wound_cv.lutseg_cv train   # -> models/lutseg_fold{0-4}.pt, models/lutseg_all.pt, results/ext_lutseg_oof.csv
    python -m wound_cv.lutseg_cv eval    # trend statistics on same-framing sequences -> results/ext_lutseg_cv.json
"""
import json, os, sys
import numpy as np
import pandas as pd

from . import segment as S, realdata as RD
from . import lutseg as L

ROOT, RES = S.ROOT, S.RES if hasattr(S, "RES") else os.path.join(S.ROOT, "results")


def folds(patients, k=5, seed=0):
    rng = np.random.default_rng(seed); p = np.array(sorted(patients, key=lambda s: int(s.split("_")[1]))); rng.shuffle(p)
    return [list(p[i::k]) for i in range(k)]


def load_all():
    idx = L.load_index(); X, Y = [], []
    for _, r in idx.iterrows():
        a, b, _ = L.load_pair(r); X.append(a); Y.append(b)
    return idx, np.stack(X), np.stack(Y)


def train_cv(epochs=12, lr=1e-4):
    idx, X, Y = load_all()
    pats = sorted(idx.patient.unique(), key=lambda s: int(s.split("_")[1])); fl = folds(pats)
    idx["oof_frac"] = np.nan; idx["oof_dice"] = np.nan; idx["fold"] = -1
    for k, test_p in enumerate(fl):
        tr_p = [p for p in pats if p not in test_p]
        rng = np.random.default_rng(100 + k); sel_p = list(rng.choice(tr_p, 4, replace=False)); fit_p = [p for p in tr_p if p not in sel_p]
        fi, si, ti = (np.where(idx.patient.isin(g))[0] for g in (fit_p, sel_p, test_p))
        print(f"[fold {k}] fit {len(fi)} imgs / select {len(si)} / test {len(ti)} ({len(test_p)} patients)", flush=True)
        model = RD.train_real(X[fi], Y[fi], X[si], Y[si], f"lutseg_fold{k}", init="real_scratch", epochs=epochs, lr=lr, bs=8, lam=0.0, seed=k)
        P, M = S.predict(model, X[ti])
        idx.loc[ti, "oof_frac"] = M.reshape(len(M), -1).mean(1) * 100
        idx.loc[ti, "oof_dice"] = [S.dice_iou(M[j], Y[ti[j]])[0] for j in range(len(ti))]
        idx.loc[ti, "fold"] = k
    # final model for the app: all patients except 4 selection patients
    rng = np.random.default_rng(999); sel_p = list(rng.choice(pats, 4, replace=False)); fit_p = [p for p in pats if p not in sel_p]
    fi, si = (np.where(idx.patient.isin(g))[0] for g in (fit_p, sel_p))
    RD.train_real(X[fi], Y[fi], X[si], Y[si], "lutseg_all", init="real_scratch", epochs=epochs, lr=lr, bs=8, lam=0.0, seed=7)
    idx["gt_frac"] = Y.reshape(len(Y), -1).mean(1) * 100
    idx.to_csv(os.path.join(RES, "ext_lutseg_oof.csv"), index=False)
    json.dump(dict(folds=fl), open(os.path.join(RES, "ext_lutseg_folds.json"), "w"), indent=1)


def seq_stats(idx, col, tol):
    """Same-framing sequences only (>=3 visits, identical image size).  Compares relative change vs visit 1 with the clinicians."""
    cmp_rows, last_rows, step_rows = [], [], []
    for p, g in idx.groupby("patient"):
        g = g.sort_values("visit")
        if len(g) < 3 or g[["h", "w"]].drop_duplicates().shape[0] != 1 or g.gt_frac.iloc[0] <= 0 or g[col].iloc[0] <= 0: continue
        dg = (g.gt_frac.iloc[0] - g.gt_frac.values) / g.gt_frac.iloc[0] * 100; dm = (g[col].iloc[0] - g[col].values) / g[col].iloc[0] * 100
        for v in range(1, len(g)): cmp_rows.append((p, v + 1, dg[v], dm[v]))
        last_rows.append((p, dg[-1], dm[-1]))
        sg, sm = np.sign(np.diff(g.gt_frac.values)), np.sign(np.diff(g[col].values)); step_rows += list(sg == sm)
    c = pd.DataFrame(cmp_rows, columns=["p", "v", "g", "m"]); l = pd.DataFrame(last_rows, columns=["p", "g", "m"])
    return dict(n_sequences=int(len(l)), n_points=int(len(c)), corr=float(np.corrcoef(c["g"], c["m"])[0, 1]), mae_pp=float((c["g"] - c["m"]).abs().mean()),
                last_visit_verdict_agreement=float((l["g"].map(lambda x: L.verdict(x, tol)) == l["m"].map(lambda x: L.verdict(x, tol))).mean() * 100),
                step_direction_agreement=float(np.mean(step_rows) * 100), tol=tol)


def eval_cv():
    idx = pd.read_csv(os.path.join(RES, "ext_lutseg_oof.csv")); zs = pd.read_csv(os.path.join(RES, "ext_lutseg_per_image.csv"))
    assert (idx.image.values == zs.image.values).all()
    idx["h"], idx["w"] = zs["h"].values, zs["w"].values
    idx["zeroshot_frac"] =zs["real_scratch_frac"].values; idx["zeroshot_dice"] = zs["real_scratch_dice"].values
    out = dict(dice=dict(zero_shot_mendeley_model=float(idx.zeroshot_dice.mean()), fine_tuned_out_of_fold=float(idx.oof_dice.mean()),
                         fine_tuned_oof_median=float(idx.oof_dice.median())))
    for col, name in (("zeroshot_frac", "zero_shot"), ("oof_frac", "fine_tuned_oof")):
        out[name] = {f"tol{t}": seq_stats(idx, col, t) for t in (10, 20)}
    # chance level for the 3-way verdict, using the clinician verdict distribution
    json.dump(out, open(os.path.join(RES, "ext_lutseg_cv.json"), "w"), indent=1); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"train": train_cv, "eval": eval_cv}[(sys.argv[1:] or ["train"])[0]]()
