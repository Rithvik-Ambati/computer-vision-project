"""Writes demo_upload/healing_visits/<trend>/visitN_dayDD.png : 6 follow-up photos per simulated patient (different camera pose each visit),
ready for 'Upload your own visits' in tab 4 of the Gradio demo.  Truth is stored in truth.json next to the photos (not in the file names)."""
import json, os, cv2, numpy as np
from wound_cv import demo_app as D
from wound_cv.experiments import ROOT, build_items

SEED = 7        # the seed where pixels-only gets the wrong call for 'stalled' and 'worsening' (seeds 5 and 6 agree with the proposed method)
for name, trend in (("healing", list(D.HEAL_TRENDS)[0]), ("stalled", list(D.HEAL_TRENDS)[1]), ("worsening", list(D.HEAL_TRENDS)[2])):
    out = os.path.join(ROOT, "demo_upload", "healing_visits", name); os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(SEED)
    r = D.run_healing(trend, SEED)
    for t, im in enumerate(r["imgs"]):
        cv2.imwrite(os.path.join(out, f"visit{t + 1}_day{D.DAYS[t]:02d}.png"), im)
    json.dump(dict(trend=trend, days=D.DAYS, true_area_mm2=[float(a) for a in r["true"]], proposed_mm2=[float(a) for a in r["full"]]),
              open(os.path.join(out, "truth.json"), "w"), indent=1)
    print(name, "true", np.round(r["true"]).astype(int).tolist(), "proposed", np.round(r["full"]).astype(int).tolist())
