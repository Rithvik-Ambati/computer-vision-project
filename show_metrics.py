"""Prints the evaluation metrics straight from results/ (no recomputation by hand)."""
import json, numpy as np, pandas as pd
R = "results/"
d = pd.read_csv(R + "main.csv")
print("== SEGMENTATION (150 simulated test photos) ==")
print(f"Dice mean {d.dice.mean():.3f} | IoU mean {d.iou.mean():.3f} | n={len(d)}")
print("\n== AREA ERROR (|est-true|/true, %) ==")
m = d[d.marker_found]
print(f"marker found in {len(m)}/{len(d)} photos")
print(f"{'method':<10}{'GT mask MAPE':>14}{'U-Net MAPE':>12}{'U-Net bias%':>13}")
for k, name in [("pixel", "pixel"), ("scale", "scale"), ("homog", "homog"), ("full", "full")]:
    s = d if k == "pixel" else m
    g = 100 * (s["gt_" + k] - s.gt_area).abs() / s.gt_area
    n = 100 * (s["net_" + k] - s.gt_area) / s.gt_area
    print(f"{name:<10}{g.mean():>14.1f}{n.abs().mean():>12.1f}{n.mean():>13.1f}")
print("\n== LOSS ABLATION (validation) ==", json.dumps(json.load(open(R + "val_area_bias.json")), indent=1))
print("== CALIBRATION ==", json.dumps(json.load(open(R + "calibration.json")), indent=1)[:600])
print("== REPEATABILITY ==", json.dumps(json.load(open(R + "repeatability.json")), indent=1)[:600])
print("== HEALING MAE ==", json.dumps(json.load(open(R + "healing_mae.json")), indent=1)[:600])
print("== FLOW IoU ==", json.dumps(json.load(open(R + "flow_iou.json")), indent=1))
r = json.load(open(R + "real_data.json"))
print("\n== REAL PHOTOS (537 test) ==")
for k, v in r.items():
    if isinstance(v, dict) and "dice" in v:
        print(f"{k.replace(chr(8594),'->'):<16} Dice {v['dice']:.3f}  IoU {v['iou']:.3f}  n={v['n']}")
