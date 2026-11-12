# Geometry-Aware Computer Vision Framework for Automated Wound Measurement

Phantom simulator with exact ground truth, checkerboard lens calibration, ArUco-marker homography,
U-Net segmentation, metric area/perimeter, healing progress, SIFT+RANSAC registration and optical-flow mask propagation.

## Layout
- `wound_cv/sim.py` simulator, `geometry.py` calibration + rectification + measurement, `segment.py` U-Net,
  `experiments.py` all experiments, `realdata.py` real-photo test (Mendeley lower-limb set), `diagrams.py` architecture figures, `report.py` HTML report
- `results/` CSV/JSON/log outputs of every experiment; `figs/` all figures (01-19); `models/` trained weights; `worked_examples.md`

## Run
```
pip install numpy opencv-contrib-python torch matplotlib pandas scipy
python -m wound_cv.experiments        # stages: calib gallery main sweeps repeat healing flow train worked
python -m wound_cv.realdata           # needs the Mendeley dataset in real_data/mendeley
python -m wound_cv.diagrams
```
Key results (150 simulated test photos): proposed pipeline MAPE 9.1% with U-Net masks (5.8% with perfect masks) vs 26.5% pixels-only; U-Net Dice 0.964.
