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

## Dataset (real-photo test only)
The simulated experiments need no data. `wound_cv/realdata.py` uses the public **Lower Limb and Feet Wound Image Dataset**
(Mendeley Data, CC BY 4.0, DOI [10.17632/hsj38fwnvr](https://doi.org/10.17632/hsj38fwnvr), v3, 2,686 photos with expert masks).

1. Open https://doi.org/10.17632/hsj38fwnvr and download the dataset (`wound.zip`; a free Mendeley account may be needed).
2. Extract it so the folders sit exactly here:
```
real_data/mendeley/wound_main/wound_main-0001.jpg ...   # photographs
real_data/mendeley/wound_mask/wound_mask-0001.jpg ...   # expert masks
```
(The `Nomal` folder in the archive is not used.) The first run resizes everything to 256x256 and caches it in `data/mendeley_all.npz`.
3. Run `python -m wound_cv.realdata`. It uses a seeded 70/10/20 train/val/test split, trains from scratch and from the simulated weights, and rewrites `results/real_data*.json` and `figs/17_real_data.png`.
