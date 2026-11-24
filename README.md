# Geometry-Aware Computer Vision Framework for Automated Wound Measurement

Phantom simulator with exact ground truth, checkerboard lens calibration, ArUco-marker homography,
U-Net segmentation, metric area/perimeter, healing progress, SIFT+RANSAC registration and optical-flow mask propagation.

## Layout
- `wound_cv/sim.py` simulator, `geometry.py` calibration + rectification + measurement, `segment.py` U-Net,
  `experiments.py` all experiments, `realdata.py` real-photo test (Mendeley lower-limb set), `diagrams.py` architecture figures, `report.py` HTML report
- `results/` CSV/JSON/log outputs of every experiment (`ext_*` = extensions); `figs/` all figures (01-19 paper, 20-30 extensions); `models/` trained weights; `worked_examples.md`

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

## Extensions (new metrics, curvature-aware area, uncertainty, domain randomisation, live demo)
All extension outputs are written to `results/ext_*` and `figs/20-30`; the original results and figures 01-19 are untouched.
```
python -m wound_cv.extensions metrics   # HD95, ASSD, precision/recall, CCC, R2, within +-10/20 %, Wilcoxon tests, runtime  -> figs 20-24
python -m wound_cv.extensions curv      # cylinder (curvature-aware) area vs planar, radius sensitivity, tolerant marker detector -> fig 25
python -m wound_cv.extensions unc       # Monte-Carlo + split-conformal intervals, reliability flag -> fig 26
python -m wound_cv.extensions grid      # tilt x distance error heat-map -> fig 27
python -m wound_cv.extensions estR      # negative result: radius cannot be estimated from the 4 marker corners alone
python -m wound_cv.domrand train        # domain-randomised U-Net (GPU used automatically if a CUDA PyTorch is installed)
python -m wound_cv.domrand eval         # zero-shot Dice on the 537 real test photos -> results/ext_domrand.json
python -m wound_cv.extensions curves    # training curves, real-photo gallery, domain-randomisation bar -> figs 28-30
python show_metrics.py                  # prints the paper metrics straight from results/
```
`wound_cv/curvature.py` holds the cylinder model; `wound_cv/domrand.py` the augmentation + training.

## Live demo
```
python -m wound_cv.demo_app --make-target   # writes demo_target.png: marker + printed shapes of known area (print at 100 %)
python -m wound_cv.demo_app                 # window with 3 tabs: simulated photo (exact truth), real photo (outline), printed target
python -m wound_cv.demo_app --selftest      # headless check of every demo path
python make_demo_set.py                     # ranks the held-out real test photos by Dice into real_data/demo_candidates/
python make_upload_set.py                   # builds demo_upload/ (real photos, simulated marker photos, printed-target stand-ins)
python -c "from wound_cv import demo_app as D; D.make_visit_pages('demo_upload/healing_pages')"   # printable visit pages (healing / stalled / worsening, 4 visits each)
python -m wound_cv.gradio_app [--share]     # browser version; tab 4 = healing over visits (upload your phone photos of the printed visit pages)
```
Real-camera caveat: the lens calibration is of the *simulated* camera, so the printed-target tab uses the marker homography only.

## GPU
`segment.py` uses CUDA automatically when `torch.cuda.is_available()`. On Windows this needs a recent NVIDIA driver and a CUDA build of PyTorch
(see pytorch.org); check with `python -c "import torch; print(torch.cuda.is_available())"`.
