# Worked examples

True wound area (analytic polygon): **820.0 mm² = 8.20 cm²**, true perimeter 105.9 mm. Marker: 40 mm ArUco (DICT_4X4_50, id 7). Lens: f=259.8 px, k1=-0.341 (from checkerboard calibration).


## Example A — Frontal, flat skin

| step | value |
|---|---|
| marker corners (px) | `[[67.30000305175781, 140.1999969482422], [107.4000015258789, 140.60000610351562], [107.5, 180.8000030517578], [68.19999694824219, 179.6999969482422]]` |
| marker scale (mean side) | 0.995 px/mm |
| homography H (image→mm) | `[[0.9541, -0.0014, -62.8329], [-0.0033, 0.9523, -133.5544], [0.0, -0.0001, 1.0]]` |
| U-Net mask | 859 px, Dice 0.956, IoU 0.915 |


| method | area (mm²) | error vs true |
|---|---|---|
| Pixels only (nominal scale) | 731.1 | -10.8% |
| Marker scale only | 869.3 | +6.0% |
| + Perspective (homography) | 736.2 | -10.2% |
| + Lens undistortion (proposed) | 801.2 | -2.3% |

true area 820.0 mm²


## Example B — Tilted 45° camera, flat skin

| step | value |
|---|---|
| marker corners (px) | `[[74.69999694824219, 129.8000030517578], [110.19999694824219, 135.5], [116.4000015258789, 164.10000610351562], [83.80000305175781, 157.39999389648438]]` |
| marker scale (mean side) | 0.797 px/mm |
| homography H (image→mm) | `[[0.9004, -0.2972, -27.9601], [-0.1558, 0.9993, -118.2607], [0.0016, -0.003, 1.0]]` |
| U-Net mask | 914 px, Dice 0.954, IoU 0.912 |


| method | area (mm²) | error vs true |
|---|---|---|
| Pixels only (nominal scale) | 778.3 | -5.1% |
| Marker scale only | 1439.1 | +75.5% |
| + Perspective (homography) | 773.9 | -5.6% |
| + Lens undistortion (proposed) | 769.8 | -6.1% |

true area 820.0 mm²


## Example C — Tilt 30° on a curved limb (R = 60 mm)

| step | value |
|---|---|
| marker corners (px) | `[[65.5999984741211, 145.0], [106.5, 141.1999969482422], [108.9000015258789, 187.60000610351562], [66.69999694824219, 189.39999389648438]]` |
| marker scale (mean side) | 1.089 px/mm |
| homography H (image→mm) | `[[1.2146, 0.0037, -78.5079], [0.11, 1.1278, -171.028], [0.0008, 0.0014, 1.0]]` |
| U-Net mask | 488 px, Dice 0.926, IoU 0.863 |


| method | area (mm²) | error vs true |
|---|---|---|
| Pixels only (nominal scale) | 414.5 | -49.4% |
| Marker scale only | 410.8 | -49.9% |
| + Perspective (homography) | 373.6 | -54.4% |
| + Lens undistortion (proposed) | 404.5 | -50.7% |

true area 820.0 mm²
