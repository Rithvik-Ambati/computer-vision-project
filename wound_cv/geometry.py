"""Marker detection, lens calibration, metric rectification and area/perimeter measurement."""
import cv2
import numpy as np

from . import sim

_params = cv2.aruco.DetectorParameters()
_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
_detector = cv2.aruco.ArucoDetector(sim.ARUCO, _params)
L = sim.MARKER_MM
MARKER_MM_PTS = np.array([[0, 0], [L, 0], [L, L], [0, L]], np.float32)
NOMINAL_SCALE = sim.F_PX / 240.0       # px/mm a "no-correction" user would assume (frontal @ 24 cm)


# ---------------------------------------------------------------- calibration
def calibrate_camera(n_views=24, seed=0):
    """Checkerboard calibration (Zhang) on simulated views. Returns dict with K, dist, rms, views."""
    rng = np.random.default_rng(seed)
    scene = sim.make_scene(0, board=True)
    obj = sim.board_object_points()
    objs, imgs, shown = [], [], None
    for i in range(n_views):
        cam = sim.default_cam(tilt=rng.uniform(0, 45), az=rng.uniform(0, 360), D=rng.uniform(150, 230),
                              jx=rng.uniform(-10, 10), jy=rng.uniform(-10, 10), noise=1.5, blur=0.5, seed=int(rng.integers(1 << 30)))
        img, *_ = sim.render(scene, cam)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ok, c = cv2.findChessboardCorners(g, (9, 6), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not ok:
            continue
        c = cv2.cornerSubPix(g, c, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4))
        objs.append(obj); imgs.append(c)
        if shown is None:
            shown = (img.copy(), c.copy())
    K0 = np.array([[300, 0, sim.IMG / 2], [0, 300, sim.IMG / 2], [0, 0, 1]], np.float64)
    flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_ASPECT_RATIO
    rms, K, dist, rv, tv = cv2.calibrateCamera(objs, imgs, (sim.IMG, sim.IMG), K0, np.zeros(5), flags=flags)
    per_view = []
    for o, i, r, t in zip(objs, imgs, rv, tv):
        p, _ = cv2.projectPoints(o, r, t, K, dist)
        per_view.append(float(np.sqrt(np.mean(np.sum((p.reshape(-1, 2) - i.reshape(-1, 2)) ** 2, -1)))))
    return dict(K=K, dist=dist.ravel(), rms=float(rms), per_view=per_view, n_used=len(objs), n_views=n_views, example=shown)


# ---------------------------------------------------------------- marker + rectification
def detect_marker(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    corners, ids, _ = _detector.detectMarkers(g)
    if ids is None:
        return None
    for c, i in zip(corners, ids.ravel()):
        if i == sim.MARKER_ID:
            return c.reshape(4, 2).astype(np.float32)
    return None


def undistort_points(pts, K, dist):
    return cv2.undistortPoints(pts.reshape(-1, 1, 2).astype(np.float32), K, dist, P=K).reshape(-1, 2)


def undistort_mask(mask, K, dist):
    u = cv2.undistort(mask.astype(np.float32), K, dist)
    return (u > 0.5).astype(np.uint8)


def largest_component(mask):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return mask.astype(np.uint8)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (lab == k).astype(np.uint8)


def homography_to_mm(corners):
    return cv2.getPerspectiveTransform(corners.astype(np.float32), MARKER_MM_PTS)


def scale_from_marker(corners):
    sides = [np.linalg.norm(corners[i] - corners[(i + 1) % 4]) for i in range(4)]
    return float(np.mean(sides) / L)


def measure(mask, H, ppm=8):
    """Warp `mask` into a metric canvas with homography H (image px -> mm) and measure it.

    Area = (#canvas pixels inside) / ppm^2  -- an unbiased pixel count in mm^2 (no staircase bias).
    Perimeter = contour length in the canvas / ppm (digital perimeter; see README caveat).
    """
    mask = largest_component(mask)
    ys, xs = np.nonzero(mask)
    if len(xs) < 5:
        return dict(area=0.0, perimeter=0.0, canvas=None, T=None)
    box = np.array([[xs.min(), ys.min()], [xs.max() + 1, ys.min()], [xs.max() + 1, ys.max() + 1], [xs.min(), ys.max() + 1]], np.float32)
    mm = cv2.perspectiveTransform(box.reshape(-1, 1, 2), H).reshape(-1, 2)
    lo, hi = mm.min(0) - 2, mm.max(0) + 2
    w, h = np.ceil((hi - lo) * ppm).astype(int)
    T = np.array([[ppm, 0, -lo[0] * ppm], [0, ppm, -lo[1] * ppm], [0, 0, 1]]) @ H
    can = cv2.warpPerspective(mask.astype(np.float32), T, (int(w), int(h)), flags=cv2.INTER_LINEAR)
    b = (can > 0.5).astype(np.uint8)
    area = b.sum() / ppm ** 2
    cs, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    per = max((cv2.arcLength(c, True) for c in cs), default=0) / ppm
    return dict(area=float(area), perimeter=float(per), canvas=b, T=T, lo=lo)


def scale_homography(s):
    return np.diag([1.0 / s, 1.0 / s, 1.0])


def pipeline_measure(img, mask, calib=None, mode="full"):
    """Measure wound area for one photo under an ablation `mode`:

      pixel    : nominal scale only (what you'd get from pixels + a guessed distance)
      scale    : marker side length -> single px/mm (no perspective correction)
      homog    : marker homography (perspective corrected), no lens correction
      full     : lens undistortion + marker homography      <- proposed pipeline
    Returns dict(area, perimeter, ok, corners, H) ; ok=False if the marker was not found.
    """
    if mode == "pixel":
        r = measure(mask, scale_homography(NOMINAL_SCALE))
        return dict(ok=True, **r, corners=None, H=None)
    corners = detect_marker(img)
    if corners is None:
        return dict(ok=False, area=np.nan, perimeter=np.nan, corners=None, H=None, canvas=None, T=None)
    if mode == "scale":
        r = measure(mask, scale_homography(scale_from_marker(corners)))
        return dict(ok=True, **r, corners=corners, H=None)
    if mode == "homog":
        H = homography_to_mm(corners)
        return dict(ok=True, **measure(mask, H), corners=corners, H=H)
    if mode == "full":
        c = undistort_points(corners, calib["K"], calib["dist"])
        m = undistort_mask(mask, calib["K"], calib["dist"])
        H = homography_to_mm(c)
        return dict(ok=True, **measure(m, H), corners=c, H=H)
    raise ValueError(mode)


def healing_percent(a0, at):
    return (a0 - at) / a0 * 100.0
