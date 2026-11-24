"""Curvature-aware area measurement (extension).

The planar pipeline assumes marker and wound lie in one plane.  On a limb that is wrong.  Here the surface is
modelled as a circular cylinder of radius R whose axis is parallel to the marker's vertical edges:

    surface point  P(s, y) = ( R sin(s/R),  y,  R - R cos(s/R) )      s = arc length, y = axial distance (mm)

1. The camera pose and the marker's arc offset s0 are fitted to the 4 detected marker corners
   (7 unknowns, 8 equations, R supplied by the clinician or swept in the sensitivity study).
2. Every wound-mask pixel is cast onto the cylinder; the area is the sum of the local area element
   |d(s,y)/d(u,v)| over the mask, i.e. the area of the *unrolled* (developed) surface, which is what a
   tape measure on the skin would give.
"""
import cv2
import numpy as np
from scipy.optimize import least_squares

from . import sim, geometry as G

L = sim.MARKER_MM

# Curvature-tolerant ArUco detector: a marker wrapped around a limb is no longer a perfect quadrilateral, so the
# polygon-approximation and border-bit tolerances are relaxed (found 88% vs 50% of photos at R = 80 mm in our test).
_tp = cv2.aruco.DetectorParameters()
_tp.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
_tp.polygonalApproxAccuracyRate = 0.12
_tp.maxErroneousBitsInBorderRate = 0.8
_tp.errorCorrectionRate = 1.0
_tp.perspectiveRemovePixelPerCell = 8
_tp.adaptiveThreshWinSizeMax = 53
_tp.minMarkerPerimeterRate = 0.02
_tol_detector = cv2.aruco.ArucoDetector(sim.ARUCO, _tp)


def detect_marker_tolerant(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    corners, ids, _ = _tol_detector.detectMarkers(g)
    if ids is None:
        return None
    for c, i in zip(corners, ids.ravel()):
        if i == sim.MARKER_ID:
            return c.reshape(4, 2).astype(np.float32)
    return None


_SQ = np.array([[0, 0], [L, 0], [L, L], [0, L]], np.float64)


def _surface(s, y, R):
    return np.stack([R * np.sin(s / R), y, R - R * np.cos(s / R)], -1)


def _project(theta, R, K):
    rv, C, s0 = theta[:3], theta[3:6], theta[6]
    Rt, _ = cv2.Rodrigues(rv)
    P = _surface(s0 + _SQ[:, 0], _SQ[:, 1], R)
    cam = (P - C) @ Rt                           # Rt^T (P - C)
    return np.stack([K[0, 0] * cam[:, 0] / cam[:, 2] + K[0, 2], K[1, 1] * cam[:, 1] / cam[:, 2] + K[1, 2]], -1)


def fit_pose(corners_u, K, R, s0_inits=(-60, -40, -20, 0, 20, 40)):
    """corners_u: undistorted marker corners (4,2).  Returns (theta, rms_px)."""
    _, rvp, tvp = cv2.solvePnP(np.c_[_SQ, np.zeros(4)].astype(np.float32), corners_u.astype(np.float32), K, None, flags=cv2.SOLVEPNP_IPPE)
    Rp, _ = cv2.Rodrigues(rvp)
    Rt0, C0 = Rp.T, (-Rp.T @ tvp).ravel()
    rv0 = cv2.Rodrigues(Rt0)[0].ravel()
    best = None
    for s0 in s0_inits:
        th0 = np.r_[rv0, C0, s0]
        try:
            sol = least_squares(lambda th: (_project(th, R, K) - corners_u).ravel(), th0, method="lm", max_nfev=400)
        except Exception:
            continue
        rms = float(np.sqrt(np.mean(sol.fun ** 2)))
        if best is None or rms < best[1]:
            best = (sol.x, rms)
    return best


def area_on_cylinder(mask, theta, R, K):
    """Area (mm^2) of `mask` (in the undistorted image) on the fitted cylinder, via the area element."""
    n = mask.shape[0]
    u, v = np.meshgrid(np.arange(n, dtype=np.float64), np.arange(mask.shape[1], dtype=np.float64))
    x, y = (u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1]
    rv, C = theta[:3], theta[3:6]
    Rt, _ = cv2.Rodrigues(rv)
    d = x[..., None] * Rt[:, 0] + y[..., None] * Rt[:, 1] + Rt[:, 2]
    ox, oz = C[0], C[2] - R
    A = d[..., 0] ** 2 + d[..., 2] ** 2
    B = 2 * (ox * d[..., 0] + oz * d[..., 2])
    Cc = ox ** 2 + oz ** 2 - R ** 2
    disc = B ** 2 - 4 * A * Cc
    ok = disc > 0
    t = (-B - np.sqrt(np.where(ok, disc, 0))) / (2 * np.maximum(A, 1e-12))
    X = C[0] + t * d[..., 0]; Z = C[2] + t * d[..., 2]; Y = C[1] + t * d[..., 1]
    s = R * np.arctan2(X, -(Z - R))
    ds_du, ds_dv = np.gradient(s, axis=1), np.gradient(s, axis=0)
    dy_du, dy_dv = np.gradient(Y, axis=1), np.gradient(Y, axis=0)
    elem = np.abs(ds_du * dy_dv - ds_dv * dy_du)
    m = (mask > 0) & ok & np.isfinite(elem)
    return float(elem[m].sum())


def measure_cylinder(img, mask, calib, R, corners=None):
    """Full curvature-aware measurement.  Returns dict(ok, area, rms, s0) ; area is NaN if the marker is missing."""
    if corners is None:
        corners = detect_marker_tolerant(img)
    if corners is None:
        return dict(ok=False, area=np.nan, rms=np.nan, s0=np.nan)
    K = calib["K"]
    cu = G.undistort_points(corners, K, calib["dist"])
    mu = G.largest_component(G.undistort_mask(mask, K, calib["dist"]))
    fit = fit_pose(cu, K, R)
    if fit is None or mu.sum() < 5:
        return dict(ok=False, area=np.nan, rms=np.nan, s0=np.nan)
    th, rms = fit
    return dict(ok=True, area=area_on_cylinder(mu, th, R, K), rms=rms, s0=float(th[6]))


def fit_pose_and_radius(corners_u, K, R_inits=(70, 120, 250), s0_inits=(-50, -20, 10)):
    """Optional: estimate the limb radius too (8 unknowns from 8 corner coordinates -> exactly determined, hence noise-sensitive)."""
    _, rvp, tvp = cv2.solvePnP(np.c_[_SQ, np.zeros(4)].astype(np.float32), corners_u.astype(np.float32), K, None, flags=cv2.SOLVEPNP_IPPE)
    Rp, _ = cv2.Rodrigues(rvp)
    rv0 = cv2.Rodrigues(Rp.T)[0].ravel(); C0 = (-Rp.T @ tvp).ravel()
    best = None
    for R0 in R_inits:
        for s0 in s0_inits:
            try:
                sol = least_squares(lambda th: (_project(th[:7], th[7], K) - corners_u).ravel(), np.r_[rv0, C0, s0, R0], method="lm", max_nfev=600)
            except Exception:
                continue
            if not (20 < sol.x[7] < 1e4):
                continue
            rms = float(np.sqrt(np.mean(sol.fun ** 2)))
            if best is None or rms < best[1]:
                best = (sol.x, rms)
    return best


def measure_cylinder_estR(img, mask, calib, corners=None):
    """Curvature-aware area with the radius estimated from the marker alone.  Returns dict(ok, area, R_est)."""
    if corners is None:
        corners = detect_marker_tolerant(img)
    if corners is None:
        return dict(ok=False, area=np.nan, R_est=np.nan)
    K = calib["K"]
    cu = G.undistort_points(corners, K, calib["dist"])
    mu = G.largest_component(G.undistort_mask(mask, K, calib["dist"]))
    fit = fit_pose_and_radius(cu, K)
    if fit is None or mu.sum() < 5:
        return dict(ok=False, area=np.nan, R_est=np.nan)
    th, _ = fit
    return dict(ok=True, area=area_on_cylinder(mu, th[:7], th[7], K), R_est=float(th[7]))
