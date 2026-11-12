"""Physically-grounded wound phantom simulator.

A phantom is a flat (or cylinder-wrapped) skin texture in millimetres that carries
  * an irregular wound with an exactly known area (analytic polygon area),
  * a 40 mm ArUco scale marker,
and is rendered through a pinhole camera with tilt, distance, lens distortion,
lighting, blur and noise.  Because the geometry is known exactly we get ground truth
for area, masks, and image-to-image geometry for every rendered photo.
"""
import cv2
import numpy as np

PPM = 3                      # texture pixels per mm
FIELD = 200                  # texture side in mm
TEX = FIELD * PPM
IMG = 256                    # output image size
F_PX = 260.0                 # focal length (px) of the simulated phone camera
MARKER_MM = 40.0
MARKER_ID = 7
ARUCO = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
K1_TRUE, K2_TRUE = -0.35, 0.0    # simulated barrel lens distortion (single-parameter)
BG = (110, 110, 110)


def _smooth_noise(rng, size, grid):
    g = rng.standard_normal((grid, grid)).astype(np.float32)
    return cv2.resize(g, (size, size), interpolation=cv2.INTER_CUBIC)


def make_skin(seed):
    rng = np.random.default_rng(seed)
    base = np.array([rng.uniform(125, 170), rng.uniform(150, 182), rng.uniform(185, 222)], np.float32)
    tex = np.ones((TEX, TEX, 3), np.float32) * base
    fine = cv2.GaussianBlur(rng.standard_normal((TEX, TEX)).astype(np.float32), (0, 0), 0.8)
    var = 14 * _smooth_noise(rng, TEX, 8) + 7 * _smooth_noise(rng, TEX, 40) + 5 * fine
    tex += var[..., None] * np.array([1.0, 0.9, 0.8], np.float32)
    ov = np.zeros((TEX, TEX), np.float32)
    for _ in range(800):
        cv2.circle(ov, (int(rng.integers(0, TEX)), int(rng.integers(0, TEX))), 1, 1.0, -1)
    tex *= (1 - 0.28 * cv2.GaussianBlur(ov, (0, 0), 0.8))[..., None]
    ov[:] = 0
    for _ in range(45):
        cv2.circle(ov, (int(rng.integers(0, TEX)), int(rng.integers(0, TEX))), int(rng.integers(3, 6)), 1.0, -1)
    ov = cv2.GaussianBlur(ov, (0, 0), 1.5)
    tex *= (1 - 0.30 * ov)[..., None] * 1.0
    tex[..., 2] += 12 * ov
    ov[:] = 0
    for _ in range(70):
        x, y = int(rng.integers(0, TEX)), int(rng.integers(0, TEX))
        a = rng.uniform(0, 2 * np.pi); L = rng.integers(12, 28)
        cv2.line(ov, (x, y), (int(x + L * np.cos(a)), int(y + L * np.sin(a))), 1.0, 1)
    tex *= (1 - 0.35 * cv2.GaussianBlur(ov, (0, 0), 0.7))[..., None]
    return tex


class Wound:
    """Smooth random blob: r(theta) = r0 (1 + sum a_k cos(k theta + phi_k)); area is exact."""

    def __init__(self, rng, area_mm2):
        self.coef = [(k, rng.uniform(0.04, 0.20) / (k ** 0.6), rng.uniform(0, 2 * np.pi)) for k in (2, 3, 4, 5)]
        self.target = float(area_mm2)
        self.r0 = np.sqrt(area_mm2 / np.pi)
        self.r0 *= np.sqrt(area_mm2 / self._area(1.0))   # normalise so the area is exact

    def _radii(self, th):
        r = np.ones_like(th)
        for k, a, p in self.coef:
            r += a * np.cos(k * th + p)
        return self.r0 * r

    def _area(self, s):
        th = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
        r = self._radii(th) * s if s != 1.0 else self._radii(th)
        return 0.5 * np.sum(r ** 2) * (2 * np.pi / 2000)

    def poly(self, scale=1.0, n=720):
        th = np.linspace(0, 2 * np.pi, n, endpoint=False)
        r = self._radii(th) * scale
        pts = np.stack([r * np.cos(th), r * np.sin(th)], 1)
        x, y = pts[:, 0], pts[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        return pts, area


def make_scene(seed, area_mm2=None, scale=1.0, healing=0.0, skin=None, wound=None, board=False):
    """Return a scene dict. `scale` shrinks the wound radius (healing); area = area0*scale^2."""
    rng = np.random.default_rng(seed + 12345)
    tex = (skin if skin is not None else make_skin(seed)).copy()
    if area_mm2 is None:
        area_mm2 = rng.uniform(150, 1500)
    wound = wound if wound is not None else Wound(rng, area_mm2)
    centre = np.array([122.0 + rng.uniform(-4, 4), 80.0 + rng.uniform(-4, 4)])
    mk = np.array([38.0 + rng.uniform(-4, 4), 112.0 + rng.uniform(-3, 3)])
    scene = dict(seed=seed, wound=wound, centre=centre, marker_tl=mk, scale=scale, healing=healing)
    if board:
        tex = _checkerboard()
        scene.update(tex=tex, mask=np.zeros((TEX, TEX), np.float32), area=0.0)
        return scene
    pts, area = wound.poly(scale)
    SS = 4
    hi = np.zeros((TEX * 2, TEX * 2), np.uint8)
    cv2.fillPoly(hi, [np.round((pts + centre) * PPM * 2 * 16).astype(np.int32)], 255, lineType=cv2.LINE_AA, shift=4)
    m = cv2.resize(hi, (TEX, TEX), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    # peri-wound erythema halo
    halo = cv2.GaussianBlur(m, (0, 0), 9) * 0.55
    tex[..., 2] += 30 * halo; tex[..., 0] -= 25 * halo; tex[..., 1] -= 25 * halo
    # wound bed: red granulation -> pink as healing progresses, with yellow slough
    rng2 = np.random.default_rng(seed + 777)
    c0, c1 = np.array([45, 45, 170], np.float32), np.array([95, 95, 205], np.float32)
    col = (1 - healing) * c0 + healing * c1
    bed = np.ones((TEX, TEX, 3), np.float32) * col
    bed *= (1 + 0.10 * _smooth_noise(rng2, TEX, 20))[..., None]
    slough = np.clip(_smooth_noise(rng2, TEX, 10) - 0.6, 0, 1)
    bed = bed * (1 - 0.7 * slough[..., None]) + np.array([90, 175, 205], np.float32) * 0.7 * slough[..., None]
    mm = cv2.GaussianBlur(m, (0, 0), 0.8)[..., None]
    tex = tex * (1 - mm) + bed * mm
    # ArUco marker on a white card
    mpx = int(MARKER_MM * PPM)
    mark = cv2.aruco.generateImageMarker(ARUCO, MARKER_ID, mpx, borderBits=1)
    x0, y0 = np.round(mk * PPM).astype(int)
    pad = 4 * PPM
    tex[y0 - pad:y0 + mpx + pad, x0 - pad:x0 + mpx + pad] = 238
    tex[y0:y0 + mpx, x0:x0 + mpx] = mark[..., None].astype(np.float32)
    scene.update(tex=np.clip(tex, 0, 255).astype(np.uint8), mask=m, area=area)
    return scene


def _checkerboard(cols=10, rows=7, sq=13.0):
    t = np.full((TEX, TEX, 3), 235, np.uint8)
    w, h = cols * sq, rows * sq
    x0, y0 = (FIELD - w) / 2, (FIELD - h) / 2
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                a = (np.array([x0 + c * sq, y0 + r * sq]) * PPM).round().astype(int)
                b = (np.array([x0 + (c + 1) * sq, y0 + (r + 1) * sq]) * PPM).round().astype(int)
                t[a[1]:b[1], a[0]:b[0]] = 20
    return t


def board_object_points(cols=10, rows=7, sq=13.0):
    """Inner corners (cols-1 x rows-1) in the board's own mm frame (origin at texture centre)."""
    o = np.array([[(c - (cols - 2) / 2) * sq, (r - (rows - 2) / 2) * sq, 0] for r in range(rows - 1) for c in range(cols - 1)], np.float32)
    return o


def default_cam(**kw):
    cam = dict(tilt=0.0, az=0.0, D=240.0, f=F_PX, jx=0.0, jy=0.0, R=None, k1=K1_TRUE, k2=K2_TRUE,
               gain=1.0, contrast=1.0, gamma=1.0, grad=0.0, grad_dir=0.0, noise=2.0, blur=0.6, seed=0)
    cam.update(kw)
    return cam


def random_cam(rng, tilt_max=45, d_range=(190, 300), curved=0.0, light_strength=1.0):
    return default_cam(
        tilt=rng.uniform(0, tilt_max), az=rng.uniform(0, 360), D=rng.uniform(*d_range),
        jx=rng.uniform(-8, 8), jy=rng.uniform(-8, 8),
        R=(rng.uniform(35, 120) if rng.random() < curved else None),
        gain=rng.uniform(1 - 0.25 * light_strength, 1 + 0.2 * light_strength),
        contrast=rng.uniform(1 - 0.2 * light_strength, 1 + 0.2 * light_strength),
        gamma=rng.uniform(1 - 0.25 * light_strength, 1 + 0.25 * light_strength),
        grad=rng.uniform(0, 0.45 * light_strength), grad_dir=rng.uniform(0, 2 * np.pi),
        noise=rng.uniform(1, 5), blur=rng.uniform(0.4, 1.0), seed=int(rng.integers(1 << 30)))


def _rays(cam, size, f, cx, cy):
    u, v = np.meshgrid(np.arange(size, dtype=np.float32), np.arange(size, dtype=np.float32))
    xd, yd = (u - cx) / f, (v - cy) / f
    x, y = xd.copy(), yd.copy()
    for _ in range(8):                                  # invert radial distortion
        r2 = x * x + y * y
        s = 1 + cam["k1"] * r2 + cam["k2"] * r2 * r2
        x, y = xd / s, yd / s
    th = np.deg2rad(cam["tilt"]); az = np.deg2rad(cam["az"])
    Rt, _ = cv2.Rodrigues(np.array([np.cos(az), np.sin(az), 0.0]) * th)
    C = Rt @ np.array([0, 0, -cam["D"]])
    d = x[..., None] * Rt[:, 0] + y[..., None] * Rt[:, 1] + Rt[:, 2]
    return C, d.astype(np.float64)


def render(scene, cam, field=None):
    """Render scene -> (bgr uint8, gt_mask uint8, tex_xy mm (IMG,IMG,2) with NaN outside, gt_area_mm2).

    `field` optionally adds a smooth tissue deformation (dx, dy in mm, shape TEXxTEX) to model
    non-rigid motion between frames of a burst.
    """
    S = 2
    n = IMG * S
    f2 = cam["f"] * S
    c2 = IMG - 0.5
    C, d = _rays(cam, n, f2, c2, c2)
    ox, oy = FIELD / 2 + cam["jx"], FIELD / 2 + cam["jy"]
    if cam["R"] is None:
        t = -C[2] / d[..., 2]
        ok = (t > 0) & (d[..., 2] > 0)
        X = C[0] + t * d[..., 0]; Y = C[1] + t * d[..., 1]
        tx, ty = X + ox, Y + oy
    else:
        R = float(cam["R"])
        dx_, dz_ = d[..., 0], d[..., 2]
        ox_, oz_ = C[0], C[2] - R
        A = dx_ ** 2 + dz_ ** 2
        B = 2 * (ox_ * dx_ + oz_ * dz_)
        Cc = ox_ ** 2 + oz_ ** 2 - R ** 2
        disc = B ** 2 - 4 * A * Cc
        ok = disc >= 0
        t = (-B - np.sqrt(np.where(ok, disc, 0))) / (2 * A)
        ok &= t > 0
        X = C[0] + t * d[..., 0]; Z = C[2] + t * d[..., 2]; Y = C[1] + t * d[..., 1]
        phi = np.arctan2(X, -(Z - R))
        ok &= np.abs(phi) < np.pi / 2
        tx, ty = R * phi + ox, Y + oy
    ok &= (tx > 0) & (tx < FIELD) & (ty > 0) & (ty < FIELD)
    if field is not None:
        ix = np.clip(np.round(tx * PPM).astype(int), 0, TEX - 1); iy = np.clip(np.round(ty * PPM).astype(int), 0, TEX - 1)
        tx = tx + field[0][iy, ix]; ty = ty + field[1][iy, ix]
    mx = np.where(ok, tx * PPM - 0.5, -10).astype(np.float32)
    my = np.where(ok, ty * PPM - 0.5, -10).astype(np.float32)
    img = cv2.remap(scene["tex"], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=BG)
    msk = cv2.remap(scene["mask"], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    img = cv2.resize(img, (IMG, IMG), interpolation=cv2.INTER_AREA).astype(np.float32)
    msk = (cv2.resize(msk, (IMG, IMG), interpolation=cv2.INTER_AREA) > 0.5).astype(np.uint8)
    txy = np.stack([np.where(ok, tx, np.nan), np.where(ok, ty, np.nan)], -1)[::S, ::S].astype(np.float32)
    # photometric model
    rng = np.random.default_rng(cam["seed"])
    uu, vv = np.meshgrid(np.arange(IMG), np.arange(IMG))
    gdir = (np.cos(cam["grad_dir"]) * (uu - IMG / 2) + np.sin(cam["grad_dir"]) * (vv - IMG / 2)) / IMG
    img *= (1 + cam["grad"] * gdir)[..., None]
    img = (img - 128) * cam["contrast"] + 128
    img = 255 * np.clip(img * cam["gain"] / 255, 0, 1) ** cam["gamma"]
    if cam["blur"] > 0:
        img = cv2.GaussianBlur(img, (0, 0), cam["blur"])
    img += rng.normal(0, cam["noise"], img.shape)
    return np.clip(img, 0, 255).astype(np.uint8), msk, txy, scene["area"]


def smooth_field(seed, amp_mm=1.5):
    rng = np.random.default_rng(seed)
    return (amp_mm * _smooth_noise(rng, TEX, 4) * 0.5, amp_mm * _smooth_noise(rng, TEX, 4) * 0.5)
