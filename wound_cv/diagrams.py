"""Architecture diagrams for the paper: system pipeline + U-Net.   python -m wound_cv.diagrams"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, "figs")
C = dict(inp="#e8eaee", cal="#dbe8fb", geo="#d8f0e4", seg="#fde9cf", meas="#e4dcf7", lon="#fbe0e0", ink="#1a1d23", edge="#5b6270", acc="#1f9d6b")


def box(ax, x, y, w, h, title, sub="", fc="#fff", fs=9.5, sfs=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.016", fc=fc, ec=C["edge"], lw=1.1))
    if sub:
        ax.text(x + w / 2, y + h - 0.022, title, ha="center", va="top", fontsize=fs, fontweight="bold", color=C["ink"])
        ax.text(x + w / 2, y + h - 0.075, sub, ha="center", va="top", fontsize=sfs, color="#33373f", linespacing=1.35)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=fs, fontweight="bold", color=C["ink"])


def arrow(ax, p, q, rad=0.0, color=None, lw=1.5):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, lw=lw, color=color or C["edge"],
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1))


def system():
    fig, ax = plt.subplots(figsize=(13.4, 7.4)); ax.set_xlim(0, 1.02); ax.set_ylim(-0.06, 1); ax.axis("off")
    ax.text(0.51, 0.985, "System architecture: calibrate once per device, then rectify, segment and measure every photo",
            ha="center", va="top", fontsize=12, fontweight="bold")
    # row A: calibration (once per device)
    ax.text(0.005, 0.915, "ONCE PER DEVICE", fontsize=8, color="#5b6270", fontweight="bold")
    box(ax, 0.005, 0.74, 0.17, 0.16, "Checkerboard photos", "10-25 views of a printed board", C["inp"])
    box(ax, 0.25, 0.74, 0.25, 0.16, "1  Lens calibration (Zhang)", "corner detection + sub-pixel refinement\nestimates K (f, c) and distortion k1", C["cal"])
    arrow(ax, (0.175, 0.82), (0.25, 0.82))
    # row B: per photo
    ax.text(0.005, 0.665, "EVERY PHOTO", fontsize=8, color="#5b6270", fontweight="bold")
    box(ax, 0.005, 0.42, 0.13, 0.22, "Wound photo", "smartphone image,\n40 mm ArUco marker\nin the wound plane", C["inp"], sfs=7.6)
    box(ax, 0.175, 0.42, 0.14, 0.22, "2a  Undistort", "x_d = x(1 + k1 r^2)\nmarker corners and\nmask use (K, k1)", C["geo"], sfs=7.6)
    box(ax, 0.35, 0.42, 0.14, 0.22, "2b  ArUco detection", "4 corners p_i found\nmetric positions\nq_i = (0,0)...(L,L)", C["geo"], sfs=7.6)
    box(ax, 0.525, 0.42, 0.14, 0.22, "2c  Homography", "q ~ H p\nimage px -> mm plane\n(needs marker and\nwound coplanar)", C["geo"], sfs=7.6)
    box(ax, 0.70, 0.42, 0.175, 0.22, "4  Metric measurement", "warp mask with\nT = diag(rho, rho, 1) H\n(rho = 8 px/mm)\nA = N / rho^2", C["meas"], sfs=7.6)
    box(ax, 0.91, 0.42, 0.105, 0.22, "Output", "area (mm$^2$)\nperimeter", "#ffffff", sfs=8)
    for a, b in [(0.135, 0.175), (0.315, 0.35), (0.49, 0.525), (0.665, 0.70), (0.875, 0.91)]:
        arrow(ax, (a, 0.53), (b, 0.53))
    ax.text(0.595, 0.665, "H", fontsize=9, color=C["acc"], fontweight="bold", ha="center")
    # calibration -> undistort
    arrow(ax, (0.34, 0.74), (0.26, 0.64), rad=0.0, color=C["acc"])
    ax.text(0.30, 0.69, "K, k1", fontsize=8.5, color=C["acc"], fontweight="bold", ha="right")
    # row C: segmentation branch
    box(ax, 0.175, 0.20, 0.30, 0.15, "3  U-Net segmentation", "256 x 256 RGB in, wound mask M out\ntrained with BCE loss (lambda = 0)", C["seg"], sfs=7.8)
    arrow(ax, (0.07, 0.42), (0.175, 0.275))
    arrow(ax, (0.475, 0.275), (0.79, 0.42), rad=-0.22)
    ax.text(0.60, 0.30, "mask M", fontsize=8.5, fontweight="bold", color="#8a5a00", ha="center")
    # row D: longitudinal (dashed container)
    ax.add_patch(FancyBboxPatch((0.50, 0.015), 0.515, 0.245, boxstyle="round,pad=0.004,rounding_size=0.016", fc="none", ec=C["edge"], lw=1.1, ls="--"))
    ax.text(0.7575, 0.252, "5  Longitudinal analysis (several photos)", ha="center", va="top", fontsize=9.5, fontweight="bold")
    box(ax, 0.515, 0.035, 0.155, 0.165, "Healing progress", "dA = (A_0 - A_t) / A_0\nx 100 %", C["lon"], fs=9, sfs=7.6)
    box(ax, 0.685, 0.035, 0.155, 0.165, "SIFT + RANSAC", "match outside the wound;\nhomography to reference\nvisit (boundary overlay)", C["lon"], fs=9, sfs=7.4)
    box(ax, 0.855, 0.035, 0.145, 0.165, "Optical flow", "Farneback flow within a\nburst; propagate the\nreference mask", C["lon"], fs=9, sfs=7.4)
    arrow(ax, (0.96, 0.42), (0.96, 0.262), lw=1.3)
    ax.text(0.968, 0.355, "A, M per visit", fontsize=7.5, color="#5b6270", va="center")
    ax.text(0.51, -0.055, "Green: calibration and geometry used by the measurement stage. Stage 5 is optional; wound area needs only stages 1-4.",
            ha="center", va="bottom", fontsize=8, color="#5b6270")
    fig.savefig(os.path.join(FIGS, "18_system_architecture.png"), dpi=200, bbox_inches="tight", facecolor="white"); plt.close(fig)


def unet():
    fig, ax = plt.subplots(figsize=(13.2, 5.8)); ax.set_xlim(0, 14.2); ax.set_ylim(0, 5.9); ax.axis("off")
    ch = [12, 24, 48, 96, 192]; res = [256, 128, 64, 32, 16]
    hs = [3.6, 2.9, 2.2, 1.55, 1.0]; w = 0.55; yb = 0.55
    xe = [1.9, 3.5, 5.1, 6.7, 8.3]
    xdec = {3: 9.4, 2: 10.5, 1: 11.6, 0: 12.7}
    enc, dec, bot = "#cfe1fb", "#d8f0e4", "#fde9cf"

    def ylo(i):
        return yb + 0.55 + (hs[0] - hs[i]) / 2

    ax.text(7.1, 5.85, "U-Net architecture (base width 12, 4 levels, about 1.1 M parameters)", ha="center", va="top", fontsize=12, fontweight="bold")
    ax.text(0.15, ylo(0) + hs[0] / 2, "Input\n3 x 256 x 256\nRGB photo", fontsize=8.5, ha="left", va="center", fontweight="bold")
    for i in range(5):
        ax.add_patch(Rectangle((xe[i], ylo(i)), w, hs[i], fc=bot if i == 4 else enc, ec=C["edge"], lw=1.1, zorder=3))
        ax.text(xe[i] + w / 2, ylo(i) + hs[i] / 2, str(ch[i]), ha="center", va="center", fontsize=9, fontweight="bold", rotation=90, zorder=4)
        ax.text(xe[i] + w / 2, ylo(i) - 0.22, f"{res[i]}x{res[i]}", ha="center", va="top", fontsize=7.6, color="#33373f")
        if i < 4:
            ax.add_patch(FancyArrowPatch((xe[i] + w + 0.06, ylo(i) + hs[i] / 2 - 0.05), (xe[i + 1] - 0.08, ylo(i + 1) + hs[i + 1] / 2 - 0.05),
                                         arrowstyle="-|>", mutation_scale=11, lw=1.3, color=C["edge"]))
    ax.text(xe[4] + w / 2, ylo(4) + hs[4] + 0.12, "bottleneck", ha="center", fontsize=8, fontweight="bold", color="#8a5a00")
    prev_x, prev_lvl = xe[4], 4
    for lvl in (3, 2, 1, 0):
        x = xdec[lvl]
        ax.add_patch(Rectangle((x, ylo(lvl)), w, hs[lvl], fc=dec, ec=C["edge"], lw=1.1, zorder=3))
        ax.text(x + w / 2, ylo(lvl) + hs[lvl] / 2, str(ch[lvl]), ha="center", va="center", fontsize=9, fontweight="bold", rotation=90, zorder=4)
        ax.text(x + w / 2, ylo(lvl) - 0.22, f"{res[lvl]}x{res[lvl]}", ha="center", va="top", fontsize=7.6, color="#33373f")
        ax.add_patch(FancyArrowPatch((prev_x + w + 0.06, ylo(prev_lvl) + hs[prev_lvl] / 2 - 0.05), (x - 0.08, ylo(lvl) + hs[lvl] / 2 - 0.05),
                                     arrowstyle="-|>", mutation_scale=11, lw=1.3, color=C["edge"]))
        # skip connection (arch above both blocks)
        ytop = ylo(lvl) + hs[lvl] + 0.32
        ax.plot([xe[lvl] + w / 2, xe[lvl] + w / 2, x + w / 2], [ylo(lvl) + hs[lvl], ytop, ytop], color=C["acc"], lw=1.3, zorder=1)
        ax.add_patch(FancyArrowPatch((x + w / 2, ytop), (x + w / 2, ylo(lvl) + hs[lvl] + 0.02), arrowstyle="-|>", mutation_scale=10, lw=1.3, color=C["acc"], zorder=1))
        prev_x, prev_lvl = x, lvl
    ax.text(7.1, 5.28, "skip connections: encoder features are concatenated with the up-sampled decoder features", fontsize=8.4, color=C["acc"], ha="center", fontweight="bold")
    ax.text(13.5, ylo(0) + hs[0] / 2, "1x1 conv\n+ sigmoid\n1 x 256 x 256\nwound mask", fontsize=8.3, ha="left", va="center", fontweight="bold")
    ax.add_patch(Rectangle((0.2, 0.2), 0.3, 0.2, fc=enc, ec=C["edge"])); ax.text(0.6, 0.3, "encoder block: 2 x [conv 3x3 + BN + ReLU], then max-pool 2x2", fontsize=8.2, va="center")
    ax.add_patch(Rectangle((9.4, 0.2), 0.3, 0.2, fc=dec, ec=C["edge"])); ax.text(9.8, 0.3, "decoder block: up-conv 2x2, concat skip, 2 x [conv 3x3 + BN + ReLU]", fontsize=8.2, va="center")
    fig.savefig(os.path.join(FIGS, "19_unet_architecture.png"), dpi=200, bbox_inches="tight", facecolor="white"); plt.close(fig)


if __name__ == "__main__":
    system(); unet(); print("ok")
