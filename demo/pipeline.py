"""Demo logic shared by the Gradio app (app.py) and the in-browser page (index.html).

Everything here is plain numpy/OpenCV/matplotlib, so it also runs under Pyodide.
"""
import os
import sys
from functools import lru_cache

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE if os.path.isdir(os.path.join(HERE, "core")) else os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.config import load_config, camera_params, distortion_params
from core.optics import matrix_K
from core.scale import select_zoom
from core.undistort import compute_housing_geometry, forward_map, invert_map
from remove_refraction import make_map_builder, compute_valid_mask, largest_valid_rectangle
from visualise_refraction import (make_checkerboard, blue_gray,
                                  BOARD_COLS, BOARD_ROWS, SQUARE_SIZE)

# Optional downscaling of the input (long side, px) to speed up the correction.
# Images are processed at their original size unless the user picks a limit.
RESIZE_RANGE = (320, 1920)
RESIZE_START = 1024  # slider position when the user turns resizing on

N_PORT = np.array([0.0, 0.0, 1.0])  # port normal in the camera frame, shared by all presets
MU_A = 1.0                          # refractive index of the air inside the housing


def _preset_params(cfg):
    """Flat demo parameters (cam_W, ..., z0) from a preset in presets.yaml."""
    hs = cfg["housing"]
    W, H, fx, fy, cx, cy = camera_params(cfg)
    k1, k2, p1, p2 = distortion_params(cfg)
    return dict(cam_W=W, cam_H=H, fx=fx, fy=fy, cx=cx, cy=cy,
                k1=k1, k2=k2, p1=p1, p2=p2,
                rflat=hs["rflat"], tglass=hs["tglass"], mu_g=hs["mu_g"], mu_w=hs["mu_w"],
                z0=cfg["z0"])


def _load_presets():
    """Presets from presets.yaml: {name: {label, params, samples: [{name, url, thumb}]}}."""
    spec = load_config(os.path.join(HERE, "presets.yaml"))
    base = f"https://huggingface.co/datasets/{spec['dataset']}/resolve/main"
    presets = {}
    for name, p in spec["presets"].items():
        samples = [dict(name=s.split("/")[-1], url=f"{base}/{s}.png",
                        thumb=f"samples/{s.replace('/', '_')}.jpg")
                   for s in p.get("samples", [])]
        presets[name] = dict(label=p["label"], params=_preset_params(p), samples=samples)
    return presets


PRESETS = _load_presets()
DEFAULT_PRESET = next(iter(PRESETS))
DEFAULTS = PRESETS[DEFAULT_PRESET]["params"]


class DemoError(Exception):
    """An error whose message is meant for the demo user."""


def housing_dict(rflat, tglass, mu_g, mu_w):
    return dict(n_port=N_PORT, rflat=rflat, tglass=tglass, mu_a=MU_A, mu_g=mu_g, mu_w=mu_w)


def scale_intrinsics(W, H, cam_W, cam_H, fx, fy, cx, cy):
    """Intrinsics of the calibrated camera resampled to a (W, H) image (pixel-centre convention)."""
    sx, sy = W / cam_W, H / cam_H
    return fx * sx, fy * sy, (cx + 0.5) * sx - 0.5, (cy + 0.5) * sy - 0.5


def _select_zoom(z0, W, H, fx, fy, cx, cy, housing):
    try:
        return select_zoom(z0, W, H, fx, fy, cx, cy, housing)[0]
    except RuntimeError as e:
        raise DemoError(str(e))


def _check_depth(z0, rflat, tglass):
    if z0 <= rflat + tglass:
        raise DemoError("Scene depth must lie beyond the port (z0 > rflat + tglass).")


@lru_cache(maxsize=8)
def correction_maps(W, H, fx, fy, cx, cy, distortion, rflat, tglass, mu_g, mu_w,
                    z0, zoom, crop):
    """Remap tables, validity mask, crop box and zoom for one camera/housing setup (cached)."""
    housing = housing_dict(rflat, tglass, mu_g, mu_w)
    if zoom is None:
        zoom = _select_zoom(z0, W, H, fx, fy, cx, cy, housing)
    build_map = make_map_builder(W, H, fx, fy, cx, cy, housing, distortion, zoom)
    map_x, map_y = build_map(z0)
    mask = compute_valid_mask(map_x, map_y, W, H)
    box = largest_valid_rectangle(mask) if crop else None
    return map_x, map_y, mask, box, zoom


def calibration_text(W, H, fx, fy, cx, cy, zoom):
    return (f"focal_length: [{fx:.6f}, {fy:.6f}]\n"
            f"principal_point: [{cx:.6f}, {cy:.6f}]\n"
            "distortion_coefficients: [0, 0, 0, 0]\n"
            f"image_dimension: [{W}, {H}]\n"
            f"zoom: {zoom:.6f}\n")


def correct_image(img, cam_W, cam_H, fx, fy, cx, cy, k1, k2, p1, p2,
                  rflat, tglass, mu_g, mu_w, z0, auto_zoom, zoom, crop,
                  max_side=None):
    """Corrects an RGB image. Returns (corrected, mask, calibration YAML, notes).

    If `max_side` is given, images whose long side exceeds it are downscaled first
    (never upscaled); otherwise the original size is kept.

    The corrected image is cropped if `crop`, otherwise RGBA with the validity mask as alpha.
    """
    if img is None:
        raise DemoError("Upload an underwater image first.")
    _check_depth(z0, rflat, tglass)

    notes = []
    H, W = img.shape[:2]
    if max_side and max(H, W) > int(max_side):
        max_side = int(max_side)
        s = max_side / max(H, W)
        img = cv2.resize(img, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA)
        notes.append(f"Downscaled the input from {W}×{H} to {img.shape[1]}×{img.shape[0]}.")
        H, W = img.shape[:2]

    cam_W, cam_H = int(cam_W), int(cam_H)
    if (W, H) != (cam_W, cam_H):
        if abs((W / cam_W) / (H / cam_H) - 1) > 0.02:
            notes.append(f"⚠️ The image aspect ratio ({W}×{H}) differs from the calibration "
                         f"({cam_W}×{cam_H}); the result will be inaccurate.")
        fx, fy, cx, cy = scale_intrinsics(W, H, cam_W, cam_H, fx, fy, cx, cy)
        notes.append(f"Intrinsics rescaled from {cam_W}×{cam_H} to {W}×{H}.")

    map_x, map_y, mask, box, zoom = correction_maps(
        W, H, round(fx, 6), round(fy, 6), round(cx, 6), round(cy, 6),
        (k1, k2, p1, p2), rflat, tglass, mu_g, mu_w, z0,
        None if auto_zoom else zoom, crop)
    if auto_zoom:
        notes.append(f"Automatic zoom: {zoom:.4f}.")

    corrected = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    out_W, out_H, out_cx, out_cy = W, H, cx, cy
    if box is not None:
        y0, y1, x0, x1 = box
        corrected = corrected[y0:y1, x0:x1]
        out_W, out_H, out_cx, out_cy = x1 - x0, y1 - y0, cx - x0, cy - y0
    else:
        corrected = np.dstack([corrected, mask])  # validity mask as alpha

    calib = calibration_text(out_W, out_H, fx * zoom, fy * zoom, out_cx, out_cy, zoom)
    return corrected, mask, calib, "\n\n".join(notes)


def simulate(cam_W, cam_H, fx, fy, cx, cy, rflat, tglass, mu_g, mu_w, z0, auto_zoom, zoom):
    """Checkerboard at depth z0 seen through the housing, corrected, and the per-pixel displacement.

    Returns (matplotlib figure, Markdown summary).
    """
    _check_depth(z0, rflat, tglass)
    W, H = int(cam_W), int(cam_H)
    housing = housing_dict(rflat, tglass, mu_g, mu_w)
    if auto_zoom:
        zoom = _select_zoom(z0, W, H, fx, fy, cx, cy, housing)

    _, K_inv = matrix_K(fx, fy, cx, cy)
    u_grid, v_grid = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    P2, ray_water = compute_housing_geometry(H, W, K_inv, **housing)
    fwd_x, fwd_y = forward_map(P2, ray_water, z0, fx, fy, cx, cy)
    map_u, map_v = invert_map((fwd_x - cx) * zoom + cx, (fwd_y - cy) * zoom + cy, H, W)

    disparity = np.sqrt((map_u - u_grid) ** 2 + (map_v - v_grid) ** 2)
    rmse = np.sqrt(np.nanmean(disparity ** 2))

    P_world = np.stack([(fwd_x - cx) / fx * z0, (fwd_y - cy) / fy * z0,
                        np.full((H, W), z0)], axis=-1)
    img_underwater = make_checkerboard(P_world, BOARD_COLS, BOARD_ROWS, SQUARE_SIZE).astype(np.uint8)
    img_corrected = cv2.remap(img_underwater,
                              np.nan_to_num(map_u, nan=-1).astype(np.float32),
                              np.nan_to_num(map_v, nan=-1).astype(np.float32),
                              cv2.INTER_LINEAR)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    axes[0].imshow(img_underwater, cmap=blue_gray, vmin=0, vmax=255)
    axes[0].set_title("underwater")
    axes[1].imshow(img_corrected, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("corrected")
    im = axes[2].imshow(disparity, cmap="hot")
    axes[2].set_title("pixel displacement (px)")
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()

    stats = (f"**z0** = {z0} · **zoom** = {zoom:.4f} · displacement **RMSE** = {rmse:.2f} px, "
             f"**p95** = {np.nanpercentile(disparity, 95):.2f} px, "
             f"**max** = {np.nanmax(disparity):.2f} px")
    return fig, stats
