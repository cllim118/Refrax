"""Removes flat-port refraction from underwater images.

Usage:
    python remove_refraction.py configs/dataset.yaml [--overwrite]

Every image in paths.rgb_dir is warped to a refraction-free pinhole image with
focal length zoom * (fx, fy). The output is sized to hold the whole corrected
image (correction.fit_canvas, default), or has the source size. Outputs:
    <output_dir>/<name>.png  corrected image; cropped to the largest rectangle
                             without invalid pixels if correction.crop_valid_bbox,
                             otherwise BGRA with the validity mask as alpha
    <mask_dir>/mask.png      validity mask (<mask_dir>/<name>.png per image when
                             a depth map is given per image)
    <calib_path>             intrinsics of the corrected images (fixed-depth mode)
"""
import argparse
import glob
import os

import cv2
import numpy as np

from core.config import load_config, camera_params, distortion_params, housing_params
from core.optics import matrix_K, apply_radtan_distortion
from core.scale import select_zoom
from core.undistort import (compute_housing_geometry, build_undistort_map_closed_form,
                            forward_map, zoom_forward_map, forward_map_canvas)
from core.undistort_newton import build_undistort_map_newton

IMAGE_EXTS = ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG",
              "*.bmp", "*.BMP", "*.tiff", "*.TIFF"]
METHODS = ("closed_form", "newton")


def find_rgb_paths(rgb_dir, step_size):
    paths = sorted(p for ext in IMAGE_EXTS for p in glob.glob(f"{rgb_dir}/{ext}"))
    return paths[::step_size]


def load_depth(name, depth_dir, W, H):
    """Per-image depth map <depth_dir>/<name>.npy resized to (W, H), or None if missing."""
    depth_path = f"{depth_dir}/{name}.npy"
    if not os.path.exists(depth_path):
        return None
    depth = np.load(depth_path).astype(np.float32)
    if depth.shape[:2] != (H, W):
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)
    return np.clip(depth, 1e-3, None)


def make_map_builder(method, W, H, fx, fy, cx, cy, housing, distortion, zoom, canvas=None):
    """Returns build_map(depth) -> (map_x, map_y), float32 maps for cv2.remap.

    The maps cover canvas = (out_W, out_H, origin), the source frame by default.
    Pixels without a source are set to -1. If radtan distortion coefficients are
    non-zero, the map points into the distorted (raw) image.
    """
    _, K_inv = matrix_K(fx, fy, cx, cy)
    if method == "closed_form":
        P2, ray_water = compute_housing_geometry(H, W, K_inv, **housing)
    newton_kwargs = dict(K_inv=K_inv, **housing)
    apply_radtan = any(d != 0.0 for d in distortion)
    out_W, out_H, origin = canvas if canvas is not None else (W, H, (0.0, 0.0))

    def build_map(depth):
        if method == "closed_form":
            map_x, map_y = build_undistort_map_closed_form(
                P2, ray_water, depth, fx, fy, cx, cy, out_H, out_W, zoom=zoom, origin=origin)
        else:
            map_x, map_y = build_undistort_map_newton(
                fx, fy, cx, cy, depth, newton_kwargs, out_W, out_H, zoom=zoom,
                origin=origin, src_size=(W, H))

        if apply_radtan:
            x_dist, y_dist = apply_radtan_distortion(
                (map_x - cx) / fx, (map_y - cy) / fy, *distortion)
            map_x, map_y = x_dist * fx + cx, y_dist * fy + cy

        return (np.nan_to_num(map_x, nan=-1).astype(np.float32),
                np.nan_to_num(map_y, nan=-1).astype(np.float32))

    return build_map


def fit_canvas(method, W, H, fx, fy, cx, cy, housing, distortion, zoom, z0):
    """Canvas (out_W, out_H, origin) holding the whole image corrected at depth z0.

    Starts from the bounding box of the zoomed forward map, with one pixel of slack
    for the interpolation hull, then shrinks to the valid pixels of the map built on
    it. origin is the source-frame coordinate of the canvas' pixel (0, 0).
    """
    _, K_inv = matrix_K(fx, fy, cx, cy)
    P2, ray_water = compute_housing_geometry(H, W, K_inv, **housing)
    map_x, map_y = zoom_forward_map(*forward_map(P2, ray_water, z0, fx, fy, cx, cy),
                                    cx, cy, zoom)
    out_W, out_H, (x0, y0) = forward_map_canvas(map_x, map_y)
    out_W, out_H, x0, y0 = out_W + 2, out_H + 2, x0 - 1.0, y0 - 1.0
    if out_W > 8 * W or out_H > 8 * H:
        raise ValueError(f"fit_canvas: corrected image would be {out_W}x{out_H} for a "
                         f"{W}x{H} source; check the housing parameters and zoom ({zoom}).")

    build_map = make_map_builder(method, W, H, fx, fy, cx, cy, housing, distortion, zoom,
                                 canvas=(out_W, out_H, (x0, y0)))
    valid = compute_valid_mask(*build_map(z0), W, H) > 0
    rows, cols = np.flatnonzero(valid.any(axis=1)), np.flatnonzero(valid.any(axis=0))
    if len(rows) == 0:
        raise ValueError("fit_canvas: the corrected image has no valid pixel.")
    return (int(cols[-1] - cols[0] + 1), int(rows[-1] - rows[0] + 1),
            (x0 + float(cols[0]), y0 + float(rows[0])))


def compute_valid_mask(map_x, map_y, W, H):
    """255 where the map samples inside the (W, H) source image, 0 elsewhere."""
    valid = (
        (map_x >= 0) & (map_x <= W - 1) &
        (map_y >= 0) & (map_y <= H - 1)
    )
    return np.where(valid, 255, 0).astype(np.uint8)


def largest_valid_rectangle(mask):
    """Largest axis-aligned rectangle of non-zero mask pixels, as (y0, y1, x0, x1).

    Row-by-row largest-rectangle-in-histogram; falls back to the whole image.
    """
    H, W = mask.shape
    valid = mask > 0

    height = np.zeros(W, dtype=np.int32)
    best_area = 0
    best_box = (0, H, 0, W)

    for y in range(H):
        height = np.where(valid[y], height + 1, 0)

        stack = []  # (start_x, h)
        for x in range(W + 1):
            h = height[x] if x < W else 0
            start = x
            while stack and stack[-1][1] > h:
                s_x, s_h = stack.pop()
                area = s_h * (x - s_x)
                if area > best_area:
                    best_area = area
                    best_box = (y - s_h + 1, y + 1, s_x, x)
                start = s_x
            stack.append((start, h))

    return best_box


def write_calibration(path, W, H, fx, fy, cx, cy, zoom):
    with open(path, "w") as f:
        f.write(f"focal_length: [{fx:.6f}, {fy:.6f}]\n")
        f.write(f"principal_point: [{cx:.6f}, {cy:.6f}]\n")
        f.write("distortion_coefficients: [0, 0, 0, 0]\n")
        f.write(f"image_dimension: [{W}, {H}]\n")
        f.write(f"zoom: {zoom:.6f}\n")
    print(f"Saved calibration: {path}")


def main():
    parser = argparse.ArgumentParser(description="Refraction removal from a YAML config")
    parser.add_argument("config", help="Path to config YAML file")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute images whose output already exists")
    args = parser.parse_args()

    cfg = load_config(args.config)
    method = cfg.get("method", "closed_form")
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")

    W, H, fx, fy, cx, cy = camera_params(cfg)
    distortion = distortion_params(cfg)
    housing = housing_params(cfg)

    paths = cfg["paths"]
    rgb_dir, output_dir, mask_dir = paths["rgb_dir"], paths["output_dir"], paths["mask_dir"]
    depth_dir = paths.get("depth_dir")
    calib_path = paths.get("calib_path",
                           os.path.join(os.path.dirname(output_dir), "new_calibration.yaml"))

    corr = cfg["correction"]
    z0_fixed = corr["z0_fixed"]
    step_size = corr.get("step_size", 1)
    crop_valid_bbox = corr.get("crop_valid_bbox", False)
    fit = corr.get("fit_canvas", True)
    zoom = corr.get("zoom")

    if zoom is None:
        print(f"Finding optimal zoom at Z0={z0_fixed}...")
        zoom, zoom_rmse = select_zoom(z0_fixed, W, H, fx, fy, cx, cy, housing,
                                      in_bounds=not fit)
        print(f"Best scale{'' if fit else ' in bounds'} = {zoom:.4f}")
        print(f"RMSE = {zoom_rmse:.4f} px")
    else:
        print(f"Using zoom from config = {zoom:.4f}")

    canvas = (W, H, (0.0, 0.0))
    if fit:
        canvas = fit_canvas(method, W, H, fx, fy, cx, cy, housing, distortion, zoom, z0_fixed)
        print(f"Fitted canvas: {canvas[0]}x{canvas[1]}, origin=({canvas[2][0]:.0f}, "
              f"{canvas[2][1]:.0f}) in the source frame")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    build_map = make_map_builder(method, W, H, fx, fy, cx, cy, housing, distortion, zoom, canvas)

    rgb_paths = find_rgb_paths(rgb_dir, step_size)
    print(f"Found {len(rgb_paths)} images (step={step_size}), method={method}, "
          f"fit_canvas={fit}, crop_valid_bbox={crop_valid_bbox}, "
          f"radtan={any(d != 0.0 for d in distortion)}")

    if depth_dir is None:
        print(f"Single depth mode (Z0={z0_fixed}) — computing undistortion map once")
        fixed_map = build_map(z0_fixed)
        fixed_mask = compute_valid_mask(*fixed_map, W, H)
        cv2.imwrite(f"{mask_dir}/mask.png", fixed_mask)
        print(f"Saved shared mask: {mask_dir}/mask.png")

        out_W, out_H, (ox, oy) = canvas
        out_fx, out_fy = fx * zoom, fy * zoom
        out_cx, out_cy = cx - ox, cy - oy
        fixed_box = None
        if crop_valid_bbox:
            fixed_box = largest_valid_rectangle(fixed_mask)
            y0, y1, x0, x1 = fixed_box
            out_W, out_H = x1 - x0, y1 - y0
            out_cx, out_cy = out_cx - x0, out_cy - y0
            print(f"Inscribed valid rectangle: rows[{y0}:{y1}], cols[{x0}:{x1}] "
                  f"({out_W}x{out_H}, no invalid pixels)")
            print(f"Adjusted intrinsics for cropped output: "
                  f"fx={out_fx:.2f}, fy={out_fy:.2f}, cx={out_cx:.2f}, cy={out_cy:.2f}")

        write_calibration(calib_path, out_W, out_H, out_fx, out_fy, out_cx, out_cy, zoom)

    for rgb_path in rgb_paths:
        name = os.path.splitext(os.path.basename(rgb_path))[0]
        out_path = f"{output_dir}/{name}.png"
        if os.path.exists(out_path) and not args.overwrite:
            print(f"Exists: {name}.png → skip (use --overwrite to recompute)")
            continue

        img = cv2.imread(rgb_path)
        if img is None:
            print(f"Unreadable: {rgb_path} → skip")
            continue
        if img.shape[:2] != (H, W):
            print(f"Size {img.shape[1]}x{img.shape[0]} != camera {W}x{H}: {name} → skip")
            continue

        if depth_dir is None:
            (map_x, map_y), mask, box = fixed_map, fixed_mask, fixed_box
        else:
            depth = load_depth(name, depth_dir, W, H)
            if depth is None:
                print(f"No depth: {name} → skip")
                continue
            map_x, map_y = build_map(depth)
            mask = compute_valid_mask(map_x, map_y, W, H)
            cv2.imwrite(f"{mask_dir}/{name}.png", mask)
            box = largest_valid_rectangle(mask) if crop_valid_bbox else None

        corrected = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        if box is not None:
            # Every pixel inside the inscribed rectangle is valid, so no alpha is needed.
            y0, y1, x0, x1 = box
            out = corrected[y0:y1, x0:x1]
        else:
            out = cv2.cvtColor(corrected, cv2.COLOR_BGR2BGRA)
            out[..., 3] = mask

        cv2.imwrite(out_path, out)
        print(f"Saved: {name}.png ({out.shape[1]}x{out.shape[0]}, {out.shape[2]} channels)")

    print("Done.")


if __name__ == "__main__":
    main()
