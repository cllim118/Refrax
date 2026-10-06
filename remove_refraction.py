"""Removes flat-port refraction from underwater images.

Usage:
    python remove_refraction.py configs/dataset.yaml [--overwrite]

Every image in paths.rgb_dir is warped to a refraction-free pinhole image with
focal length zoom * (fx, fy). Outputs:
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
from core.undistort import compute_housing_geometry, build_undistort_map_closed_form

IMAGE_EXTS = ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG",
              "*.bmp", "*.BMP", "*.tiff", "*.TIFF"]


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


def make_map_builder(W, H, fx, fy, cx, cy, housing, distortion, zoom):
    """Returns build_map(depth) -> (map_x, map_y), float32 maps for cv2.remap.

    Pixels without a source are set to -1. If radtan distortion coefficients are
    non-zero, the map points into the distorted (raw) image.
    """
    _, K_inv = matrix_K(fx, fy, cx, cy)
    P2, ray_water = compute_housing_geometry(H, W, K_inv, **housing)
    apply_radtan = any(d != 0.0 for d in distortion)

    def build_map(depth):
        map_x, map_y = build_undistort_map_closed_form(
            P2, ray_water, depth, fx, fy, cx, cy, H, W, zoom=zoom)

        if apply_radtan:
            x_dist, y_dist = apply_radtan_distortion(
                (map_x - cx) / fx, (map_y - cy) / fy, *distortion)
            map_x, map_y = x_dist * fx + cx, y_dist * fy + cy

        return (np.nan_to_num(map_x, nan=-1).astype(np.float32),
                np.nan_to_num(map_y, nan=-1).astype(np.float32))

    return build_map


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
    zoom = corr.get("zoom")

    if zoom is None:
        print(f"Finding optimal zoom at Z0={z0_fixed}...")
        zoom, zoom_rmse = select_zoom(z0_fixed, W, H, fx, fy, cx, cy, housing)
        print(f"Best scale in bounds = {zoom:.4f}")
        print(f"RMSE = {zoom_rmse:.4f} px")
    else:
        print(f"Using zoom from config = {zoom:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    build_map = make_map_builder(W, H, fx, fy, cx, cy, housing, distortion, zoom)

    rgb_paths = find_rgb_paths(rgb_dir, step_size)
    print(f"Found {len(rgb_paths)} images (step={step_size}), "
          f"crop_valid_bbox={crop_valid_bbox}, "
          f"radtan={any(d != 0.0 for d in distortion)}")

    if depth_dir is None:
        print(f"Single depth mode (Z0={z0_fixed}) — computing undistortion map once")
        fixed_map = build_map(z0_fixed)
        fixed_mask = compute_valid_mask(*fixed_map, W, H)
        cv2.imwrite(f"{mask_dir}/mask.png", fixed_mask)
        print(f"Saved shared mask: {mask_dir}/mask.png")

        out_W, out_H = W, H
        out_fx, out_fy = fx * zoom, fy * zoom
        out_cx, out_cy = cx, cy
        fixed_box = None
        if crop_valid_bbox:
            fixed_box = largest_valid_rectangle(fixed_mask)
            y0, y1, x0, x1 = fixed_box
            out_W, out_H = x1 - x0, y1 - y0
            out_cx, out_cy = cx - x0, cy - y0
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
