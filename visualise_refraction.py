"""Visualises the refraction correction on a synthetic checkerboard.

Usage:
    python visualise_refraction.py configs/dataset.yaml

Renders a checkerboard at depth correction.z0_fixed as seen through the
housing, corrects it with the configured zoom, and plots the
per-pixel displacement of the correction.
"""
import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from core.config import load_config, camera_params, housing_params
from core.optics import matrix_K
from core.scale import select_zoom
from core.undistort import compute_housing_geometry, forward_map, invert_map

BOARD_COLS, BOARD_ROWS, SQUARE_SIZE = 10, 10, 0.1

blue_gray = LinearSegmentedColormap.from_list('blue_gray', ['black', (0.70, 0.82, 1.0)])


def make_checkerboard(P_world, cols, rows, size, background=127):
    """Intensity of a checkerboard centred on the optical axis at world points P_world."""
    X, Y = P_world[..., 0], P_world[..., 1]
    w, h = cols * size, rows * size
    Xl, Yl = X + w / 2, Y + h / 2

    mask = (Xl >= 0) & (Xl < w) & (Yl >= 0) & (Yl < h)
    squares = np.full_like(X, background, dtype=np.int32)
    ix = np.floor(Xl[mask] / size).astype(int)
    iy = np.floor(Yl[mask] / size).astype(int)
    squares[mask] = np.where((ix + iy) % 2 == 0, 255, 0)
    return squares


def plot_radial_disparity(disparity, u_grid, v_grid, cx, cy, n_bins=50):
    r = np.sqrt((u_grid - cx) ** 2 + (v_grid - cy) ** 2).ravel()
    d = disparity.ravel()

    bins = np.linspace(0, r.max(), n_bins + 1)
    bin_idx = np.digitize(r, bins) - 1
    bin_means = [d[bin_idx == i].mean() if np.any(bin_idx == i) else np.nan
                 for i in range(n_bins)]
    bin_centers = (bins[:-1] + bins[1:]) / 2

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(r, d, s=2, alpha=0.1, color='gray', label='per-pixel')
    ax.plot(bin_centers, bin_means, color='crimson', linewidth=2, label='binned mean')
    ax.set_xlabel("radial distance r (px)")
    ax.set_ylabel("pixel disparity (px)")
    ax.set_title("Disparity vs radius")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()


def main():
    parser = argparse.ArgumentParser(description="Visualise refraction correction from a YAML config")
    parser.add_argument("config", help="Path to config YAML file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    W, H, fx, fy, cx, cy = camera_params(cfg)
    housing = housing_params(cfg)
    Z0 = cfg["correction"]["z0_fixed"]
    zoom = cfg["correction"].get("zoom")
    if zoom is None:
        zoom, _ = select_zoom(Z0, W, H, fx, fy, cx, cy, housing)

    _, K_inv = matrix_K(fx, fy, cx, cy)
    u_grid, v_grid = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))

    P2, ray_water = compute_housing_geometry(H, W, K_inv, **housing)
    fwd_x, fwd_y = forward_map(P2, ray_water, Z0, fx, fy, cx, cy)
    map_u, map_v = invert_map((fwd_x - cx) * zoom + cx, (fwd_y - cy) * zoom + cy, H, W)

    disparity = np.sqrt((map_u - u_grid) ** 2 + (map_v - v_grid) ** 2)
    rmse = np.sqrt(np.nanmean(disparity ** 2))
    p95 = np.nanpercentile(disparity, 95)
    print(f"Z0={Z0}, zoom={zoom:.4f}")
    print(f"pixel movement: RMSE={rmse:.2f}px, max={np.nanmax(disparity):.2f}px, p95={p95:.2f}px")

    # Underwater view of the board: each pixel sees the world point traced through the housing.
    P_world = np.stack([(fwd_x - cx) / fx * Z0, (fwd_y - cy) / fy * Z0,
                        np.full((H, W), Z0)], axis=-1)
    img_underwater = make_checkerboard(P_world, BOARD_COLS, BOARD_ROWS, SQUARE_SIZE).astype(np.uint8)

    map_u_f = np.nan_to_num(map_u, nan=-1).astype(np.float32)
    map_v_f = np.nan_to_num(map_v, nan=-1).astype(np.float32)
    img_corrected = cv2.remap(img_underwater, map_u_f, map_v_f, cv2.INTER_LINEAR)

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle(f"rflat={housing['rflat']}, tglass={housing['tglass']}, "
                 f"Z0={Z0}, zoom={zoom:.4f}", fontsize=12)
    axes[0].imshow(img_underwater, cmap=blue_gray, vmin=0, vmax=255)
    axes[0].set_title(f"underwater ({W}x{H})")
    axes[1].imshow(img_corrected, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title("corrected")
    im = axes[2].imshow(disparity, cmap='hot', origin='upper')
    axes[2].set_title(f"pixel displacement\nRMSE={rmse:.2f}px")
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    axes[0].axis('off')
    axes[1].axis('off')
    plt.tight_layout()

    plot_radial_disparity(disparity, u_grid, v_grid, cx, cy)
    plt.show()


if __name__ == "__main__":
    main()
