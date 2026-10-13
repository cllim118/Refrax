"""Sensitivity of the corrected pixel positions to the refraction model parameters.

Usage:
    python evaluation/parameter_sensitivity.py configs/dataset.yaml [--out DIR] [--no-plot]

Every underwater pixel is mapped to its position in the corrected image with
the forward model of core/undistort.py (forward map at depth z0_fixed, scaled
by the zoom), for the nominal configuration and for one parameter perturbed at
a time: the scene depth Z0, the camera-to-port distance d0 (rflat), the glass
thickness t_g (tglass) and the refractive indices mu_w and mu_g. The
displacement between the perturbed and the nominal position is the error a
correction built with the nominal value makes when the true value is the
perturbed one. The zoom is fixed to its nominal value, as the corrected
images' calibration is.

A uniform scale about the principal point is equivalent to a focal-length
change, which SfM/SLAM can absorb, so the displacement is reported both raw and
as the residual after the best-fitting isotropic scale.

Linearity is measured by comparing the displacements of +delta and -delta: for
a linear response d(-delta) = -d(delta), so median|d(+) + d(-)| / median|d(+) - d(-)|
(the second-order term relative to the first-order one) is 0.

Outputs, in --out (default output/parameter_sensitivity):
    sensitivity.csv    median / p95 / max displacement per parameter and delta
    maps.png           displacement maps at each parameter's reference delta
    curves.png         median and p95 displacement against |delta|
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.config import load_config, camera_params, housing_params  # noqa: E402
from core.optics import matrix_K  # noqa: E402
from core.scale import select_zoom  # noqa: E402
from core.undistort import compute_housing_geometry, forward_map, zoom_forward_map  # noqa: E402

# Perturbation magnitudes per parameter (each applied with + and - sign), in the
# parameter's unit; Z0 is relative to z0_fixed. `ref` is the delta shown in maps.png.
PERTURBATIONS = {
    "Z0":     dict(label="Z0 (scene depth)",    unit="x z0", deltas=[0.05, 0.1, 0.25, 0.5],        ref=0.25),
    "rflat":  dict(label="d0 (camera-to-port)", unit="m",    deltas=[0.0005, 0.001, 0.002, 0.005], ref=0.002),
    "tglass": dict(label="t_g (glass)",         unit="m",    deltas=[0.0001, 0.00025, 0.0005, 0.001], ref=0.0005),
    "mu_w":   dict(label="mu_w (water)",        unit="",     deltas=[0.002, 0.005, 0.01, 0.02],    ref=0.01),
    "mu_g":   dict(label="mu_g (glass)",        unit="",     deltas=[0.01, 0.02, 0.05, 0.1],       ref=0.05),
}


def corrected_positions(W, H, fx, fy, cx, cy, housing, z, zoom):
    """(H, W, 2) position in the corrected image of every underwater pixel."""
    _, K_inv = matrix_K(fx, fy, cx, cy)
    P2, ray_water = compute_housing_geometry(H, W, K_inv, **housing)
    map_x, map_y = zoom_forward_map(*forward_map(P2, ray_water, z, fx, fy, cx, cy), cx, cy, zoom)
    return np.stack([map_x, map_y], axis=-1)


def perturbed(param, delta, housing, z0):
    """(housing, depth) with one parameter perturbed by delta."""
    if param == "Z0":
        return housing, z0 * (1.0 + delta)
    return dict(housing, **{param: housing[param] + delta}), z0


def remove_scale(d, r):
    """Residual of displacements d after the least-squares isotropic scale of r = q - (cx, cy)."""
    valid = np.all(np.isfinite(d) & np.isfinite(r), axis=-1)
    s = np.sum(d[valid] * r[valid]) / np.sum(r[valid] ** 2)
    return d - s * r


def stats(d):
    n = np.linalg.norm(d, axis=-1)
    n = n[np.isfinite(n)]
    return np.median(n), np.percentile(n, 95), n.max()


def linearity(d_plus, d_minus):
    """median|d(+) + d(-)| / median|d(+) - d(-)|: 0 for a linear response."""
    second = np.linalg.norm(d_plus + d_minus, axis=-1)
    first = np.linalg.norm(d_plus - d_minus, axis=-1)
    return np.nanmedian(second) / np.nanmedian(first)


def plot_maps(maps, W, H, path):
    """Raw and residual displacement magnitude per parameter, with sparse direction arrows."""
    fig, axes = plt.subplots(2, len(maps), figsize=(4.2 * len(maps), 5.6), squeeze=False)
    step = max(W, H) // 24
    v, u = np.mgrid[step // 2:H:step, step // 2:W:step]
    for col, (param, (delta, raw, res)) in enumerate(maps.items()):
        p = PERTURBATIONS[param]
        for row, (d, kind) in enumerate(((raw, "raw"), (res, "after scale"))):
            ax = axes[row, col]
            mag = np.linalg.norm(d, axis=-1)
            im = ax.imshow(mag, cmap="hot", origin="upper")
            dv = d[v, u]
            ax.quiver(u, v, dv[..., 0], -dv[..., 1], color="cyan", angles="uv", width=0.004)
            ax.set_title(f"{p['label']} +{delta:g} {p['unit']}\n{kind}: median {np.nanmedian(mag):.2f} px",
                         fontsize=9)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.03, label="px")
    fig.suptitle("Displacement in the corrected image, per underwater pixel", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_curves(rows, path):
    """Median and p95 displacement against |delta| per parameter, raw and after scale."""
    fig, axes = plt.subplots(1, len(PERTURBATIONS), figsize=(4 * len(PERTURBATIONS), 3.6), squeeze=False)
    for ax, (param, p) in zip(axes[0], PERTURBATIONS.items()):
        for kind, color in (("raw", "tab:blue"), ("residual", "tab:orange")):
            for sign, ls in ((1, "-"), (-1, "--")):
                sel = [r for r in rows if r["param"] == param and np.sign(r["delta"]) == sign]
                x = [abs(r["delta"]) for r in sel]
                ax.plot(x, [r[f"{kind}_median"] for r in sel], ls, marker="o", color=color,
                        label=f"{kind} median ({'+' if sign > 0 else '-'})")
                ax.plot(x, [r[f"{kind}_p95"] for r in sel], ls, marker="^", color=color, alpha=0.4)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"|delta| ({p['unit']})" if p["unit"] else "|delta|")
        ax.set_title(p["label"], fontsize=10)
        ax.grid(True, which="both", alpha=0.3)
    axes[0, 0].set_ylabel("displacement (px); circle median, triangle p95")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Parameter sensitivity of the refraction correction")
    parser.add_argument("config", help="Path to config YAML file")
    parser.add_argument("--out", default="output/parameter_sensitivity", help="Output directory")
    parser.add_argument("--no-plot", action="store_true", help="Only write and print the metrics")
    args = parser.parse_args()

    cfg = load_config(args.config)
    W, H, fx, fy, cx, cy = camera_params(cfg)
    housing = housing_params(cfg)
    corr = cfg["correction"]
    z0 = corr["z0_fixed"]
    zoom = corr.get("zoom")
    if zoom is None:
        zoom, _ = select_zoom(z0, W, H, fx, fy, cx, cy, housing,
                              in_bounds=not corr.get("fit_canvas", True))
    print(f"Nominal: Z0={z0}, rflat={housing['rflat']}, tglass={housing['tglass']}, "
          f"mu_w={housing['mu_w']}, mu_g={housing['mu_g']}, zoom={zoom:.4f}")

    q0 = corrected_positions(W, H, fx, fy, cx, cy, housing, z0, zoom)
    r = q0 - np.array([cx, cy])

    rows, maps = [], {}
    print(f"\n{'parameter':22s} {'delta':>9s} | {'raw median':>10s} {'p95':>7s} {'max':>7s} | "
          f"{'resid median':>12s} {'p95':>7s} {'max':>7s} | {'nonlin':>6s}")
    for param, p in PERTURBATIONS.items():
        for delta in p["deltas"]:
            d = {}
            for sign in (1, -1):
                h, z = perturbed(param, sign * delta, housing, z0)
                d[sign] = corrected_positions(W, H, fx, fy, cx, cy, h, z, zoom) - q0
            nonlin = linearity(d[1], d[-1])
            for sign in (1, -1):
                res = remove_scale(d[sign], r)
                (rm, rp, rx), (sm, sp, sx) = stats(d[sign]), stats(res)
                rows.append(dict(param=param, delta=sign * delta, raw_median=rm, raw_p95=rp, raw_max=rx,
                                 residual_median=sm, residual_p95=sp, residual_max=sx, nonlinearity=nonlin))
                print(f"{p['label']:22s} {sign * delta:+9.4g} | {rm:10.3f} {rp:7.3f} {rx:7.3f} | "
                      f"{sm:12.3f} {sp:7.3f} {sx:7.3f} | {nonlin:6.3f}")
                if sign == 1 and delta == p["ref"]:
                    maps[param] = (delta, d[sign], res)

    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "sensitivity.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {csv_path}")

    if not args.no_plot:
        plot_maps(maps, W, H, os.path.join(args.out, "maps.png"))
        plot_curves(rows, os.path.join(args.out, "curves.png"))
        print(f"Saved: {args.out}/maps.png, {args.out}/curves.png")


if __name__ == "__main__":
    main()
