"""Planarity of one or more COLMAP reconstructions of the same planar scene.

Usage:
    python evaluation/planarity.py                  # the clouds in INPUTS
    python evaluation/planarity.py A/points3D.txt [B/points3D.txt ...] [--names A B ...]

Inputs are COLMAP sparse models in text format; convert a binary model with
    colmap model_converter --input_path sparse/0 --output_path sparse/0 --output_type TXT

A plane is fitted to each sparse cloud and the out-of-plane residuals are
reported, normalised by the in-plane diagonal D of the cloud. The first cloud is
the reference: all others are scaled so that their in-plane diagonal matches
it, and their plane normals are oriented to agree with it. All clouds are then
voxel-downsampled with the same voxel size, percentile filtered and randomly
subsampled to the same number of points before the plane fit.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Default inputs, label -> COLMAP points3D.txt; edit to point at your reconstructions.
# The first entry is the reference for scale and normal alignment.
INPUTS = {
    "COLMAP_A": "path/to/colmap/output/points3D.txt",
    "COLMAP_B": "path/to/colmap/output/points3D.txt",
    "COLMAP_C": "path/to/colmap/output/points3D.txt",
}

MAX_POINTS = 100_000


def load_points(path):
    """XYZ of a COLMAP points3D.txt."""
    return np.loadtxt(path, comments="#", usecols=(1, 2, 3))


def voxel_downsample(points, size):
    idx = np.floor(points / size).astype(np.int64)
    _, keep = np.unique(idx, axis=0, return_index=True)
    return points[keep]


def percentile_filter(points, p=1):
    lo, hi = np.percentile(points, [p, 100 - p], axis=0)
    return points[np.all((points >= lo) & (points <= hi), axis=1)]


def sample(points, n, seed=0):
    if len(points) <= n:
        return points
    rng = np.random.default_rng(seed)
    return points[rng.choice(len(points), n, replace=False)]


def plane_axes(n):
    """Orthonormal in-plane axes (u, v) for a plane with normal n."""
    n = n / np.linalg.norm(n)
    ref = np.array([1., 0., 0.]) if abs(n[0]) < .9 else np.array([0., 1., 0.])
    u = np.cross(n, ref)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    v /= np.linalg.norm(v)
    return u, v


def fit_plane(points):
    """Least-squares plane through the centroid. Returns (center, unit normal)."""
    center = points.mean(0)
    _, _, vh = np.linalg.svd(points - center, full_matrices=False)
    return center, vh[-1] / np.linalg.norm(vh[-1])


def plane_diagonal(points):
    """Diagonal of the bounding box of the points in the fitted plane's (u, v) frame."""
    center, normal = fit_plane(points)
    u_axis, v_axis = plane_axes(normal)
    return np.hypot(np.ptp((points - center) @ u_axis), np.ptp((points - center) @ v_axis))


def analyze(points, ref_normal=None):
    """Plane fit and residuals.

    dist   signed point-to-plane distance
    u, v   in-plane coordinates of the projected points
    D      in-plane diagonal, used to normalise dist
    angle  elevation (deg) of each point seen from the plane centre; points
           within 1% of the 99th-percentile radius of the centre are excluded
    If ref_normal is given, the normal is flipped to agree with it so that
    signed residuals of two reconstructions are comparable.
    """
    center, normal = fit_plane(points)
    if ref_normal is not None and np.dot(normal, ref_normal) < 0:
        normal = -normal

    dist = (points - center) @ normal
    proj = points - np.outer(dist, normal)

    u_axis, v_axis = plane_axes(normal)
    u = (proj - center) @ u_axis
    v = (proj - center) @ v_axis
    D = np.hypot(np.ptp(u), np.ptp(v))

    radius = np.linalg.norm(proj - center, axis=1)
    valid = radius > .01 * np.percentile(radius, 99)
    angle = np.degrees(np.arctan2(dist[valid], radius[valid]))

    return {
        "points": points,
        "center": center,
        "normal": normal,
        "dist": dist,
        "u": u,
        "v": v,
        "angle": angle,
        "valid": valid,
        "D": D,
    }


def fit_quadric(r):
    """Least-squares fit dist ~ a + b*u + c*v + c3*(u^2 + v^2) of the plane residuals.

    Adds the fitted surface "quad" and the curvature coefficient "c3" to r.
    """
    u, v = r["u"], r["v"]
    A = np.column_stack([np.ones_like(u), u, v, u ** 2 + v ** 2])
    coef, *_ = np.linalg.lstsq(A, r["dist"], rcond=None)
    r["quad"] = A @ coef
    r["c3"] = coef[3]
    return r


def stats(r):
    dist, angle, D, c3 = r["dist"], r["angle"], r["D"], r["c3"]
    rmse = np.sqrt(np.mean(dist ** 2))
    p1, p99 = np.percentile(dist, [1, 99])

    return {
        "Points": f"{len(dist):,}",
        "Diagonal": f"{D:.4f}",
        "RMSE": f"{rmse:.6f}",
        "Relative RMSE [%]": f"{rmse / D * 100:.4f}",
        "Relative Median [%]": f"{np.median(np.abs(dist)) / D * 100:.4f}",
        "Relative PTV [%]": f"{(p99 - p1) / D * 100:.4f}",
        "Angular RMSE [deg]": f"{np.sqrt(np.mean(angle ** 2)):.4f}",
        "Angular Median [deg]": f"{np.median(np.abs(angle)):.4f}",
        "Angular |P99| [deg]": f"{np.percentile(np.abs(angle), 99):.4f}",
        "Curvature c3*D": f"{c3 * D:.6f}",
        # Sagitta of c3 * r^2 at r = D/2, relative to D
        "Sagitta/D [%]": f"{c3 * D / 4 * 100:.4f}",
    }


def print_table(names, results):
    cols = [stats(r) for r in results]
    keys = list(cols[0])

    label_w = max(len(k) for k in keys)
    widths = [max(len(n), *(len(c[k]) for k in keys)) for n, c in zip(names, cols)]

    print("\n" + " " * label_w + "  "
          + "  ".join(n.rjust(w) for n, w in zip(names, widths)))
    for k in keys:
        print(k.ljust(label_w) + "  "
              + "  ".join(c[k].rjust(w) for c, w in zip(cols, widths)))


def plot_maps(names, data, clim, label):
    """One (u, v) scatter per cloud, coloured by value on a shared scale [-clim, clim]."""
    lim = max(max(np.abs(u).max(), np.abs(v).max()) for u, v, _ in data)

    n = len(data)
    fig, ax = plt.subplots(1, n, figsize=(3.4 * n + 1.4, 3.6), squeeze=False)

    for a, name, (u, v, val) in zip(ax[0], names, data):
        sc = a.scatter(u, v, c=val, cmap="coolwarm", vmin=-clim, vmax=clim,
                       s=4, edgecolors="none")
        a.scatter(0, 0, marker="x", c="k", s=40)
        a.set(title=name, xlabel=r"$u/D$", ylabel=r"$v/D$",
              xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal")

    fig.subplots_adjust(right=0.87, wspace=0.3)
    cbar_ax = fig.add_axes([0.89, 0.15, 0.02, 0.7])
    fig.colorbar(sc, cax=cbar_ax, label=label)


def plot_violins(names, results, kde_points=20_000):
    panels = [
        ([np.abs(r["dist"]) / r["D"] * 100 for r in results], r"Normalized residual $|r|/D$ [%]"),
        ([np.abs(r["angle"]) for r in results], r"Angular residual $|\theta|$ [deg]"),
    ]

    fig, ax = plt.subplots(1, 2, figsize=(2.2 * len(results) + 4, 4))

    rng = np.random.default_rng(0)
    for a, (series, label) in zip(ax, panels):
        series = [s if len(s) <= kde_points
                  else s[rng.choice(len(s), kde_points, replace=False)]
                  for s in series]

        parts = a.violinplot(series, showmedians=True, showextrema=False,
                             quantiles=[[.25, .75]] * len(series))
        for body in parts["bodies"]:
            body.set_alpha(.6)

        ymax = max(np.percentile(s, 99.5) for s in series)
        a.axhline(0, color="k", lw=.5)
        a.set(ylabel=label, ylim=(0, ymax))
        a.set_xticks(np.arange(1, len(names) + 1), names,
                     rotation=20, ha="right", fontsize=8)

    fig.tight_layout()


def main():
    parser = argparse.ArgumentParser(
        description="Planarity of one or more COLMAP point clouds. "
                    "The first path is the reference for scale and normal alignment.")
    parser.add_argument("paths", nargs="*",
                        help="points3D.txt paths (default: INPUTS in this file)")
    parser.add_argument("--names", nargs="*",
                        help="labels, one per path (default: parent directory names)")
    parser.add_argument("--no-plot", action="store_true", help="Only print the metrics")
    args = parser.parse_args()

    if args.paths:
        paths = args.paths
        names = args.names or [Path(p).parent.name for p in paths]
    else:
        paths, names = list(INPUTS.values()), args.names or list(INPUTS)
    if len(names) != len(paths):
        parser.error(f"got {len(names)} --names for {len(paths)} paths")

    found = [(p, n) for p, n in zip(paths, names) if Path(p).exists()]
    for p in sorted(set(paths) - {p for p, _ in found}):
        print(f"Missing, skipped: {p}")
    if not found:
        parser.error("none of the input point clouds exist; edit INPUTS or pass paths")
    paths, names = map(list, zip(*found))

    clouds = [load_points(p) for p in paths]
    for name, pts in zip(names, clouds):
        print(f"Raw {name}: {len(pts):,}")

    # Scale alignment: all -> first (COLMAP reconstructions are up to scale)
    ref_diag = plane_diagonal(clouds[0])
    for i in range(1, len(clouds)):
        scale = ref_diag / plane_diagonal(clouds[i])
        clouds[i] *= scale
        print(f"{names[i]} scale: {scale:.6e}")

    # Same voxel size (0.1% of the reference in-plane diagonal) and filtering for all
    voxel_size = ref_diag * 0.001
    clouds = [percentile_filter(voxel_downsample(pts, voxel_size)) for pts in clouds]
    for name, pts in zip(names, clouds):
        print(f"After voxel/filter {name}: {len(pts):,}")

    # Same number of points
    N = min(MAX_POINTS, *(len(pts) for pts in clouds))
    clouds = [sample(pts, N) for pts in clouds]
    print(f"Evaluation points: {N:,}")

    ref = analyze(clouds[0])
    results = [ref] + [analyze(pts, ref_normal=ref["normal"]) for pts in clouds[1:]]
    results = [fit_quadric(r) for r in results]
    print_table(names, results)

    if args.no_plot:
        return

    plot_maps(names,
              [(r["u"] / r["D"], r["v"] / r["D"], r["dist"] / r["D"]) for r in results],
              max(np.percentile(np.abs(r["dist"] / r["D"]), 99) for r in results),
              r"Normalized residual $r/D$")
    plot_maps(names,
              [(r["u"][r["valid"]] / r["D"], r["v"][r["valid"]] / r["D"], r["angle"])
               for r in results],
              max(np.percentile(np.abs(r["angle"]), 99) for r in results),
              r"Angular residual $\theta$ [deg]")
    plot_maps(names,
              [(r["u"] / r["D"], r["v"] / r["D"], r["quad"] / r["D"]) for r in results],
              max(np.abs(r["quad"] / r["D"]).max() for r in results),
              r"Quadric fit $q/D$")
    plot_violins(names, results)
    plt.show()


if __name__ == "__main__":
    main()
