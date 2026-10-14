"""Closed-form refraction correction: forward map + scattered-data inversion."""
import numpy as np
from scipy.interpolate import griddata
from core.optics import normalize_batch, refract_batch, intersect_plane_batch


def compute_housing_geometry(H, W, K_inv, n_port, d0, tglass, mu_a, mu_g, mu_w):
    """Depth-independent part of the trace: exit point P2 on the port and water ray per pixel."""
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    pixels = np.stack([u, v, np.ones((H, W), dtype=np.float32)], axis=-1)
    rays   = normalize_batch(pixels @ K_inv.T)

    plane1    = n_port * d0
    P1        = intersect_plane_batch(np.zeros(3), rays, plane1, n_port)
    ray_glass = refract_batch(rays, n_port, mu_a, mu_g)

    plane2    = plane1 + n_port * tglass
    P2        = intersect_plane_batch(P1, ray_glass, plane2, n_port)
    ray_water = refract_batch(ray_glass, n_port, mu_g, mu_w)
    return P2, ray_water


def forward_map(P2, ray_water, depth, fx, fy, cx, cy):
    """Pinhole projection (fx, fy, cx, cy) of the point each underwater pixel sees at `depth`."""
    Z = np.asarray(depth, dtype=np.float32)
    if Z.ndim == 0:
        Z = np.full(P2.shape[:2], float(Z), dtype=np.float32)

    denom      = ray_water[..., 2]
    safe_denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    t          = (Z - P2[..., 2]) / safe_denom

    Pw_x = P2[..., 0] + t * ray_water[..., 0]
    Pw_y = P2[..., 1] + t * ray_water[..., 1]

    map_x = (fx * Pw_x / Z + cx).astype(np.float32)
    map_y = (fy * Pw_y / Z + cy).astype(np.float32)
    return map_x, map_y


def invert_map(map_x, map_y, H, W, step=4, origin=(0.0, 0.0)):
    """Inverts a forward map by linear interpolation over a subsampled grid (every `step` px).

    map_x, map_y give, per source pixel, its position in the corrected image plane. The
    inverse is evaluated on an H x W output canvas whose pixel (0, 0) sits at corrected-plane
    coordinate `origin`; (0, 0) with the source size is the source frame itself. The last
    source row and column are always sites, so the hull reaches the source border.
    Pixels outside the convex hull of the forward map are NaN.
    """
    src_h, src_w = map_x.shape
    u_s = np.unique(np.append(np.arange(0, src_w, step), src_w - 1))
    v_s = np.unique(np.append(np.arange(0, src_h, step), src_h - 1))
    ug_s, vg_s = np.meshgrid(u_s, v_s)

    src = np.stack([map_x[np.ix_(v_s, u_s)].ravel(),
                     map_y[np.ix_(v_s, u_s)].ravel()], axis=1)
    finite = np.all(np.isfinite(src), axis=1)
    src, ug_s, vg_s = src[finite], ug_s.ravel()[finite], vg_s.ravel()[finite]

    x0, y0 = origin
    ug_full, vg_full = np.meshgrid(np.arange(W, dtype=np.float32) + x0,
                                   np.arange(H, dtype=np.float32) + y0)

    undist_x = griddata(src, ug_s.astype(np.float32), (ug_full, vg_full), method='linear')
    undist_y = griddata(src, vg_s.astype(np.float32), (ug_full, vg_full), method='linear')
    return undist_x.astype(np.float32), undist_y.astype(np.float32)


def zoom_forward_map(map_x, map_y, cx, cy, zoom):
    """Scales a forward map about (cx, cy) by zoom (None: unchanged).

    Computed in float64 so the result does not depend on whether zoom is a Python
    or a NumPy float.
    """
    if zoom is None:
        return map_x, map_y
    return ((map_x.astype(np.float64) - cx) * float(zoom) + cx,
            (map_y.astype(np.float64) - cy) * float(zoom) + cy)


def forward_map_canvas(map_x, map_y):
    """Smallest canvas holding every finite point of a forward map, as (W, H, origin)."""
    finite = np.isfinite(map_x) & np.isfinite(map_y)
    if not np.any(finite):
        raise ValueError("Forward map has no finite entries; cannot size a canvas.")
    x_min, x_max = np.floor(map_x[finite].min()), np.ceil(map_x[finite].max())
    y_min, y_max = np.floor(map_y[finite].min()), np.ceil(map_y[finite].max())
    return int(x_max - x_min) + 1, int(y_max - y_min) + 1, (float(x_min), float(y_min))


def build_undistort_map_closed_form(P2, ray_water, depth, fx, fy, cx, cy, H, W, zoom=None,
                                    origin=(0.0, 0.0)):
    """Undistortion map for cv2.remap: corrected pixel -> underwater pixel.

    The map covers an H x W canvas with pixel (0, 0) at corrected-plane coordinate
    `origin` (see invert_map).
    """
    map_x, map_y = forward_map(P2, ray_water, depth, fx, fy, cx, cy)
    map_x, map_y = zoom_forward_map(map_x, map_y, cx, cy, zoom)
    return invert_map(map_x, map_y, H, W, origin=origin)
