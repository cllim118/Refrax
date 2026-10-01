"""Closed-form refraction correction: forward map + scattered-data inversion."""
import numpy as np
from scipy.interpolate import griddata
from core.optics import normalize_batch, refract_batch, intersect_plane_batch


def compute_housing_geometry(H, W, K_inv, n_port, rflat, tglass, mu_a, mu_g, mu_w):
    """Depth-independent part of the trace: exit point P2 on the port and water ray per pixel."""
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    pixels = np.stack([u, v, np.ones((H, W), dtype=np.float32)], axis=-1)
    rays   = normalize_batch(pixels @ K_inv.T)

    plane1    = n_port * rflat
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


def invert_map(map_x, map_y, H, W, step=4):
    """Inverts a forward map by linear interpolation over a subsampled grid (every `step` px).

    Pixels outside the convex hull of the forward map are NaN.
    """
    u_s = np.arange(0, W, step)
    v_s = np.arange(0, H, step)
    ug_s, vg_s = np.meshgrid(u_s, v_s)

    src = np.stack([map_x[np.ix_(v_s, u_s)].ravel(),
                     map_y[np.ix_(v_s, u_s)].ravel()], axis=1)
    ug_full, vg_full = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))

    undist_x = griddata(src, ug_s.ravel().astype(np.float32), (ug_full, vg_full), method='linear')
    undist_y = griddata(src, vg_s.ravel().astype(np.float32), (ug_full, vg_full), method='linear')
    return undist_x.astype(np.float32), undist_y.astype(np.float32)


def build_undistort_map_closed_form(P2, ray_water, depth, fx, fy, cx, cy, H, W, zoom=None):
    """Undistortion map for cv2.remap: corrected pixel -> underwater pixel."""
    map_x, map_y = forward_map(P2, ray_water, depth, fx, fy, cx, cy)
    if zoom is not None:
        map_x = (map_x - cx) * zoom + cx
        map_y = (map_y - cy) * zoom + cy
    return invert_map(map_x, map_y, H, W)
