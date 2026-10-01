"""Iterative refraction correction: per-pixel Newton solve of the inverse trace."""
import numpy as np
from core.optics import trace_underwater, get_inair_world


def _jacobian(u, v, depth_at_uv, kwargs, h=0.5):
    """Central-difference Jacobian d(Px, Py)/d(u, v) of the underwater trace."""
    Pu1 = trace_underwater(u + h, v, Z0=depth_at_uv, **kwargs)
    Pu0 = trace_underwater(u - h, v, Z0=depth_at_uv, **kwargs)
    Pv1 = trace_underwater(u, v + h, Z0=depth_at_uv, **kwargs)
    Pv0 = trace_underwater(u, v - h, Z0=depth_at_uv, **kwargs)

    dPx_du = (Pu1[..., 0] - Pu0[..., 0]) / (2*h)
    dPy_du = (Pu1[..., 1] - Pu0[..., 1]) / (2*h)
    dPx_dv = (Pv1[..., 0] - Pv0[..., 0]) / (2*h)
    dPy_dv = (Pv1[..., 1] - Pv0[..., 1]) / (2*h)

    return np.stack([np.stack([dPx_du, dPx_dv], axis=-1),
                      np.stack([dPy_du, dPy_dv], axis=-1)], axis=-2)


def _solve(P_air, u0, v0, depth_map, kwargs, W, H, max_iter=20, tol=1e-9, damping=1.0):
    """Finds (u, v) such that trace_underwater(u, v) matches P_air in X and Y."""
    shape  = u0.shape
    u_flat = u0.astype(np.float64).ravel().copy()
    v_flat = v0.astype(np.float64).ravel().copy()
    P_air_flat = P_air.reshape(-1, P_air.shape[-1])
    depth_flat = depth_map.astype(np.float64).ravel()
    active     = np.ones(u_flat.shape, dtype=bool)

    for _ in range(max_iter):
        if not np.any(active):
            break
        idx = np.where(active)[0]
        u_act, v_act = u_flat[idx], v_flat[idx]
        P_air_act, depth_act = P_air_flat[idx], depth_flat[idx]

        P_water = trace_underwater(u_act, v_act, Z0=depth_act, **kwargs)
        r = P_water[..., :2] - P_air_act[..., :2]
        converged = np.sum(r*r, axis=-1) < tol
        active[idx[converged]] = False
        still = ~converged
        if not np.any(still):
            break

        u_act, v_act, depth_act, r = u_act[still], v_act[still], depth_act[still], r[still]
        idx_active = idx[still]

        J = _jacobian(u_act, v_act, depth_act, kwargs)
        det = J[..., 0, 0]*J[..., 1, 1] - J[..., 0, 1]*J[..., 1, 0]
        det_safe = np.where(np.abs(det) < 1e-12, np.nan, det)
        inv00, inv01 =  J[..., 1, 1]/det_safe, -J[..., 0, 1]/det_safe
        inv10, inv11 = -J[..., 1, 0]/det_safe,  J[..., 0, 0]/det_safe

        delta_u = inv00*r[..., 0] + inv01*r[..., 1]
        delta_v = inv10*r[..., 0] + inv11*r[..., 1]
        singular = np.isnan(delta_u) | np.isnan(delta_v)
        delta_u  = np.where(singular, 0.0, delta_u)
        delta_v  = np.where(singular, 0.0, delta_v)

        u_flat[idx_active] = np.clip(u_act - damping*delta_u, -W, 2*W)
        v_flat[idx_active] = np.clip(v_act - damping*delta_v, -H, 2*H)
        active[idx_active[singular]] = False

    return u_flat.reshape(shape), v_flat.reshape(shape)


def build_undistort_map_newton(fx, fy, cx, cy, depth, kwargs, W, H, zoom=1.4):
    """Undistortion map for cv2.remap via Newton's method.

    kwargs are the trace_underwater arguments (K_inv and the housing parameters).
    """
    u_grid, v_grid = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))

    if np.isscalar(depth):
        depth = np.full((H, W), depth, dtype=np.float64)

    fx_z, fy_z = fx * zoom, fy * zoom
    P_air = get_inair_world(u_grid, v_grid, fx_z, fy_z, cx, cy, depth)

    return _solve(P_air, u_grid, v_grid, depth, kwargs, W, H)
