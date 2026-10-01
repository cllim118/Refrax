"""Ray tracing through a flat-port housing (air -> glass -> water)."""
import numpy as np

EPS = 1e-12


def normalize_batch(v: np.ndarray) -> np.ndarray:
    """Normalises vectors along the last axis, leaving near-zero vectors unchanged."""
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.where(norm < EPS, v, v / norm)


def refract_batch(X, n, mu1, mu2):
    """Snell refraction of unit rays X at a surface with normal n (mu1 -> mu2).

    Rays undergoing total internal reflection are returned as NaN.
    """
    r     = mu1 / mu2
    cos_i = np.einsum('...i,i->...', X, n)
    term  = 1.0 - r**2 * (1.0 - cos_i**2)

    tir   = term < 0
    term  = np.where(tir, 0.0, term)

    coeff    = np.sqrt(term) - r * cos_i
    refracted = r * X + coeff[..., np.newaxis] * n
    refracted[tir] = np.nan
    return refracted


def intersect_plane_batch(origin, ray, plane_point, normal):
    """Intersection of rays origin + t * ray with a plane."""
    denom = np.einsum('...i,i->...', ray, normal)
    denom = np.where(np.abs(denom) < EPS, EPS, denom)

    diff = plane_point - origin
    t    = np.einsum('...i,i->...', diff, normal) / denom
    return origin + t[..., np.newaxis] * ray


def matrix_K(fx, fy, cx, cy):
    K     = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
    return K, np.linalg.inv(K)


def trace_underwater(u, v, K_inv, n_port, rflat, tglass, Z0,
                      mu_a, mu_g, mu_w):
    """Traces pixels (u, v) through the housing onto the plane Z = Z0.

    The port's inner surface lies at distance rflat from the camera centre
    along n_port, followed by glass of thickness tglass.

    Returns
    -------
    P_water : (..., 3) float64 — world point each pixel is viewing
    """
    # 1. Pixel coords → unit rays in air
    u, v = np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64)
    pixel = np.stack([u, v, np.ones_like(u)], axis=-1)
    ray_air = normalize_batch(pixel @ K_inv.T)

    # 2. Air → glass interface
    plane1 = n_port * rflat
    P1     = intersect_plane_batch(np.zeros(3), ray_air, plane1, n_port)

    # 3. Refract air → glass
    ray_glass = refract_batch(ray_air, n_port, mu_a, mu_g)

    # 4. Glass → water interface
    plane2 = plane1 + n_port * tglass
    P2     = intersect_plane_batch(P1, ray_glass, plane2, n_port)

    # 5. Refract glass → water
    ray_water = refract_batch(ray_glass, n_port, mu_g, mu_w)

    # 6. Intersect scene plane Z = Z0
    denom      = ray_water[..., 2]
    safe_denom = np.where(np.abs(denom) < EPS, EPS, denom)
    t          = (Z0 - P2[..., 2]) / safe_denom
    return P2 + t[..., np.newaxis] * ray_water


def apply_radtan_distortion(x_norm, y_norm, k1, k2, p1, p2):
    """Applies radial-tangential distortion to normalised image coordinates."""
    r2 = x_norm ** 2 + y_norm ** 2
    radial = 1 + k1 * r2 + k2 * r2 ** 2

    x_dist = x_norm * radial + 2 * p1 * x_norm * y_norm + p2 * (r2 + 2 * x_norm ** 2)
    y_dist = y_norm * radial + p1 * (r2 + 2 * y_norm ** 2) + 2 * p2 * x_norm * y_norm

    return x_dist, y_dist


def get_inair_world(u, v, fx, fy, cx, cy, Z0):
    """In-air pixel (u, v) -> world point at depth Z0 (pinhole, no refraction)."""
    X = (u - cx) / fx * Z0
    Y = (v - cy) / fy * Z0

    return np.stack([X, Y, np.full_like(X, Z0)], axis=-1)
