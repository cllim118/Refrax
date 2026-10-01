"""Loading and unpacking of the YAML configuration (see configs/)."""
import numpy as np
import yaml


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def camera_params(cfg):
    """Returns (W, H, fx, fy, cx, cy) of the in-air pinhole camera."""
    cam = cfg["camera"]
    return cam["W"], cam["H"], cam["fx"], cam["fy"], cam["cx"], cam["cy"]


def distortion_params(cfg):
    """Returns the radial-tangential coefficients (k1, k2, p1, p2)."""
    cam = cfg["camera"]
    return tuple(cam.get(k, 0.0) for k in ("k1", "k2", "p1", "p2"))


def housing_params(cfg):
    """Flat-port housing parameters, as keyword arguments for core.optics/core.undistort."""
    hs = cfg["housing"]
    return dict(
        n_port=np.array(hs["n_port"], dtype=float),
        rflat=hs["rflat"], tglass=hs["tglass"],
        mu_a=hs["mu_a"], mu_g=hs["mu_g"], mu_w=hs["mu_w"],
    )
