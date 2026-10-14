"""Loading and unpacking of the YAML configuration (see configs/)."""
import warnings

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
    if "rflat" in hs:
        if "d0" in hs:
            raise ValueError("housing: set either d0 or the deprecated rflat, not both")
        warnings.warn("housing.rflat is deprecated; rename it to d0", DeprecationWarning, stacklevel=2)
        hs = {**hs, "d0": hs["rflat"]}
    return dict(
        n_port=np.array(hs["n_port"], dtype=float),
        d0=hs["d0"], tglass=hs["tglass"],
        mu_a=hs["mu_a"], mu_g=hs["mu_g"], mu_w=hs["mu_w"],
    )
