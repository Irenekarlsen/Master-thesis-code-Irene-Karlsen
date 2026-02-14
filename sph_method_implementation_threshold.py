from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter1d


# -----------------------------
# Global parameters
# -----------------------------
SPH_THRESHOLD = 10.0   # Default threshold (ng m-3)
SPH_MIN_BINS = 2       # Minimum number of bins above threshold per layer
MAX_LAYERS = 4         # How many layers to keep (ranked)
LAYER_RANKING = "thickness"  # "sum", "max", or "thickness"


def _as_1d_pair(z, beta):
    z = np.asarray(z, dtype=float)
    beta = np.asarray(beta, dtype=float)

    if z.ndim != 1 or beta.ndim != 1 or z.size != beta.size:
        raise ValueError("z and beta must be 1-D arrays of the same length.")
    if z.size < 2:
        raise ValueError("z and beta must have at least 2 elements.")
    if not np.all(np.diff(z) > 0):
        raise ValueError("z must be strictly increasing.")
    return z, beta


def gaussian_smooth_nan(x, sigma, mode="nearest"):
    """
    Gaussian smooth a 1-D series while ignoring NaNs (normalized convolution).
    Intended for along-track smoothing of detected heights.
    """
    x = np.asarray(x, float)
    m = np.isfinite(x).astype(float)
    xf = np.where(np.isfinite(x), x, 0.0)

    num = gaussian_filter1d(xf, sigma=sigma, mode=mode)
    den = gaussian_filter1d(m, sigma=sigma, mode=mode)

    out = num / den
    out[den == 0] = np.nan
    return out


def sph_layers(
    z,
    beta,
    threshold: float | None = None,
    min_bins: int | None = None,
    max_layers: int | None = None,
    ranking: str | None = None,
):
    """
    Detect multiple threshold-defined layers in a single 1-D profile.

    A layer is a contiguous run where beta >= threshold.

    Returns a dict:
      {
        "n_layers": int,
        "bottom": np.ndarray shape (n_layers,),
        "top": np.ndarray shape (n_layers,),
        "ext": np.ndarray shape (n_layers,),
        "score": np.ndarray shape (n_layers,),
        "k0": np.ndarray (start indices),
        "k1": np.ndarray (end indices, inclusive),
      }

    Layers are ranked by `ranking` and then capped to `max_layers`. Returned
    layers are then sorted by altitude (bottom ascending) for plotting.
    """
    z, beta = _as_1d_pair(z, beta)

    thr = SPH_THRESHOLD if threshold is None else float(threshold)
    mb = int(SPH_MIN_BINS if min_bins is None else min_bins)
    ml = int(MAX_LAYERS if max_layers is None else max_layers)
    rk = str(LAYER_RANKING if ranking is None else ranking).lower()

    # Define where layer exists
    in_layer = np.isfinite(beta) & (beta >= thr)

    # Find contiguous runs of True in in_layer
    x = in_layer.astype(np.int8)
    dx = np.diff(np.r_[0, x, 0])
    starts = np.where(dx == 1)[0]          # inclusive
    ends_excl = np.where(dx == -1)[0]      # exclusive

    # Each run is [s, e-1]
    k0_list = []
    k1_list = []
    bottom_list = []
    top_list = []
    ext_list = []
    score_list = []

    for s, e in zip(starts, ends_excl):
        if e <= s:
            continue

        k0 = int(s)
        k1 = int(e - 1)

        nbin = (k1 - k0 + 1)
        if nbin < mb:
            continue


        bottom = float(z[k0])

        # Exit boundary: first bin after the run (if exists), else last bin in run
        if (k1 + 1) < z.size:
            top = float(z[k1 + 1])
        else:
            top = float(z[k1])

        # Weighted mean within the run (weights = beta)
        w = beta[k0 : k1 + 1]
        zw = z[k0 : k1 + 1]
        sw = float(np.nansum(w))
        if not np.isfinite(sw) or sw <= 0:
            ext = np.nan
        else:
            ext = float(np.nansum(w * zw) / sw)

        # Layer score for ranking
        if rk == "sum":
            score = float(np.nansum(w))
        elif rk == "max":
            score = float(np.nanmax(w))
        elif rk == "thickness":
            score = float(top - bottom)
        else:
            raise ValueError(f"Unknown ranking={ranking!r}. Use 'sum', 'max', or 'thickness'.")

        k0_list.append(k0)
        k1_list.append(k1)
        bottom_list.append(bottom)
        top_list.append(top)
        ext_list.append(ext)
        score_list.append(score)

    bottom_arr = np.asarray(bottom_list, dtype=float)
    top_arr = np.asarray(top_list, dtype=float)
    ext_arr = np.asarray(ext_list, dtype=float)
    score_arr = np.asarray(score_list, dtype=float)
    k0_arr = np.asarray(k0_list, dtype=int)
    k1_arr = np.asarray(k1_list, dtype=int)

    # Rank by score desc, keep top ml
    order = np.argsort(score_arr)[::-1]
    if order.size > ml:
        order = order[:ml]

    bottom_arr = bottom_arr[order]
    top_arr = top_arr[order]
    ext_arr = ext_arr[order]
    score_arr = score_arr[order]
    k0_arr = k0_arr[order]
    k1_arr = k1_arr[order]

    # Sort selected layers by altitude for nicer downstream plotting
    alt_order = np.argsort(bottom_arr)
    bottom_arr = bottom_arr[alt_order]
    top_arr = top_arr[alt_order]
    ext_arr = ext_arr[alt_order]
    score_arr = score_arr[alt_order]
    k0_arr = k0_arr[alt_order]
    k1_arr = k1_arr[alt_order]

    return dict(
        n_layers=int(bottom_arr.size),
        bottom=bottom_arr,
        top=top_arr,
        ext=ext_arr,
        score=score_arr,
        k0=k0_arr,
        k1=k1_arr,
    )


def sph_bottom(z, beta, threshold: float | None = None, all_layers: bool = False):
    """
    Threshold-based bottom height(s).

    If all_layers=False (default): returns bottom of the *best* layer (float or None).
    If all_layers=True: returns bottoms for up to MAX_LAYERS (np.ndarray, possibly empty).
    """
    layers = sph_layers(z=z, beta=beta, threshold=threshold, max_layers=MAX_LAYERS)
    if layers["n_layers"] < 1:
        return np.array([]) if all_layers else None
    return layers["bottom"] if all_layers else float(layers["bottom"][0])


def sph_top(z, beta, threshold: float | None = None, all_layers: bool = False):
    """
    Threshold-based top height(s).

    If all_layers=False (default): returns top of the *best* layer (float or None).
    If all_layers=True: returns tops for up to MAX_LAYERS (np.ndarray, possibly empty).
    """
    layers = sph_layers(z=z, beta=beta, threshold=threshold, max_layers=MAX_LAYERS)
    if layers["n_layers"] < 1:
        return np.array([]) if all_layers else None
    return layers["top"] if all_layers else float(layers["top"][0])


def sph_ext(beta, z, threshold: float | None = None, all_layers: bool = False):
    """
    Threshold-based weighted-mean height(s) (sph_ext).

    If all_layers=False (default): returns ext of the *best* layer (float or np.nan).
    If all_layers=True: returns ext for up to MAX_LAYERS (np.ndarray, possibly empty).
    """
    layers = sph_layers(z=z, beta=beta, threshold=threshold, max_layers=MAX_LAYERS)
    if layers["n_layers"] < 1:
        return np.array([]) if all_layers else np.nan
    return layers["ext"] if all_layers else float(layers["ext"][0])
