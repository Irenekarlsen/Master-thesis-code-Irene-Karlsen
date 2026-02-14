"""
FLEXPART ↔ EarthCARE (ATL_TC / ATL_EBD / ATL_ALD) ↔ GFAS plotting

Refactor notes (behavior-preserving):
- Replaced most globals() lookups with a config dataclass.
- Removed large duplicate / commented-out algorithm blocks.
- Consolidated repeated “choose variable” and “mask like plot” logic.
- Standardized naming (track/time/height grids) and array orientation.
- Kept outputs in the returned dict compatible with your downstream plotting code.
"""

from __future__ import annotations
import os
import glob
import re
import warnings
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, Optional, Tuple, Iterable, Any
from typing import List

import h5py
import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, ListedColormap, BoundaryNorm
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

import cartopy.feature as cfeature
from cartopy import crs as ccrs
from cartopy.io import DownloadWarning

from pyproj import Geod
from netCDF4 import num2date
from scipy.interpolate import interp1d
from scipy import ndimage as ndi

from sph_method_implementation_threshold import (
    SPH_THRESHOLD,
    SPH_MIN_BINS,
    MAX_LAYERS,
    sph_layers,
    sph_ext,
    sph_bottom,
    sph_top,
    gaussian_smooth_nan,
)

# -----------------------------
# Warnings / Matplotlib defaults
# -----------------------------
warnings.simplefilter("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="facecolor will have no effect*", category=UserWarning)
warnings.filterwarnings("ignore", message="The input coordinates to pcolormesh*", category=UserWarning)
warnings.filterwarnings("ignore", message="set_ticklabels() should only be used*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*set_ticklabels.*FixedLocator.*")
warnings.simplefilter("ignore", category=DownloadWarning)
plt.ioff()


# -----------------------------
# Configuration
# -----------------------------
@dataclass(frozen=True)
class PlotConfig:
    # Date filters
    only_dates: Optional[Iterable[str]] = None                 # e.g. ["20250601", "20250602"] or ["2025-06-01"]
    date_range: Optional[Tuple[str, str]] = ("2025-05-25", "2025-06-15")

    # Replot / quality / resolution
    force_replot: bool = True
    quality_flags: Tuple[int, ...] = (0, 1)
    resolution: str = "low"           

    # SPH
    sph_smooth_sigma: Optional[float] = None
    sph_min_bins: int = SPH_MIN_BINS
    max_layers: int = MAX_LAYERS

    sph_threshold_ext: float = 1e-6 
    sph_threshold_flex: float = SPH_THRESHOLD

    # --- Optional: restrict ATL_TC-based SPH to smoke-like TC classes ---
    only_smoke_for_tc_sph: bool = True

    # TC class IDs considered "smoke". Defaults cover Smoke (13), Dusty smoke (14),
    # and Stratospheric smoke (27) in the ATL_TC classification.
    tc_smoke_class_ids: Tuple[int, ...] = (13, 14, 27)


    # Cloud masking (EBD)
    reduce_classifications: bool = True
    cloud_pad_time: int = 1
    cloud_pad_z: int = 1
    filter_out_ebd_values_above: Optional[float] = None
    no_clouds_in_flexpart: bool = False

    # Remove small pixel groups
    remove_pixel_groups_below: Optional[int] = 10   # e.g. 20
    remove_pixel_groups_connectivity: int = 2         # 1=4-neigh, 2=8-neigh

    # FLEXPART vertical handling
    set_flexpart_res_to_original: bool = True

    # GFAS
    gfas_min_frp: float = 0.1
    only_overpasses_near_gfas: bool = False
    gfas_max_distance_km: float = 100.0
    use_canada_only: bool = True
    canada_bbox: Dict[str, float] = None

    # --- GFAS altitude variable(s) ---
    gfas_altitude_var_candidates: Tuple[str, ...] = ("mami", "apt", "apb", "injh")

    # --- Plume selection on EarthCARE track ---
    # "plume exists" if this EarthCARE SPH series is finite at the track index
    plume_presence_series: str = "sph_top_on_tc_dt"  # options: sph_top_on_tc_dt, sph_ext_on_tc_dt, sph_bottom_on_tc_dt
    plume_search_max_km: float = 200.0              # if nearest point has no plume, search up to this distance along-track window

    # Plot toggles
    plot_atl_ald_heights: bool = False
    plot_sph_ext: bool = True
    plot_sph_top: bool = True
    plot_sph_bottom: bool = True

    # EarthCARE-only simplified plot (TC + EBD extinction), RAW/untampered
    plot_only_TC_and_extinction_alone: bool = True

    # Map / rendering
    flexpart_column_threshold_map: float = 200.0
    flexpart_conc_norm: Tuple[float, float] = (0.0, 1000.0)     # Normalize(vmin, vmax)

    def __post_init__(self):
        if object.__getattribute__(self, "canada_bbox") is None:
            object.__setattr__(
                self,
                "canada_bbox",
                dict(lat_min=41.6, lat_max=83.1, lon_min=-141.0, lon_max=-52.6),
            )


from dataclasses import dataclass

@dataclass
class FontSizes:
    base: int = 14          # overall baseline
    title: int = 18
    axes_label: int = 12
    tick: int = 12
    legend: int = 12
    cbar_label: int = 12
    cbar_tick: int = 12

def apply_rcparams(fs: FontSizes):
    plt.rcParams.update({
        "font.size": fs.base,
        "axes.titlesize": fs.title,
        "axes.labelsize": fs.axes_label,
        "xtick.labelsize": fs.tick,
        "ytick.labelsize": fs.tick,
        "legend.fontsize": fs.legend,
        "figure.titlesize": fs.title,
        # Optional but often helpful:
        # "axes.titlepad": 10,
        # "axes.labelpad": 8,
    })



# -----------------------------
# Utilities
# -----------------------------
def wrap_longitudes(lon: np.ndarray) -> np.ndarray:
    """Wrap longitudes to [-180, 180)."""
    lon = np.asarray(lon, dtype=float)
    return ((lon + 180.0) % 360.0) - 180.0


def ensure_2d_time_height(arr: np.ndarray, n_time: int, n_z: int) -> np.ndarray:
    """
    Ensure array is shaped (time, z). If it matches (z, time), transpose it.
    Otherwise return as-is.
    """
    a = np.asarray(arr)
    if a.ndim != 2:
        return a
    if a.shape == (n_time, n_z):
        return a
    if a.shape == (n_z, n_time):
        return a.T
    return a


def mask_clouds_on_tc_grid(arr_on_tc: Optional[np.ndarray], class_data: np.ndarray, cloud_classes=(1, 2, 3)) -> Optional[np.ndarray]:
    """
    Mask an array (time, z) using TC classification on the same grid.
    Returns a float array with clouds set to NaN.
    """
    if arr_on_tc is None:
        return None
    arr = np.array(arr_on_tc, copy=True, dtype=float)
    cls = np.asarray(class_data)
    if arr.shape != cls.shape:
        return arr
    cloud_mask = np.isin(cls, cloud_classes)
    arr[cloud_mask] = np.nan
    return arr


def dilate_mask_2d(mask: np.ndarray, dt_pad: int = 1, dz_pad: int = 1) -> np.ndarray:
    """Expand a 2D boolean mask in time (axis=0) and height (axis=1)."""
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2:
        return m

    out = m.copy()
    for k in range(1, int(dt_pad) + 1):
        out |= np.roll(m, k, axis=0)
        out |= np.roll(m, -k, axis=0)
    for k in range(1, int(dz_pad) + 1):
        out |= np.roll(m, k, axis=1)
        out |= np.roll(m, -k, axis=1)
    return out


def _fill_nonfinite_rows_2d(H: np.ndarray) -> np.ndarray:
    """Fill non-finite values in a 2D height array by row-wise interpolation, then nearest-row fill."""
    H = np.asarray(H, dtype=float)
    if H.ndim != 2:
        return H

    out = H.copy()
    nrow, ncol = out.shape
    x = np.arange(ncol)

    for i in range(nrow):
        row = out[i, :]
        m = np.isfinite(row)
        if m.all():
            continue
        if m.sum() == 0:
            continue
        out[i, :] = np.interp(x, x[m], row[m])

    good_rows = np.where(np.isfinite(out).all(axis=1))[0]
    if good_rows.size:
        for i in range(nrow):
            if not np.isfinite(out[i, :]).all():
                j = good_rows[np.argmin(np.abs(good_rows - i))]
                out[i, :] = out[j, :]
    return out


def _as_float_nan(a: Any) -> np.ndarray:
    """Convert to float array with masked values filled as NaN."""
    if np.ma.isMaskedArray(a):
        return np.ma.filled(a, np.nan).astype(float, copy=True)
    return np.array(a, dtype=float, copy=True)


def _mask_like_plot(arr: Any, kind: str) -> np.ndarray:
    """
    Apply plot-equivalent masks so SPH sees what the plot would show.
    """
    a = _as_float_nan(arr)

    if kind == "ebd_ext":
        a[~np.isfinite(a) | (a <= 0.0) | (a > 9.9)] = np.nan
        return a

    if kind == "flex_bc":
        a[~np.isfinite(a) | (a < 10.0)] = np.nan
        return a

    raise ValueError(f"Unknown kind={kind}")


def _interp_times(x_src: np.ndarray, y_src: np.ndarray, x_dst: np.ndarray) -> np.ndarray:
    """
    1D time interpolation over finite y points.
    x_src / x_dst are datetime64 arrays.
    """
    y_src = np.asarray(y_src, dtype=float)
    ok = np.isfinite(y_src)
    if ok.sum() < 2:
        return np.full(len(x_dst), np.nan, dtype=float)

    t_src = x_src.astype("datetime64[ns]").astype("int64")
    t_dst = x_dst.astype("datetime64[ns]").astype("int64")
    return np.interp(t_dst, t_src[ok], y_src[ok], left=np.nan, right=np.nan)


def _pick_first_existing(h5_group: h5py.Group, candidates: Iterable[str]) -> Optional[np.ndarray]:
    for v in candidates:
        if v in h5_group:
            try:
                return np.array(h5_group[v])
            except Exception:
                continue
    return None

def _tc_plot_classes_and_remap(
    class_tc: np.ndarray,
    *,
    tc_classes: dict,
    only_smoke: bool,
    smoke_ids: tuple[int, ...],
    include_clear: bool = True,
    include_surface: bool = True,
    include_missing: bool = True,
) -> tuple[np.ndarray, list[int], ListedColormap, BoundaryNorm]:
    """
    Returns:
      tc_remapped (same shape as class_tc, values 0..N-1),
      keep_classes (class IDs actually shown),
      cmap, norm
    """
    # Decide which class IDs are allowed to appear in plot + colorbar
    if only_smoke:
        keep = []
        if include_missing:
            # include whatever missing/noise keys exist in tc_classes
            for k in (-3, -2, -1):
                if k in tc_classes:
                    keep.append(k)
        # else:
        #     if (-3 in tc_classes):  # fallback for remap fill
        #         keep.append(-3)

        if include_surface and (-2 in tc_classes) and (-2 not in keep):
            keep.append(-2)

        if include_clear and (0 in tc_classes):
            keep.append(0)

        # smoke classes (only those present in tc_classes dict)
        keep += [cid for cid in smoke_ids if cid in tc_classes]

        # make unique but keep order
        seen = set()
        keep_classes = [x for x in keep if not (x in seen or seen.add(x))]
    else:
        keep_classes = sorted(tc_classes.keys())

    cid_to_idx = {cid: i for i, cid in enumerate(keep_classes)}

    # fill id: use missing if available, else 0
    fill_cid = -3 if (-3 in cid_to_idx) else (0 if 0 in cid_to_idx else keep_classes[0])

    tc_remapped = np.full_like(class_tc, fill_value=cid_to_idx[fill_cid], dtype=float)
    for cid, idx in cid_to_idx.items():
        tc_remapped[np.asarray(class_tc) == cid] = idx

    class_cmap = ListedColormap([tc_classes[cid][1] for cid in keep_classes])
    bounds = np.arange(-0.5, len(keep_classes) + 0.5, 1.0)
    class_norm = BoundaryNorm(bounds, len(keep_classes))

    return tc_remapped, keep_classes, class_cmap, class_norm


def _choose_ebd_extinction(ds_ebd: h5py.Group, resolution: str) -> Optional[np.ndarray]:
    candidates_by_res = {
        "high": [
            "particle_extinction_coefficient_355nm_high_resolution",
            "extinction_high_resolution",
            "particle_extinction_coefficient_355nm",
        ],
        "medium": [
            "particle_extinction_coefficient_355nm_medium_resolution",
            "extinction_medium_resolution",
            "particle_extinction_coefficient_355nm",
        ],
        "low": [
            "particle_extinction_coefficient_355nm_low_resolution",
            "extinction_low_resolution",
            "particle_extinction_coefficient_355nm",
        ],
    }
    resolution = (resolution or "low").lower().strip()
    ordered = candidates_by_res.get(resolution, []) + sum(candidates_by_res.values(), [])
    # unique keep order
    seen = set()
    ordered = [v for v in ordered if not (v in seen or seen.add(v))]
    return _pick_first_existing(ds_ebd, ordered)





def remove_small_pixel_groups_2d(
    arr: np.ndarray,
    *,
    min_size: Optional[int],
    connectivity: int = 2,
    foreground_mask: Optional[np.ndarray] = None,
    fill_value: float = np.nan,
) -> np.ndarray:
    """
    Remove connected foreground regions smaller than min_size by setting them to fill_value.

    arr: 2D (time, height) array
    foreground_mask: if None, uses np.isfinite(arr)
    connectivity: 1 -> 4-neighborhood, 2 -> 8-neighborhood
    """
    if min_size is None or int(min_size) <= 1:
        return arr

    a = np.array(arr, copy=True, dtype=float)
    if a.ndim != 2:
        return a

    fg = foreground_mask
    if fg is None:
        fg = np.isfinite(a)
    fg = np.asarray(fg, dtype=bool)
    if fg.shape != a.shape:
        return a

    # Define connectivity structure
    structure = ndi.generate_binary_structure(rank=2, connectivity=int(connectivity))

    labels, nlab = ndi.label(fg, structure=structure)
    if nlab == 0:
        return a

    # Count pixels per label (label 0 is background)
    counts = np.bincount(labels.ravel())
    # labels to remove: those with 1..nlab and count < min_size
    remove = np.zeros(nlab + 1, dtype=bool)
    remove[1:] = counts[1:] < int(min_size)

    # Apply removal
    a[remove[labels]] = fill_value
    return a





def _pick_first_existing_var(ds: xr.Dataset, candidates: Iterable[str]) -> Optional[str]:
    for v in candidates:
        if v in ds.variables:
            return v
    return None


def gfas_fire_points_with_altitude(
    ds_gfas: xr.Dataset,
    *,
    frp_var: str = "frpfire",
    time_index: int = 0,
    min_frp: float = 0.1,
    altitude_candidates: Iterable[str] = (),
    use_canada_only: bool = True,
    canada_bbox: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Returns a dataframe of GFAS fire pixels:
      lon, lat, frp, gfas_alt_km (optional), plus row/col indices for traceability.
    """
    if frp_var not in ds_gfas:
        raise KeyError(f"GFAS FRP variable '{frp_var}' not in dataset variables: {list(ds_gfas.variables)}")

    frp = ds_gfas[frp_var]
    frp2d = frp.isel(time=time_index) if ("time" in frp.dims and frp.sizes.get("time", 1) > 1) else frp.squeeze()

    # Identify altitude variable (optional but required for your comparison)
    alt_name = _pick_first_existing_var(ds_gfas, altitude_candidates)
    alt2d = None
    if alt_name is not None:
        alt = ds_gfas[alt_name]
        alt2d = alt.isel(time=time_index) if ("time" in alt.dims and alt.sizes.get("time", 1) > 1) else alt.squeeze()

    lon = ds_gfas["longitude"].values
    lat = ds_gfas["latitude"].values
    lon2d, lat2d = np.meshgrid(lon, lat)

    frp_vals = np.asarray(frp2d.values, dtype=float)
    mask = np.isfinite(frp_vals) & (frp_vals >= float(min_frp))

    if use_canada_only and canada_bbox is not None:
        bbox = canada_bbox
        mask &= (
            (lat2d >= bbox["lat_min"]) & (lat2d <= bbox["lat_max"]) &
            (lon2d >= bbox["lon_min"]) & (lon2d <= bbox["lon_max"])
        )

    if not np.any(mask):
        return pd.DataFrame(columns=["lon", "lat", "frp", "gfas_alt_km", "i_lat", "i_lon", "gfas_alt_var"])

    ij = np.argwhere(mask)
    i_lat = ij[:, 0]
    i_lon = ij[:, 1]

    out = pd.DataFrame({
        "lon": lon2d[mask].astype(float),
        "lat": lat2d[mask].astype(float),
        "frp": frp_vals[mask].astype(float),
        "i_lat": i_lat.astype(int),
        "i_lon": i_lon.astype(int),
        "gfas_alt_var": alt_name if alt_name is not None else None,
    })

    if alt2d is not None:
        a = np.asarray(alt2d.values, dtype=float)
        out["gfas_alt_km"] = a[mask].astype(float)

        # Heuristic: if GFAS height is in meters, convert to km.
        # (Adjust/remove if your files are already km.)
        if np.nanmedian(out["gfas_alt_km"].values) > 100.0:
            out["gfas_alt_km"] = out["gfas_alt_km"] / 1000.0
    else:
        out["gfas_alt_km"] = np.nan

    # Wrap longitudes to [-180, 180)
    out["lon"] = wrap_longitudes(out["lon"].values)

    return out


def _track_cumulative_km(lon: np.ndarray, lat: np.ndarray, geod: Geod) -> np.ndarray:
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    if len(lon) < 2:
        return np.zeros(len(lon), dtype=float)
    _, _, dist_m = geod.inv(lon[:-1], lat[:-1], lon[1:], lat[1:])
    seg_km = np.r_[0.0, dist_m / 1000.0]
    return np.cumsum(seg_km)


def match_gfas_fires_to_earthcare_plumes(
    fires: pd.DataFrame,
    *,
    track_lon: np.ndarray,
    track_lat: np.ndarray,
    plume_series: np.ndarray,
    sph_top: Optional[np.ndarray] = None,
    sph_bottom: Optional[np.ndarray] = None,
    sph_ext: Optional[np.ndarray] = None,
    max_track_distance_km: float = 100.0,
    plume_search_max_km: float = 300.0,
) -> pd.DataFrame:
    """
    For each GFAS fire point:
      - find closest track index (geodesic)
      - if no plume there, search for closest plume-containing track point in a ±window along-track
    Returns fires + match columns + EarthCARE SPH at matched index.
    """
    if fires.empty:
        return fires.copy()

    geod = Geod(ellps="WGS84")
    tlon = np.asarray(track_lon, dtype=float)
    tlat = np.asarray(track_lat, dtype=float)

    # Precompute along-track distance coordinate for windowing
    s_km = _track_cumulative_km(tlon, tlat, geod)

    # Plume mask
    plume_series = np.asarray(plume_series, dtype=float)
    has_plume = np.isfinite(plume_series)

    out_rows: List[dict] = []

    for _, r in fires.iterrows():
        flon = float(r["lon"])
        flat = float(r["lat"])

        # Dist to all track points (vectorized)
        _, _, dist_m = geod.inv(
            tlon, tlat,
            np.full_like(tlon, flon, dtype=float),
            np.full_like(tlat, flat, dtype=float),
        )
        dist_km = dist_m / 1000.0

        i0 = int(np.nanargmin(dist_km))
        d0 = float(dist_km[i0])

        # If not within the track-distance threshold, skip (or mark as unmatched)
        if not np.isfinite(d0) or d0 > float(max_track_distance_km):
            row = dict(r)
            row.update({
                "matched": False,
                "track_i_nearest": i0,
                "dist_km_nearest": d0,
                "track_i_used": np.nan,
                "dist_km_used": np.nan,
                "used_plume_fallback": False,
                "ec_sph_top_km": np.nan,
                "ec_sph_bottom_km": np.nan,
                "ec_sph_ext_km": np.nan,
            })
            out_rows.append(row)
            continue

        # Use nearest if plume exists there; otherwise find closest plume point in window
        i_used = i0
        used_fallback = False

        if not has_plume[i0]:
            used_fallback = True

            # Along-track window around i0
            s0 = s_km[i0]
            in_win = (s_km >= (s0 - plume_search_max_km)) & (s_km <= (s0 + plume_search_max_km)) & has_plume

            if np.any(in_win):
                # among plume points in window, pick the one closest in geodesic distance to the fire
                cand_idx = np.where(in_win)[0]
                j = int(cand_idx[np.argmin(dist_km[cand_idx])])
                i_used = j
            else:
                # No plume anywhere near: keep nearest (but will produce NaNs for SPH)
                i_used = i0

        row = dict(r)
        row.update({
            "matched": True,
            "track_i_nearest": i0,
            "dist_km_nearest": d0,
            "track_i_used": int(i_used),
            "dist_km_used": float(dist_km[i_used]),
            "used_plume_fallback": used_fallback and bool(has_plume[i_used]),
            "ec_sph_top_km": float(sph_top[i_used]) if sph_top is not None and np.isfinite(sph_top[i_used]) else np.nan,
            "ec_sph_bottom_km": float(sph_bottom[i_used]) if sph_bottom is not None and np.isfinite(sph_bottom[i_used]) else np.nan,
            "ec_sph_ext_km": float(sph_ext[i_used]) if sph_ext is not None and np.isfinite(sph_ext[i_used]) else np.nan,
        })
        out_rows.append(row)

    return pd.DataFrame(out_rows)



def gfas_fire_points_with_injection_heights(
    ds_gfas: xr.Dataset,
    *,
    frp_var: str = "frpfire",
    time_index: int = 0,
    min_frp: float = 0.1,
    use_canada_only: bool = True,
    canada_bbox: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    GFAS fire pixels with lon/lat/FRP and injection heights:
      mami, apt, apb, injh (as stored in file; typically meters)
    """
    if frp_var not in ds_gfas:
        raise KeyError(f"GFAS FRP variable '{frp_var}' not in dataset variables.")

    frp = ds_gfas[frp_var]
    frp2d = frp.isel(time=time_index) if ("time" in frp.dims and frp.sizes.get("time", 1) > 1) else frp.squeeze()

    def _get2d(name: str) -> Optional[np.ndarray]:
        if name not in ds_gfas:
            return None
        v = ds_gfas[name]
        v2d = v.isel(time=time_index) if ("time" in v.dims and v.sizes.get("time", 1) > 1) else v.squeeze()
        return np.asarray(v2d.values, dtype=float)

    mami = _get2d("mami")
    apt  = _get2d("apt")
    apb  = _get2d("apb")
    injh = _get2d("injh")

    lon = wrap_longitudes(ds_gfas["longitude"].values)
    lat = ds_gfas["latitude"].values
    lon2d, lat2d = np.meshgrid(lon, lat)

    frp_vals = np.asarray(frp2d.values, dtype=float)
    mask = np.isfinite(frp_vals) & (frp_vals >= float(min_frp))

    if use_canada_only and canada_bbox is not None:
        bbox = canada_bbox
        mask &= (
            (lat2d >= bbox["lat_min"]) & (lat2d <= bbox["lat_max"]) &
            (lon2d >= bbox["lon_min"]) & (lon2d <= bbox["lon_max"])
        )

    if not np.any(mask):
        return pd.DataFrame(columns=["lon","lat","frp","mami","apt","apb","injh","i_lat","i_lon"])

    ij = np.argwhere(mask)
    i_lat = ij[:, 0].astype(int)
    i_lon = ij[:, 1].astype(int)

    out = pd.DataFrame({
        "lon": lon2d[mask].astype(float),
        "lat": lat2d[mask].astype(float),
        "frp": frp_vals[mask].astype(float),
        "i_lat": i_lat,
        "i_lon": i_lon,
        "mami": mami[mask].astype(float) if mami is not None else np.nan,
        "apt":  apt[mask].astype(float)  if apt  is not None else np.nan,
        "apb":  apb[mask].astype(float)  if apb  is not None else np.nan,
        "injh": injh[mask].astype(float) if injh is not None else np.nan,
    })
    return out








# -----------------------------
# Core data preparation
# -----------------------------
def prepare_data_flexpart_earthcare_gfas(
    flexpart_nc: str,
    earthcare_file: str,
    *,
    ebd_file: Optional[str] = None,
    ald_file: Optional[str] = None,
    gfas_folder: Optional[str] = None,
    gfas_variant: str = "01",
    config: PlotConfig = PlotConfig(),
    geop_nc = "/xnilu_wrk/users/sec/kleinprojekte/kerstin_wildfire/geop_surf_05.nc"
) -> Dict[str, Any]:
    """
    Load and compute all inputs required for plotting.
    Returns a dictionary so plotting can be “pure rendering”.
    """

    # ---------- Load FLEXPART ----------
    ds_fp = xr.open_dataset(flexpart_nc, decode_timedelta=True).squeeze(drop=True)
    ds_fp = ds_fp.assign_coords(longitude=wrap_longitudes(ds_fp["longitude"])).sortby("longitude").sortby("latitude")

    z_fp_agl_m = ds_fp["height"] - 250  
    bc_mr_all = ds_fp["spec001_mr"]

    # ---------- Load EarthCARE ATL_TC ----------
    with h5py.File(earthcare_file, "r") as f:
        sci = f["ScienceData"]

        lat_track = np.array(sci["latitude"])
        lon_track = np.array(sci["longitude"])
        lon_track_wrapped = wrap_longitudes(lon_track)

        dt_utc = pd.to_datetime(num2date(sci["time"], "seconds since 2000-01-01").astype("datetime64[us]"))
        dt_lt = dt_utc + pd.to_timedelta(np.round(lon_track / 15.0 * 3600.0).astype(int), unit="s")

        tropopause_km = 0.001 * np.array(sci["tropopause_height"])

        geoid_offset = np.array(sci["geoid_offset"])
        # (time, z) km, with original reverse in z
        height_km_tc = 0.001 * (np.array(sci["height"])[::-1] - geoid_offset[:, np.newaxis])

        dt_2d_tc = np.tile(dt_utc.to_numpy()[:, None], (1, height_km_tc.shape[1]))

        class_tc_raw = np.array(sci["classification_low_resolution"])
        class_tc = class_tc_raw  # default: full satellite classes

        # Only reduce when NOT in EarthCARE-only mode
        if config.reduce_classifications and (not config.plot_only_TC_and_extinction_alone) and (not config.only_smoke_for_tc_sph):
            mask_missing = np.isin(class_tc, [-3, -1])
            mask_surface = (class_tc == -2)
            mask_clear = (class_tc == 0)
            mask_cloud = np.isin(class_tc, [1, 2, 3])
            mask_aerosol = np.isin(
                class_tc,
                [10, 11, 12, 13, 14, 15, 20, 21, 22, 25, 26, 27, 101, 102, 104, 105, 106, 107],
            )

            class_tc = class_tc.copy()
            class_tc[mask_missing] = -1  # Missing
            class_tc[mask_surface] = -2  # Surface
            class_tc[mask_clear] = 0     # Clear
            class_tc[mask_cloud] = 1     # Clouds
            class_tc[mask_aerosol] = 4   # Aerosols
        
        if config.only_smoke_for_tc_sph:
            mask_smoke = np.isin(class_tc, config.tc_smoke_class_ids)
            class_tc = class_tc.copy()
            class_tc[~mask_smoke] = 0  # Set non-smoke to clear (0), keep smoke classes as-is


        # --- ATL_TC extinction (optional), cloud-masked ---
        tc_ext = _pick_first_existing(
            sci,
            [
                "particle_extinction_coefficient_355nm",
                "particle_extinction_coefficient_355nm_low_resolution",
                "extinction_coefficient_355nm",
                "extinction_low_resolution",
                "extinction_coefficient",
            ],
        )

        tc_ext_raw = None
        if tc_ext is not None:
            tc_ext = ensure_2d_time_height(tc_ext, n_time=len(dt_utc), n_z=height_km_tc.shape[1])
            tc_ext_raw = np.array(tc_ext, copy=True)

            # --- Smoke-only mask for TC extinction (only affects TC-SPH; does NOT change plotted TC classes unless you want it to) ---
            if config.only_smoke_for_tc_sph:
                cls_only_smoke = ensure_2d_time_height(class_tc, n_time=len(dt_utc), n_z=height_km_tc.shape[1])
                smoke_mask = np.isin(cls_only_smoke, config.tc_smoke_class_ids)  # (time, z)
                tc_ext = np.where(smoke_mask, tc_ext, np.nan)

            # Only cloud-mask TC extinction in the comparison mode
            if not config.plot_only_TC_and_extinction_alone:
                tc_ext = mask_clouds_on_tc_grid(
                    tc_ext, class_tc,
                    cloud_classes=(1, 2, 3) if not config.reduce_classifications else (1,)
        )


    # ---------- Surface height from ECMWF geopotential ----------
    ds_geop = xr.open_dataset(geop_nc).squeeze(drop=True)
    ds_geop = ds_geop.assign_coords(longitude=wrap_longitudes(ds_geop["longitude"])).sortby("longitude")

    inv_g = 1.0 / 9.80665
    h_surf_m = (
        ds_geop["z"] * inv_g
    ).interp(
        latitude=xr.DataArray(lat_track, dims=["track"]),
        longitude=xr.DataArray(lon_track_wrapped, dims=["track"]),
        method="nearest",
    ).to_numpy()
    elev_km = 0.001 * h_surf_m.astype(float)

    # Smooth orography along-track to match FLEXPART-ish horizontal resolution (original logic)
    dlat = float(np.median(np.diff(ds_fp["latitude"].values)))
    dlon = float(np.median(np.diff(ds_fp["longitude"].values)))
    mean_lat = float(np.nanmean(lat_track))
    fp_dx_km = 111.0 * np.hypot(dlat, dlon * np.cos(np.deg2rad(mean_lat)))

    geod = Geod(ellps="WGS84")
    seg_km = np.r_[0.0, geod.inv(lon_track[:-1], lat_track[:-1], lon_track[1:], lat_track[1:])[2] / 1000.0]
    km_per_sample = max(1e-6, float(np.median(seg_km[1:])))

    window_km = max(2.0 * fp_dx_km, 80.0)
    win = max(3, int(round(window_km / km_per_sample)))
    box = np.ones(win, dtype=float) / win
    elev_km_coarse = np.convolve(elev_km, box, mode="same")

    # FLEXPART heights (km)
    z_fp_agl_km = (z_fp_agl_m.values.astype(float)) / 1000.0
    z_fp_amsl_km_track = elev_km_coarse[:, None] + z_fp_agl_km[None, :]

    # ---------- FLEXPART time selection + column for map ----------
    overpass_time = pd.to_datetime(dt_utc.to_numpy()[len(dt_utc) // 2])
    bc_3d = bc_mr_all.sel(time=overpass_time, method="nearest") if "time" in bc_mr_all.dims else bc_mr_all

    if "height" in bc_3d.dims:
        col_bc = (bc_3d * z_fp_agl_m).sum("height") / 1_000.0
        col_bc.attrs["units"] = "μg m⁻²"
    else:
        col_bc = bc_3d

    # EarthCARE metadata for naming
    ec_base = os.path.basename(earthcare_file)
    baseline = ec_base[6:8]
    frame = ec_base[-4]
    ts = pd.to_datetime(overpass_time)
    date_str = f"{ts.day}-{ts.strftime('%b').lower()}-{ts.year}"
    date_ymd = ts.strftime("%Y%m%d")

    # FLEXPART on EarthCARE track (nearest, as you currently use)
    lat_da = xr.DataArray(lat_track, dims=["track"])
    lon_da = xr.DataArray(lon_track_wrapped, dims=["track"])
    bc_on_track = bc_3d.interp(latitude=lat_da, longitude=lon_da, method="nearest")


    # -------------------------------------------------
    # Reduce ONLY FLEXPART sampling to FLEXPART horizontal resolution (km),
    # keep EarthCARE/TC arrays full resolution.
    # Uses along-track distance bins so sampling is uniform in km
    # -------------------------------------------------
    keep_idx = None
    if config.set_flexpart_res_to_original:
        # 1) Along-track cumulative distance (km)
        # geod already exists above in your code: geod = Geod(ellps="WGS84")
        seg_km = np.r_[0.0, geod.inv(
            lon_track_wrapped[:-1], lat_track[:-1],
            lon_track_wrapped[1:],  lat_track[1:]
        )[2] / 1000.0]
        s_km = np.cumsum(seg_km)

        # 2) Bin by distance using a step comparable to FLEXPART grid size
        # fp_dx_km was computed above (based on FLEXPART lat/lon resolution and mean lat)
        step_km = float(fp_dx_km)
        step_km = max(step_km, 1e-3)  # guard

        dist_bin = np.floor(s_km / step_km).astype(np.int64)

        # keep first sample in each distance bin
        _, keep_idx = np.unique(dist_bin, return_index=True)
        keep_idx = np.sort(keep_idx)

        # (optional) ensure last point is included
        if keep_idx.size == 0:
            keep_idx = np.array([0], dtype=int)
        elif keep_idx[-1] != (len(s_km) - 1):
            keep_idx = np.r_[keep_idx, len(s_km) - 1]

        # FLEXPART-only reduced track arrays
        dt_fp = dt_utc[keep_idx]
        lt_fp = dt_lt[keep_idx]
        lat_fp = lat_track[keep_idx]
        lon_fp = lon_track[keep_idx]
        lon_fp_wrapped = lon_track_wrapped[keep_idx]

        tropo_fp = tropopause_km[keep_idx]
        elev_fp = elev_km[keep_idx]
        elev_fp_coarse = elev_km_coarse[keep_idx]

        bc_on_track_fp = bc_on_track.isel(track=keep_idx)
        z_fp_amsl_km_track_fp = z_fp_amsl_km_track[keep_idx, :]

    else:
        # no reduction: FLEXPART uses full EarthCARE track
        dt_fp = dt_utc
        lt_fp = dt_lt
        lat_fp = lat_track
        lon_fp = lon_track
        lon_fp_wrapped = lon_track_wrapped
        tropo_fp = tropopause_km
        elev_fp = elev_km
        elev_fp_coarse = elev_km_coarse
        bc_on_track_fp = bc_on_track
        z_fp_amsl_km_track_fp = z_fp_amsl_km_track





    if ("height" in bc_on_track_fp.dims) and ("track" in bc_on_track_fp.dims):
        bc_on_track_fp = bc_on_track_fp.transpose("height", "track")

    fp_on_track = np.asarray(bc_on_track_fp)  # (z_fp, track_fp)
    n_track_fp = len(lat_fp)



    # ---------- FLEXPART cross-section resampling for plotting ----------
    if config.set_flexpart_res_to_original:
        fp_plot = np.array(np.ma.filled(fp_on_track.T, np.nan), dtype=float)  # (track, z_fp)
        fp_height_plot_km = _fill_nonfinite_rows_2d(np.asarray(z_fp_amsl_km_track_fp, dtype=float))
        fp_time_plot = np.tile(dt_fp.to_numpy()[:, None], (1, fp_plot.shape[1]))

    else:
        # Interpolate FLEXPART profiles onto TC height grid (track, z_tc)
        fp_plot = np.full_like(height_km_tc, np.nan, dtype=float)  # (track, z_tc)
        for i in range(n_track_fp):
            h_tc = height_km_tc[i, :]                 # EarthCARE z grid (km)
            z_fp_amsl = z_fp_amsl_km_track[i, :]      # FLEXPART z grid (km AMSL)
            order = np.argsort(z_fp_amsl)
            prof_fp = fp_on_track[:, i]               # (z_fp,)
            fz = interp1d(z_fp_amsl[order], prof_fp[order], bounds_error=False, fill_value=np.nan)
            fp_plot[i, :] = fz(h_tc)

        fp_height_plot_km = height_km_tc
        fp_time_plot = dt_2d_tc

    fp_plot_raw = fp_plot.copy()
    fp_plot_masked = np.ma.masked_where((fp_plot < SPH_THRESHOLD) | ~np.isfinite(fp_plot), fp_plot)
    has_plotted_fp = np.any(np.isfinite(np.ma.filled(fp_plot_masked, np.nan)), axis=1)

    # ---------- Read ATL_EBD extinction ----------
    ebd_ext = ebd_height_km = ebd_dt = None
    ebd_ext_raw = None
    if ebd_file is not None:
        with h5py.File(ebd_file, "r") as f_ebd:
            ds_ebd = f_ebd["ScienceData"]
            ebd_time_1d = np.array(ds_ebd["time"])
            ebd_dt = pd.to_datetime(num2date(ebd_time_1d, "seconds since 2000-01-01").astype("datetime64[us]"))

            ext = _choose_ebd_extinction(ds_ebd, config.resolution)
            if ext is None and "particle_extinction_coefficient_355nm" in ds_ebd:
                ext = np.array(ds_ebd["particle_extinction_coefficient_355nm"])

            if ("height" in ds_ebd) and ("geoid_offset" in ds_ebd):
                geoid_offset_ebd = np.array(ds_ebd["geoid_offset"])
                height_2d = 0.001 * (np.array(ds_ebd["height"])[::-1] - geoid_offset_ebd[:, np.newaxis])
                ebd_height_km = np.mean(height_2d, axis=0)
            else:
                ebd_height_km = np.array(ds_ebd.get("JSG_height", [])) * 0.001

            if ext is not None and ext.ndim == 2 and ebd_height_km is not None and ebd_dt is not None:
                ext = ensure_2d_time_height(ext, n_time=len(ebd_dt), n_z=len(ebd_height_km))

            # RAW copy (only masked invalid)
            ebd_ext_raw = np.ma.masked_invalid(np.array(ext, dtype=float)) if ext is not None else None

            # In EarthCARE-only mode: keep EBD extinction RAW (no QF, no cloud masking, no pixel removal)
            if config.plot_only_TC_and_extinction_alone:
                ebd_ext = ebd_ext_raw
            else:
                # Quality flag filtering
                q_keep = set(config.quality_flags)
                qf = np.array(ds_ebd.get("quality_status", []))
                if qf.size and ext is not None:
                    # try align qf to ext shape
                    if qf.shape == ext.T.shape:
                        qf = qf.T
                    elif qf.shape == (ext.shape[0],):
                        qf = qf[:, None]
                    elif qf.shape == (ext.shape[1],):
                        qf = qf[None, :]

                    if not np.issubdtype(qf.dtype, np.integer):
                        qf = qf.astype(np.int32, copy=False)

                    if qf.shape == ext.shape:
                        keep = np.isin(qf, list(q_keep))
                        ext = np.where(keep, ext, np.nan)

                ebd_ext = np.ma.masked_invalid(ext) if ext is not None else None

    # ---------- Read ATL_ALD layer heights (optional) ----------
    ald_time = None
    ald_top_height_km = None
    ald_bottom_height_km = None
    ald_layer_number = None

    if ald_file is not None:
        try:
            with h5py.File(ald_file, "r") as f_ald:
                ds_ald = f_ald["ScienceData"]

                ald_time = pd.to_datetime(
                    num2date(ds_ald["time"][:], "seconds since 2000-01-01").astype("datetime64[us]")
                )

                ald_layer_number = np.array(ds_ald["aerosol_layer_number"], dtype=int)
                ald_top_height_km = 0.001 * np.array(ds_ald["aerosol_layer_top"], dtype=float)
                ald_bottom_height_km = 0.001 * np.array(ds_ald["aerosol_layer_base"], dtype=float)

                # Replace fill values
                ald_top_height_km[ald_top_height_km > 1e5] = np.nan
                ald_bottom_height_km[ald_bottom_height_km > 1e5] = np.nan

        except Exception as e:
            print(f"Warning: failed to read ATL_ALD from {ald_file}: {e}")
            ald_time = ald_top_height_km = ald_bottom_height_km = ald_layer_number = None

    # ---------- Build TC cloud mask + map to EBD grid (for masking EBD) ----------
    ebd_bad_mask = None
    if config.reduce_classifications:
        # in reduced scheme clouds=1 and missing=-1.
        tc_cloud_mask = np.isin(class_tc, (1, -1))
    else:
        tc_cloud_mask = np.isin(class_tc, (1, 2, 3))


    if not config.plot_only_TC_and_extinction_alone:
        class_data_nocloud = np.where(tc_cloud_mask, np.nan, class_tc)

        ebd_cloud_mask_on_ebdgrid = None
        if (ebd_dt is not None) and (ebd_height_km is not None) and (ebd_ext is not None):
            tc_times = dt_utc.to_numpy().astype("datetime64[ns]").astype("int64")
            ebd_times = ebd_dt.to_numpy().astype("datetime64[ns]").astype("int64")

            idx = np.searchsorted(tc_times, ebd_times, side="left")
            idx0 = np.clip(idx - 1, 0, len(tc_times) - 1)
            idx1 = np.clip(idx, 0, len(tc_times) - 1)
            choose0 = np.abs(tc_times[idx0] - ebd_times) < np.abs(tc_times[idx1] - ebd_times)
            nearest_tc_idx = np.where(choose0, idx0, idx1)  # (n_ebd_time,)

            z_tc = np.asarray(height_km_tc[0, :], dtype=float)
            z_ebd = np.asarray(ebd_height_km, dtype=float)

            if np.any(np.diff(z_tc) <= 0):
                order = np.argsort(z_tc)
                z_tc_sorted = z_tc[order]
                tc_cloud_mask_sorted = tc_cloud_mask[:, order]
            else:
                z_tc_sorted = z_tc
                tc_cloud_mask_sorted = tc_cloud_mask

            idx_float = np.interp(z_ebd, z_tc_sorted, np.arange(z_tc_sorted.size), left=0, right=z_tc_sorted.size - 1)
            iz = np.clip(np.rint(idx_float).astype(int), 0, z_tc_sorted.size - 1)

            ebd_cloud_mask_on_ebdgrid = tc_cloud_mask_sorted[nearest_tc_idx, :][:, iz]

            # --- Smoke-only mask for EBD extinction: map TC smoke classes onto the EBD grid ---
            if config.only_smoke_for_tc_sph and (ebd_ext is not None):
                tc_smoke_mask = np.isin(class_tc, config.tc_smoke_class_ids)  # (tc_time, tc_z)

                # keep same height ordering logic as the cloud mask mapping
                if np.any(np.diff(z_tc) <= 0):
                    tc_smoke_mask_sorted = tc_smoke_mask[:, order]
                else:
                    tc_smoke_mask_sorted = tc_smoke_mask

                ebd_smoke_mask_on_ebdgrid = tc_smoke_mask_sorted[nearest_tc_idx, :][:, iz]  # (ebd_time, ebd_z)

                # apply ONLY where shapes match
                if ebd_smoke_mask_on_ebdgrid.shape == np.array(ebd_ext).shape:
                    ebd_ext = np.ma.array(ebd_ext, copy=False)
                    # keep smoke, mask everything else
                    ebd_ext = np.ma.masked_where(~ebd_smoke_mask_on_ebdgrid, ebd_ext)


            

            # Optional: expand mask to neighbors
            ebd_cloud_mask_on_ebdgrid = dilate_mask_2d(
                ebd_cloud_mask_on_ebdgrid,
                dt_pad=int(config.cloud_pad_time),
                dz_pad=int(config.cloud_pad_z),
            )

            # Optional: filter huge values (before masking)
            if config.filter_out_ebd_values_above is not None:
                ebd_ext = np.ma.masked_where(ebd_ext > float(config.filter_out_ebd_values_above), ebd_ext)

            # Apply cloud mask ONCE
            if ebd_cloud_mask_on_ebdgrid.shape == np.array(ebd_ext).shape:
                ebd_ext = np.ma.masked_where(ebd_cloud_mask_on_ebdgrid, ebd_ext)

            # -----------------------------
            # Remove small pixel groups in EBD extinction (time x height)
            # -----------------------------
            if (ebd_ext is not None) and (config.remove_pixel_groups_below is not None):
                e = np.array(ebd_ext.filled(np.nan), dtype=float)

                # Ensure (time, z)
                e = ensure_2d_time_height(e, n_time=len(ebd_dt), n_z=len(ebd_height_km))

                # Apply the same “plot mask” rules (so groups match what you’d actually see)
                e = _mask_like_plot(e, "ebd_ext")

                # Remove connected components smaller than N pixels
                e = remove_small_pixel_groups_2d(
                    e,
                    min_size=config.remove_pixel_groups_below,
                    connectivity=config.remove_pixel_groups_connectivity,
                    foreground_mask=np.isfinite(e),
                    fill_value=np.nan,
                )

                # Put back into masked array used downstream (plot + SPH)
                ebd_ext = np.ma.masked_invalid(e)

            # ============================================================
            # Build an EBD "bad pixels" mask (cloud OR missing/invalid extinction)
            # This is what we want to remove from FLEXPART.
            # ============================================================
            ebd_bad_mask = None
            if (ebd_ext is not None) and (ebd_dt is not None) and (ebd_height_km is not None):
                # turn EBD extinction into float array with NaNs where it will not plot
                ebd_ext_arr = np.array(ebd_ext.filled(np.nan), dtype=float)
                ebd_ext_arr = ensure_2d_time_height(ebd_ext_arr, n_time=len(ebd_dt), n_z=len(ebd_height_km))

                # apply the same plot masking rules (<=0, >9.9, nonfinite etc.)
                ebd_ext_arr = _mask_like_plot(ebd_ext_arr, "ebd_ext")

                # missing/invalid after all rules
                ebd_missing_mask = ~np.isfinite(ebd_ext_arr)

                # cloud mask on EBD grid (already computed above)
                ebd_cloud = (ebd_cloud_mask_on_ebdgrid is not None) & np.asarray(ebd_cloud_mask_on_ebdgrid, dtype=bool)

                # combine
                ebd_bad_mask = ebd_cloud | ebd_missing_mask


    else:
        # Keep classification arrays untouched; don't create a no-cloud version
        tc_cloud_mask = None
        class_data_nocloud = np.array(class_tc, copy=True)
        ebd_cloud_mask_on_ebdgrid = None

        sph_ext_km = sph_top_km = sph_bottom_km = None
        sph_ext_layers_ebd = sph_top_layers_ebd = sph_bottom_layers_ebd = None
        ebd_n_layers = None
        ebd_is_multilayer = None


    # ---------- Optional: remove FLEXPART pixels where EarthCARE is cloud OR missing ----------
    fp_bad_mask_on_tc = None
    if config.no_clouds_in_flexpart and (ebd_bad_mask is not None) and (ebd_dt is not None) and (ebd_height_km is not None):
        # Map EBD bad mask to TC time (nearest)
        tc_times  = dt_utc.to_numpy().astype("datetime64[ns]").astype("int64")
        ebd_times = ebd_dt.to_numpy().astype("datetime64[ns]").astype("int64")

        idx  = np.searchsorted(ebd_times, tc_times, side="left")
        idx0 = np.clip(idx - 1, 0, len(ebd_times) - 1)
        idx1 = np.clip(idx,     0, len(ebd_times) - 1)
        choose0 = np.abs(ebd_times[idx0] - tc_times) < np.abs(ebd_times[idx1] - tc_times)
        nearest_ebd_idx = np.where(choose0, idx0, idx1)

        # Map EBD z-grid → TC z-grid
        z_tc  = np.asarray(height_km_tc[0, :], dtype=float)
        z_ebd = np.asarray(ebd_height_km, dtype=float)

        if np.any(np.diff(z_ebd) <= 0):
            order = np.argsort(z_ebd)
            z_ebd_sorted = z_ebd[order]
            bad_sorted = ebd_bad_mask[:, order]
        else:
            z_ebd_sorted = z_ebd
            bad_sorted = ebd_bad_mask

        idx_float = np.interp(z_tc, z_ebd_sorted, np.arange(z_ebd_sorted.size), left=0, right=z_ebd_sorted.size - 1)
        iz = np.clip(np.rint(idx_float).astype(int), 0, z_ebd_sorted.size - 1)

        # Final bad mask on TC grid (tc_time, tc_z)
        fp_bad_mask_on_tc = bad_sorted[nearest_ebd_idx, :][:, iz]

    # ---- Apply mask to FLEXPART arrays used later ----
    if fp_bad_mask_on_tc is not None:
        if config.set_flexpart_res_to_original:
            # FLEXPART arrays are (track_fp, z_fp) but fp_bad_mask_on_tc is (tc_time, z_tc)

            # 1) time: reduce TC mask to FLEXPART track points
            if keep_idx is None:
                keep_idx = np.arange(fp_bad_mask_on_tc.shape[0])
            bad_tc_t = fp_bad_mask_on_tc[keep_idx, :]  # (track_fp, z_tc)

            # 2) height: map TC z -> FLEXPART z (native FP vertical)
            z_tc = np.asarray(height_km_tc[0, :], dtype=float)          # (z_tc,)
            z_fp_plot = np.nanmean(fp_height_plot_km, axis=0)           # (z_fp,)

            if np.any(np.diff(z_tc) <= 0):
                order = np.argsort(z_tc)
                z_tc_sorted = z_tc[order]
                bad_tc_t = bad_tc_t[:, order]
            else:
                z_tc_sorted = z_tc

            iz = np.interp(z_fp_plot, z_tc_sorted, np.arange(z_tc_sorted.size),
                        left=0, right=z_tc_sorted.size - 1)
            iz = np.clip(np.rint(iz).astype(int), 0, z_tc_sorted.size - 1)

            bad_on_fp = bad_tc_t[:, iz]  # (track_fp, z_fp)

            fp_plot_raw = np.where(bad_on_fp, np.nan, fp_plot_raw)
            fp_plot_masked = np.ma.masked_where(bad_on_fp, fp_plot_masked)

        else:
            # FLEXPART already on TC grid (track, z_tc)
            fp_plot_raw = np.where(fp_bad_mask_on_tc, np.nan, fp_plot_raw)
            fp_plot_masked = np.ma.masked_where(fp_bad_mask_on_tc, fp_plot_masked)



    # ============================================================
    # SPH METRICS
    # ============================================================

    # ---------- SPH from ATL_EBD extinction (multilayer) ----------
    sph_ext_time = ebd_dt
    sph_top_time = ebd_dt
    sph_bottom_time = ebd_dt

    sph_ext_km = sph_top_km = sph_bottom_km = None
    sph_ext_layers_ebd = sph_top_layers_ebd = sph_bottom_layers_ebd = None
    ebd_n_layers = None
    ebd_is_multilayer = None

    if (ebd_ext is not None) and (ebd_dt is not None) and (ebd_height_km is not None):
        ext_arr = np.array(ebd_ext.filled(np.nan))
        ext_arr = ensure_2d_time_height(ext_arr, n_time=len(ebd_dt), n_z=len(ebd_height_km))
        ext_arr = _mask_like_plot(ext_arr, "ebd_ext")

        ext_arr = _mask_like_plot(ext_arr, "ebd_ext")

        if config.remove_pixel_groups_below is not None:
            # foreground = "pixels that exist in the plot" (finite after masking)
            fg = np.isfinite(ext_arr)

            ext_arr = remove_small_pixel_groups_2d(
                ext_arr,
                min_size=config.remove_pixel_groups_below,
                connectivity=config.remove_pixel_groups_connectivity,
                foreground_mask=fg,
                fill_value=np.nan,
            )


        ebd_has_plotted = np.any(np.isfinite(ext_arr), axis=1)

        z = np.asarray(ebd_height_km, dtype=float)
        # sanitize z
        if np.any(~np.isfinite(z)):
            good = np.isfinite(z)
            z = z[good]
            ext_arr = ext_arr[:, good]
        if np.any(np.diff(z) <= 0):
            order = np.argsort(z)
            z = z[order]
            ext_arr = ext_arr[:, order]

        n_t = ext_arr.shape[0]
        max_layers_keep = int(config.max_layers)
        min_bins_keep = int(config.sph_min_bins)

        sph_ext_layers_ebd = np.full((n_t, max_layers_keep), np.nan, dtype=float)
        sph_top_layers_ebd = np.full((n_t, max_layers_keep), np.nan, dtype=float)
        sph_bottom_layers_ebd = np.full((n_t, max_layers_keep), np.nan, dtype=float)

        sph_ext_km = np.full(n_t, np.nan, dtype=float)
        sph_top_km = np.full(n_t, np.nan, dtype=float)
        sph_bottom_km = np.full(n_t, np.nan, dtype=float)

        ebd_n_layers = np.full(n_t, np.nan, dtype=float)
        ebd_is_multilayer = np.zeros(n_t, dtype=bool)

        for i in range(n_t):
            prof = ext_arr[i, :]
            if np.count_nonzero(np.isfinite(prof)) < min_bins_keep:
                continue
            if not np.any(np.isfinite(prof)):
                continue

            try:
                layers = sph_layers(
                    z=z,
                    beta=prof,
                    threshold=float(config.sph_threshold_ext),
                    min_bins=min_bins_keep,
                    max_layers=max_layers_keep,
                )
            except Exception:
                layers = dict(n_layers=0, bottom=np.array([]), top=np.array([]), ext=np.array([]))

            nL = int(layers.get("n_layers", 0))
            if nL > 0:
                ebd_n_layers[i] = float(nL)
                ebd_is_multilayer[i] = (nL >= 2)

                sph_bottom_layers_ebd[i, :nL] = np.asarray(layers["bottom"], dtype=float)
                sph_top_layers_ebd[i, :nL] = np.asarray(layers["top"], dtype=float)
                sph_ext_layers_ebd[i, :nL] = np.asarray(layers["ext"], dtype=float)

            # strongest layer series
            try:
                sph_ext_km[i] = sph_ext(beta=prof, z=z, threshold=float(config.sph_threshold_ext))
            except Exception:
                pass
            try:
                st = sph_top(z=z, beta=prof, threshold=float(config.sph_threshold_ext))
                sph_top_km[i] = st if st is not None else np.nan
            except Exception:
                pass
            try:
                sb = sph_bottom(z=z, beta=prof, threshold=float(config.sph_threshold_ext))
                sph_bottom_km[i] = sb if sb is not None else np.nan
            except Exception:
                print(f"Warning: failed to compute sph_bottom at time index {i}")
                pass

        # Mask where EBD plot has nothing
        no_plot = ~ebd_has_plotted
        sph_ext_km[no_plot] = np.nan
        sph_top_km[no_plot] = np.nan
        sph_bottom_km[no_plot] = np.nan
        sph_ext_layers_ebd[no_plot, :] = np.nan
        sph_top_layers_ebd[no_plot, :] = np.nan
        sph_bottom_layers_ebd[no_plot, :] = np.nan
        ebd_n_layers[no_plot] = np.nan
        ebd_is_multilayer[no_plot] = False

        # Optional smoothing along-track
        if config.sph_smooth_sigma is not None and float(config.sph_smooth_sigma) > 0:
            sigma = float(config.sph_smooth_sigma)
            sph_ext_km = gaussian_smooth_nan(sph_ext_km, sigma=sigma)
            sph_top_km = gaussian_smooth_nan(sph_top_km, sigma=sigma)
            sph_bottom_km = gaussian_smooth_nan(sph_bottom_km, sigma=sigma)
            for j in range(sph_ext_layers_ebd.shape[1]):
                sph_ext_layers_ebd[:, j] = gaussian_smooth_nan(sph_ext_layers_ebd[:, j], sigma=sigma)
                sph_top_layers_ebd[:, j] = gaussian_smooth_nan(sph_top_layers_ebd[:, j], sigma=sigma)
                sph_bottom_layers_ebd[:, j] = gaussian_smooth_nan(sph_bottom_layers_ebd[:, j], sigma=sigma)

    # Multilayer -> single series aggregation (for stats/comparisons)
    sph_ext_km_multi = None
    sph_top_km_multi = None
    sph_bottom_km_multi = None

    if sph_ext_layers_ebd is not None and sph_top_layers_ebd is not None and sph_bottom_layers_ebd is not None:
        extL = np.asarray(sph_ext_layers_ebd, dtype=float)
        topL = np.asarray(sph_top_layers_ebd, dtype=float)
        botL = np.asarray(sph_bottom_layers_ebd, dtype=float)

        sph_top_km_multi = np.nanmax(topL, axis=1)       # uppermost top
        sph_bottom_km_multi = np.nanmin(botL, axis=1)    # lowermost bottom

        with np.errstate(all='ignore'):
            sph_ext_km_multi = np.nanmean(extL, axis=1)  # average over layers
        all_nan = ~np.any(np.isfinite(extL), axis=1)
        sph_ext_km_multi[all_nan] = np.nan


    # Interpolate EBD-derived SPH onto TC time (dt_utc)
    sph_ext_on_tc_dt = _interp_times(sph_ext_time.to_numpy(), sph_ext_km, dt_utc.to_numpy()) if sph_ext_time is not None and sph_ext_km is not None else None
    sph_top_on_tc_dt = _interp_times(sph_top_time.to_numpy(), sph_top_km, dt_utc.to_numpy()) if sph_top_time is not None and sph_top_km is not None else None
    sph_bottom_on_tc_dt = _interp_times(sph_bottom_time.to_numpy(), sph_bottom_km, dt_utc.to_numpy()) if sph_bottom_time is not None and sph_bottom_km is not None else None

    # ---------- SPH from FLEXPART concentration (multilayer) ----------
    fp_arr = _mask_like_plot(fp_plot_masked, "flex_bc")  # (track, z)
    has_plotted_fp = np.any(np.isfinite(fp_arr), axis=1)

    if config.set_flexpart_res_to_original:
        # Stable detection grid: representative heights (km AMSL)
        z_fp_plot = np.nanmean(fp_height_plot_km, axis=0)

        # Per-time offset relative to representative grid
        dz_shift = np.full(fp_arr.shape[0], np.nan, dtype=float)
        for i in range(fp_arr.shape[0]):
            zi = np.asarray(fp_height_plot_km[i, :], dtype=float)
            m = np.isfinite(zi) & np.isfinite(z_fp_plot)
            if np.any(m):
                dz_shift[i] = np.nanmedian(zi[m] - z_fp_plot[m])
    else:
        z_fp_plot = np.asarray(height_km_tc[0, :], dtype=float)
        dz_shift = np.zeros(fp_arr.shape[0], dtype=float)

    # Ensure increasing z for detection
    if np.any(np.diff(z_fp_plot) <= 0):
        order = np.argsort(z_fp_plot)
        z_fp_plot = z_fp_plot[order]
        fp_arr = fp_arr[:, order]

    n_t_fp = fp_arr.shape[0]
    max_layers_keep = int(config.max_layers)
    min_bins_keep = int(config.sph_min_bins)

    sph_flex_ext_layers = np.full((n_t_fp, max_layers_keep), np.nan, dtype=float)
    sph_flex_top_layers = np.full((n_t_fp, max_layers_keep), np.nan, dtype=float)
    sph_flex_bottom_layers = np.full((n_t_fp, max_layers_keep), np.nan, dtype=float)

    sph_flex_ext_km = np.full(n_t_fp, np.nan, dtype=float)
    sph_flex_top_km = np.full(n_t_fp, np.nan, dtype=float)
    sph_flex_bottom_km = np.full(n_t_fp, np.nan, dtype=float)

    sph_flex_n_layers = np.full(n_t_fp, np.nan, dtype=float)
    sph_flex_is_multilayer = np.zeros(n_t_fp, dtype=bool)

    for i in range(n_t_fp):
        prof = fp_arr[i, :]
        if np.count_nonzero(np.isfinite(prof)) < min_bins_keep:
            continue
        if not np.any(np.isfinite(prof)):
            continue

        try:
            layers = sph_layers(
                z=z_fp_plot,
                beta=prof,
                threshold=float(config.sph_threshold_flex),
                min_bins=min_bins_keep,
                max_layers=max_layers_keep,
            )
        except Exception:
            layers = dict(n_layers=0, bottom=np.array([]), top=np.array([]), ext=np.array([]))

        dz = dz_shift[i] if np.isfinite(dz_shift[i]) else 0.0
        nL = int(layers.get("n_layers", 0))
        if nL > 0:
            sph_flex_n_layers[i] = float(nL)
            sph_flex_is_multilayer[i] = (nL >= 2)

            sph_flex_bottom_layers[i, :nL] = np.asarray(layers["bottom"], dtype=float) + dz
            sph_flex_top_layers[i, :nL] = np.asarray(layers["top"], dtype=float) + dz
            # NOTE: layers["ext"] is *not* a height; your original code added dz to it.
            # Keeping that behavior for compatibility, even if naming is misleading.
            sph_flex_ext_layers[i, :nL] = np.asarray(layers["ext"], dtype=float) + dz

        # strongest layer series
        try:
            v = sph_ext(beta=prof, z=z_fp_plot, threshold=float(config.sph_threshold_flex))
            sph_flex_ext_km[i] = (v + dz) if np.isfinite(v) else np.nan
        except Exception:
            pass
        try:
            st = sph_top(z=z_fp_plot, beta=prof, threshold=float(config.sph_threshold_flex))
            sph_flex_top_km[i] = (st + dz) if (st is not None and np.isfinite(st)) else np.nan
        except Exception:
            pass
        try:
            sb = sph_bottom(z=z_fp_plot, beta=prof, threshold=float(config.sph_threshold_flex))
            sph_flex_bottom_km[i] = (sb + dz) if (sb is not None and np.isfinite(sb)) else np.nan
        except Exception:
            pass

    # Mask where FLEXPART plot has nothing
    no_plot_fp = ~has_plotted_fp
    sph_flex_ext_km[no_plot_fp] = np.nan
    sph_flex_top_km[no_plot_fp] = np.nan
    sph_flex_bottom_km[no_plot_fp] = np.nan
    sph_flex_ext_layers[no_plot_fp, :] = np.nan
    sph_flex_top_layers[no_plot_fp, :] = np.nan
    sph_flex_bottom_layers[no_plot_fp, :] = np.nan
    sph_flex_n_layers[no_plot_fp] = np.nan
    sph_flex_is_multilayer[no_plot_fp] = False


    # Multilayer -> single series aggregation (for stats/comparisons)
    sph_flex_ext_km_multi = None
    sph_flex_top_km_multi = None
    sph_flex_bottom_km_multi = None

    if sph_flex_ext_layers is not None and sph_flex_top_layers is not None and sph_flex_bottom_layers is not None:
        extL = np.asarray(sph_flex_ext_layers, dtype=float)
        topL = np.asarray(sph_flex_top_layers, dtype=float)
        botL = np.asarray(sph_flex_bottom_layers, dtype=float)

        sph_flex_top_km_multi = np.nanmax(topL, axis=1)
        sph_flex_bottom_km_multi = np.nanmin(botL, axis=1)

        with np.errstate(all='ignore'):
            sph_flex_ext_km_multi = np.nanmean(extL, axis=1)
        all_nan = ~np.any(np.isfinite(extL), axis=1)
        sph_flex_ext_km_multi[all_nan] = np.nan



    sph_flex_time = dt_fp

    # ---------- SPH from ATL_TC extinction (optional, strongest layer only) ----------
    sph_tc_time = dt_utc
    sph_tc_km = None
    if tc_ext is not None:
        tc_arr = np.array(tc_ext, dtype=float)
        tc_arr = ensure_2d_time_height(tc_arr, n_time=len(dt_utc), n_z=height_km_tc.shape[1])

        z_tc = np.asarray(height_km_tc[0, :], dtype=float)
        if np.any(np.diff(z_tc) <= 0):
            order = np.argsort(z_tc)
            z_tc = z_tc[order]
            tc_arr = tc_arr[:, order]

        sph_tc_km = np.full(len(dt_utc), np.nan, dtype=float)
        for i in range(tc_arr.shape[0]):
            prof = tc_arr[i, :]
            if not np.any(np.isfinite(prof)):
                continue
            try:
                sph_tc_km[i] = sph_ext(beta=prof, z=z_tc, beta_min=0.0)
            except Exception:
                pass

    # ---------- GFAS ----------
    gfas = None
    gfas_min_distance_km = np.nan
    gfas_fire_matches = None   


    if gfas_folder is not None:
        gfas_path = os.path.join(gfas_folder, f"BC_GFAS_{gfas_variant}_{date_ymd}.nc")
        ds_gfas = xr.open_dataset(gfas_path, decode_times=True)
        if (ds_gfas["longitude"] >= 0).all():
            ds_gfas = ds_gfas.assign_coords(longitude=wrap_longitudes(ds_gfas["longitude"])).sortby("longitude")

        # --- Robust latitude slicing (works whether latitude is ascending or descending) ---
        lat = ds_gfas["latitude"].values
        if lat[0] < lat[-1]:  # ascending
            ds_gfas_sub = ds_gfas.sel(latitude=slice(30, 90), longitude=slice(-180, 180))
        else:                 # descending
            ds_gfas_sub = ds_gfas.sel(latitude=slice(90, 30), longitude=slice(-180, 180))

        # --- Force longitude range to [-180, 180) for the subset used below ---
        lon = ds_gfas_sub["longitude"].values
        if np.nanmax(lon) > 180.0:
            ds_gfas_sub = ds_gfas_sub.assign_coords(
                longitude=(((ds_gfas_sub["longitude"] + 180) % 360) - 180)
            ).sortby("longitude")



        fires_df = gfas_fire_points_with_injection_heights(
                ds_gfas_sub,
                frp_var="frpfire",
                time_index=0,
                min_frp=float(config.gfas_min_frp),
                use_canada_only=bool(config.use_canada_only),
                canada_bbox=config.canada_bbox if config.use_canada_only else None,
            )
        
        # --- Build GFAS plotting payload for the top map panel (independent of smoke-only TC-SPH) ---
        gfas = None
        if fires_df is not None and (not fires_df.empty):
            # Use fire points (already filtered by min_frp and optional Canada bbox)
            vals = fires_df["frp"].to_numpy(dtype=float)
            lon_flat = fires_df["lon"].to_numpy(dtype=float)
            lat_flat = fires_df["lat"].to_numpy(dtype=float)

            bins = [float(config.gfas_min_frp), 5, 10, 50, 100, np.inf]
            colors = ["#ff6e6e", "#ff0000", "#b62020", "#800000", "#000000"]
            labels = [f"{bins[0]}-5", "5-10", "10-50", "50-100", ">100"]

            gfas = dict(
                path=gfas_path,
                vals=vals,
                lon_flat=lon_flat,
                lat_flat=lat_flat,
                bins=bins,
                colors=colors,
                labels=labels,
            )

        
        if not fires_df.empty:
            # Use the same ds_geop you already opened earlier in prepare_data(...)
            inv_g = 1.0 / 9.80665
            h_surf_fire_m = (ds_geop["z"] * inv_g).interp(
                latitude=xr.DataArray(fires_df["lat"].values, dims=["p"]),
                longitude=xr.DataArray(wrap_longitudes(fires_df["lon"].values), dims=["p"]),
                method="nearest",
            ).to_numpy()
            fires_df["fire_elev_km"] = 0.001 * h_surf_fire_m.astype(float)

            def _m_to_km(s: pd.Series) -> pd.Series:
                s = pd.to_numeric(s, errors="coerce")
                # if values look like meters, convert to km
                if np.nanmedian(s.values) > 100.0:
                    return s / 1000.0
                return s

            fires_df["mami_km_raw"] = _m_to_km(fires_df["mami"])
            fires_df["injh_km_raw"] = _m_to_km(fires_df["injh"])
            fires_df["apt_km_agl"]   = _m_to_km(fires_df["apt"])  # AGL by definition
            fires_df["apb_km_agl"]   = _m_to_km(fires_df["apb"])  # AGL by definition

            # AGL -> AMSL
            fires_df["apt_km_amsl"] = fires_df["fire_elev_km"] + fires_df["apt_km_agl"]
            fires_df["apb_km_amsl"] = fires_df["fire_elev_km"] + fires_df["apb_km_agl"]

            # mami/injh sometimes are stored as AMSL in some GFAS-derived products.
            # Provide BOTH interpretations so you can see which one makes sense.
            fires_df["mami_km_amsl_assume_amsl"] = fires_df["mami_km_raw"]
            fires_df["mami_km_amsl_assume_agl"]  = fires_df["fire_elev_km"] + fires_df["mami_km_raw"]
            fires_df["injh_km_amsl_assume_amsl"] = fires_df["injh_km_raw"]
            fires_df["injh_km_amsl_assume_agl"]  = fires_df["fire_elev_km"] + fires_df["injh_km_raw"]

        if not fires_df.empty:
            # Choose plume presence series: use top by default
            plume_series = sph_top_on_tc_dt if sph_top_on_tc_dt is not None else sph_ext_on_tc_dt

            gfas_fire_matches = match_gfas_fires_to_earthcare_plumes(
                fires_df,
                track_lon=lon_track_wrapped,
                track_lat=lat_track,
                plume_series=np.asarray(plume_series, dtype=float),
                sph_top=np.asarray(sph_top_on_tc_dt, dtype=float) if sph_top_on_tc_dt is not None else None,
                sph_bottom=np.asarray(sph_bottom_on_tc_dt, dtype=float) if sph_bottom_on_tc_dt is not None else None,
                sph_ext=np.asarray(sph_ext_on_tc_dt, dtype=float) if sph_ext_on_tc_dt is not None else None,
                max_track_distance_km=float(config.gfas_max_distance_km),
                plume_search_max_km=100.0,  # or make it config-driven if you want
            )

        if gfas_fire_matches is not None and not gfas_fire_matches.empty:
            # EarthCARE SPH top ↔ GFAS apt (top AGL → AMSL)
            gfas_fire_matches["d_top_minus_apt"] = gfas_fire_matches["ec_sph_top_km"] - gfas_fire_matches["apt_km_amsl"]

            # EarthCARE SPH bottom ↔ GFAS apb (bottom AGL → AMSL)
            gfas_fire_matches["d_bottom_minus_apb"] = gfas_fire_matches["ec_sph_bottom_km"] - gfas_fire_matches["apb_km_amsl"]

            # EarthCARE SPH ext (representative/centroid-like) ↔ GFAS mami (two interpretations)
            gfas_fire_matches["d_ext_minus_mami_assume_amsl"] = (
                gfas_fire_matches["ec_sph_ext_km"] - gfas_fire_matches["mami_km_amsl_assume_amsl"]
            )
            gfas_fire_matches["d_ext_minus_mami_assume_agl"] = (
                gfas_fire_matches["ec_sph_ext_km"] - gfas_fire_matches["mami_km_amsl_assume_agl"]
            )

            # Optional: also compare top to injh both ways (often “top-like”)
            gfas_fire_matches["d_top_minus_injh_assume_amsl"] = (
                gfas_fire_matches["ec_sph_top_km"] - gfas_fire_matches["injh_km_amsl_assume_amsl"]
            )
            gfas_fire_matches["d_top_minus_injh_assume_agl"] = (
                gfas_fire_matches["ec_sph_top_km"] - gfas_fire_matches["injh_km_amsl_assume_agl"]
            )
        
        # -------------------------------------------------------------
        # tag whether the matched EarthCARE track point contains smoke-like aerosols in the ATL_TC classification.
        # -------------------------------------------------------------
        try:
            smoke_ids = tuple(getattr(config, "tc_smoke_class_ids", (13, 14, 27)))
            cls2d = ensure_2d_time_height(class_tc, n_time=len(dt_utc), n_z=height_km_tc.shape[1])
            ti = pd.to_numeric(gfas_fire_matches.get("track_i_used"), errors="coerce")

            def _row_is_smoke(v) -> bool:
                if not np.isfinite(v):
                    return False
                i = int(v)
                if i < 0 or i >= cls2d.shape[0]:
                    return False
                return bool(np.any(np.isin(cls2d[i, :], smoke_ids)))

            gfas_fire_matches["is_smoke"] = ti.apply(_row_is_smoke).astype(bool)
        except Exception:
            # If anything goes wrong (missing variables, etc.), just skip tagging.
            pass


        # ---------------------------------------------------------------------
        # Distance from track to nearest GFAS fire (use fires_df; do NOT re-mask via lon_flat/lat_flat)
        # ---------------------------------------------------------------------
        gfas_min_distance_km = np.nan

        if fires_df is not None and (not fires_df.empty):
            dmin = np.inf

            fire_lons = fires_df["lon"].to_numpy(dtype=float)
            fire_lats = fires_df["lat"].to_numpy(dtype=float)

            # lon_track_wrapped already exists in your function (you use it above)
            tlon = np.asarray(lon_track_wrapped, dtype=float)
            tlat = np.asarray(lat_track, dtype=float)

            for lg, bg in zip(fire_lons, fire_lats):
                _, _, dist_m = geod.inv(
                    tlon, tlat,
                    np.full_like(tlon, lg, dtype=float),
                    np.full_like(tlat, bg, dtype=float),
                )
                dm = np.nanmin(dist_m) / 1000.0
                if dm < dmin:
                    dmin = dm

            if np.isfinite(dmin):
                gfas_min_distance_km = float(dmin)

        if config.only_overpasses_near_gfas:
            max_km = float(config.gfas_max_distance_km)
            md = gfas_min_distance_km
            if not (np.isfinite(md) and (md <= max_km)):
                print(f"      Skipping (min distance to GFAS = {md} km > {max_km} km)")
                return {
                    "skipped": True,
                    "skip_reason": f"min distance to GFAS = {md} km > {max_km} km",
                    "gfas_min_distance_km": md,
                    "earthcare_file": earthcare_file,
                    "flexpart_nc": flexpart_nc,
                }




    # ---------- Track styling ----------
    frame_colors = {"D": "red", "B": "blue", "C": "black"}
    track_color = frame_colors.get(frame, "gray")
    track_name = {"D": "N -> S", "B": "S -> N", "C": "Cross Polar"}

    return dict(
        # Inputs
        flexpart_nc=flexpart_nc,
        earthcare_file=earthcare_file,
        ebd_file=ebd_file,

        # Track grids
        lat_ec=lat_track,
        lon_ec=lon_track,
        lon_ec_wrapped=lon_track_wrapped,
        dt=dt_utc,
        lt=dt_lt,

        # TC products
        tropo=tropopause_km,
        height=height_km_tc,
        dt_ext=dt_2d_tc,
        class_data_nocloud=class_data_nocloud,
        class_data=class_tc,
        class_data_raw=class_tc_raw,
        tc_ext=tc_ext,
        tc_ext_raw=tc_ext_raw,

        # Surface
        elev=elev_km,

        # FLEXPART
        col=col_bc,
        fp_interp=fp_plot_masked,
        fp_plot_height=fp_height_plot_km,
        fp_plot_time=fp_time_plot,
        fp_height_shifted_km=z_fp_amsl_km_track,

        fp_dt_1d=dt_fp,
        fp_lt_1d=lt_fp,
        fp_lat=lat_fp,
        fp_lon=lon_fp_wrapped,
        fp_elev=elev_fp,
        fp_tropo=tropo_fp,


        # EBD
        ebd_ext=ebd_ext,
        ebd_ext_raw=ebd_ext_raw,
        ebd_cloud_mask_on_ebdgrid=ebd_cloud_mask_on_ebdgrid,
        ebd_height_km=ebd_height_km,
        ebd_dt=ebd_dt,

        # SPH (EBD)
        sph_ext_time=sph_ext_time,
        sph_ext_km=sph_ext_km,
        sph_top_time=sph_top_time,
        sph_top_km=sph_top_km,
        sph_bottom_time=sph_bottom_time,
        sph_bottom_km=sph_bottom_km,
        sph_ext_on_tc_dt=sph_ext_on_tc_dt,
        sph_top_on_tc_dt=sph_top_on_tc_dt,
        sph_bottom_on_tc_dt=sph_bottom_on_tc_dt,

        ebd_is_multilayer=ebd_is_multilayer,
        ebd_n_layers=ebd_n_layers,
        sph_ext_layers_ebd=sph_ext_layers_ebd,
        sph_top_layers_ebd=sph_top_layers_ebd,
        sph_bottom_layers_ebd=sph_bottom_layers_ebd,

        # SPH (FLEXPART)
        sph_flex_time=sph_flex_time,
        sph_flex_ext_km=sph_flex_ext_km,
        sph_flex_top_km=sph_flex_top_km,
        sph_flex_bottom_km=sph_flex_bottom_km,
        sph_flex_is_multilayer=sph_flex_is_multilayer,
        sph_flex_n_layers=sph_flex_n_layers,
        sph_flex_ext_layers=sph_flex_ext_layers,
        sph_flex_top_layers=sph_flex_top_layers,
        sph_flex_bottom_layers=sph_flex_bottom_layers,

        # SPH (TC)
        sph_tc_time=sph_tc_time,
        sph_tc_km=sph_tc_km,

        # GFAS
        gfas=gfas,
        gfas_min_distance_km=gfas_min_distance_km,
        gfas_fire_matches=gfas_fire_matches,

        # Meta
        frame=frame,
        track_color=track_color,
        track_name=track_name,
        date_str=date_str,
        date_ymd=date_ymd,
        baseline=baseline,
        config=config,

        # ALD
        ald_time=ald_time,
        ald_top_height_km=ald_top_height_km,
        ald_bottom_height_km=ald_bottom_height_km,
        ald_layer_number=ald_layer_number,

        # Aggregated multilayer series (preferred for statistics)
        sph_ext_km_multi=sph_ext_km_multi,
        sph_top_km_multi=sph_top_km_multi,
        sph_bottom_km_multi=sph_bottom_km_multi,

        # Aggregated multilayer series (preferred for statistics)
        sph_flex_ext_km_multi=sph_flex_ext_km_multi,
        sph_flex_top_km_multi=sph_flex_top_km_multi,
        sph_flex_bottom_km_multi=sph_flex_bottom_km_multi,



    )


def plot_earthcare_only_tc_and_ebd(data: Dict[str, Any], output_dir: str) -> None:
    cfg: PlotConfig = data.get("config", PlotConfig())
    os.makedirs(output_dir, exist_ok=True)

    out_name = re.sub(
        r".*_(\d{8})T(\d{6})Z_(\d{8})T(\d{6})Z_([A-Za-z0-9]+).*",
        r"EC_TC_EBD_RAW_\1_T\2Z_T\4Z_\5.png",
        os.path.basename(data["earthcare_file"]),
    )
    out_png = os.path.join(output_dir, out_name)

    plt.rcParams.update({
        "axes.titlesize": 26,
        "axes.labelsize": 26,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 20,
        "figure.titlesize": 30,
    })

    fig = plt.figure(figsize=(23, 17.0))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 1], hspace=0.08)  # CHANGED: slightly smaller gap

    # --- TOP: ATL_TC classification (RAW full classes) ---
    ax_tc = fig.add_subplot(gs[0, 0])

    class_tc = data.get("class_data_raw", data["class_data"])
    dt_ext = data["dt_ext"]
    height = data["height"]

    TC_CLASSES = {
        -3: ("Missing data", "#f5f5dc"),
        -2: ("Surface or sub-surface", "#000000"),
        -1: ("Noise in Mie & Ray channels", "#c4c4c4"),
        0: ("Clear", "#ffffff"),
        1: ("(Warm) Liquid cloud", "#5555fe"),
        2: ("(Supercooled) Liquid cloud", "#0018e0"),
        3: ("Ice cloud", "#009bce"),
        10: ("Dust", "#a22f2e"),
        11: ("Sea salt", "#acd6e2"),
        12: ("Continental pollution", "#02fd7f"),
        13: ("Smoke", "#2f504e"),
        14: ("Dusty smoke", "#996633"),
        15: ("Dusty mix", "#e0b98b"),
        20: ("STS", "#ff00ff"),
        21: ("NAT", "#8f37df"),
        22: ("Stratospheric ice", "#4a0061"),
        25: ("Stratospheric ash", "#fece05"),
        26: ("Stratospheric sulfate", "#fdfb0d"),
        27: ("Stratospheric smoke", "#c3b27f"),
    }

    cfg: PlotConfig = data.get("config", PlotConfig())

    # IMPORTANT: choose what we show based on cfg.only_smoke_for_tc_sph
    tc_remapped, keep_classes, class_cmap, class_norm = _tc_plot_classes_and_remap(
        class_tc,
        tc_classes=TC_CLASSES,
        only_smoke=bool(cfg.only_smoke_for_tc_sph),
        smoke_ids=tuple(cfg.tc_smoke_class_ids),
        include_clear=True,
        include_surface=True,
        include_missing=True,
    )

    im_tc = ax_tc.pcolormesh(dt_ext, height, tc_remapped, cmap=class_cmap, norm=class_norm, shading="auto")

    cax_tc = fig.add_axes([...])
    cbar_tc = plt.colorbar(im_tc, cax=cax_tc, spacing="uniform")
    cbar_tc.set_ticks(np.arange(len(keep_classes)))
    cbar_tc.set_ticklabels([TC_CLASSES[cid][0] for cid in keep_classes])


    im_tc = ax_tc.pcolormesh(dt_ext, height, tc_remapped, cmap=class_cmap, norm=class_norm, shading="auto")
    ax_tc.plot(data["dt"], data["elev"], color="black", linewidth=2.0)
    ax_tc.plot(data["dt"], data["tropo"], color="black", linestyle="--", linewidth=2.0, label="Tropopause")
    ax_tc.set_ylabel("Altitude [km]")
    ax_tc.set_ylim(0, 20)
    ax_tc.legend(loc="upper right") #, fontsize=18)
    ax_tc.set_title(f"EarthCARE ATL_TC Target Classification for {data['date_str']}",
                    fontweight="bold") #, fontsize=30)  # CHANGED

    # -----------------------------
    # Remove time labels between plots (from TC)
    # -----------------------------
    ax_tc.tick_params(axis="x", which="both", bottom=False, labelbottom=False)  # <- key line

    cax_tc = fig.add_axes([ax_tc.get_position().x1 + 0.01, ax_tc.get_position().y0, 0.015, ax_tc.get_position().height])
    cbar_tc = plt.colorbar(im_tc, cax=cax_tc, spacing="uniform")
    cbar_tc.set_ticks(np.arange(len(keep_classes)))
    cbar_tc.set_ticklabels([TC_CLASSES[cid][0] for cid in keep_classes])
    cbar_tc.ax.tick_params(labelsize=16, pad=4)  

    # --- BOTTOM: ATL_EBD extinction ---
    ax_ebd = fig.add_subplot(gs[1, 0], sharex=ax_tc)

    ebd_ext = data.get("ebd_ext", data.get("ebd_ext_raw"))
    ebd_dt = data.get("ebd_dt")
    ebd_z = data.get("ebd_height_km")


    if (ebd_ext is None) or (ebd_dt is None) or (ebd_z is None) or (len(np.asarray(ebd_z)) == 0):
        ax_ebd.text(0.5, 0.5, "No EBD available", transform=ax_ebd.transAxes,
                    ha="center", va="center") #, fontsize=22)  # CHANGED
        ax_ebd.set_ylabel("Altitude [km]")
        ax_ebd.set_ylim(0, 20)
    else:
        e = np.array(np.ma.filled(ebd_ext, np.nan), dtype=float)
        # e_mask = np.isfinite(e) & (e > 0.0)
        e_mask = np.isfinite(e) & (e > 0.0) & (e <= 9.0)

        e_plot = np.where(e_mask, e, np.nan)

        im_ebd = ax_ebd.pcolormesh(
            pd.to_datetime(ebd_dt).to_numpy(),
            np.asarray(ebd_z, dtype=float),
            e_plot.T,
            shading="auto",
            cmap="jet",
            norm=LogNorm(vmin=1e-6, vmax=1e-3),
        )
        ax_ebd.plot(data["dt"], data["elev"], color="black", linewidth=2.0)
        ax_ebd.plot(data["dt"], data["tropo"], color="black", linestyle="--", linewidth=2.0)
        ax_ebd.set_ylabel("Altitude [km]")
        ax_ebd.set_ylim(0, 20)
        ax_ebd.set_title("EarthCARE ATL_EBD Extinction", fontweight="bold") #, fontsize=30)

        cax_ebd = fig.add_axes([ax_ebd.get_position().x1 + 0.01, ax_ebd.get_position().y0, 0.015, ax_ebd.get_position().height])
        cbar_ebd = plt.colorbar(im_ebd, cax=cax_ebd)
        cbar_ebd.set_label(r"[m$^{-1}$]" +"\n \n"  "Extinction \nCoefficient", fontweight="bold") # fontsize=22)
        cbar_ebd.ax.tick_params(labelsize=18)  

    # --- ticks / labels only on bottom axis ---
    forced_lat_min = 35.0
    track_mask = np.asarray(data["lat_ec"]) >= forced_lat_min

    if np.any(track_mask):
        dt_m  = np.asarray(data["dt"])[track_mask]
        lt_m  = np.asarray(data["lt"])[track_mask]
    else:
        dt_m  = np.asarray(data["dt"])
        lt_m  = np.asarray(data["lt"])

    if dt_m.size:
        ax_tc.set_xlim(dt_m[0], dt_m[-1])

    ticks = pd.date_range(pd.to_datetime(dt_m[0]).ceil("1min"),
                          pd.to_datetime(dt_m[-1]).floor("1min"),
                          freq="1min")
    dt_m_pd = pd.to_datetime(dt_m)
    indices = np.clip(np.searchsorted(dt_m_pd, ticks), 1, len(dt_m_pd) - 1)
    indices -= (ticks - dt_m_pd[indices - 1] < dt_m_pd[indices] - ticks)
    lticks = pd.to_datetime(lt_m)[indices]

    ax_ebd.set_xticks(ticks)
    ax_ebd.set_xticklabels([f"{t:%H:%M}\n{lt:%H:%M}" for t, lt in zip(ticks, lticks)]) #, fontsize=20)  
    ax_ebd.set_xlabel("Time [UTC / LT]", labelpad=10, fontweight="bold")  # fontsize=24)

    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Plotting
# -----------------------------
def plot_flexpart_earthcare_from_data(data: Dict[str, Any], output_dir: str) -> None:
    cfg: PlotConfig = data.get("config", PlotConfig())

    if cfg.plot_only_TC_and_extinction_alone:
        return plot_earthcare_only_tc_and_ebd(data, output_dir)


    def format_ticks(datetime_utc: pd.Series, datetime_lt: pd.Series):
        ticks = pd.date_range(datetime_utc.min().ceil("1min"), datetime_utc.max().floor("1min"), freq="1min")
        lticks = pd.date_range(datetime_lt.min().ceil("1min"), datetime_lt.max().floor("1min"), freq="1min")
        indices = np.clip(np.searchsorted(datetime_utc, ticks), 1, len(datetime_utc) - 1)
        indices -= (ticks - datetime_utc[indices - 1] < datetime_utc[indices] - ticks)
        return ticks, lticks, indices

    def setup_ax_labels(ax, time_ticks, local_ticks, datetime_utc, lat, lon, indices):
        # Ensure datetime array is pandas datetime for consistent comparisons
        datetime_utc = pd.to_datetime(datetime_utc)

        # Bottom axis: UTC / LT tick labels
        ax.set_xticks(time_ticks)
        ax.set_xticklabels(
            [f"{t:%H:%M}\n{lt:%H:%M}" for t, lt in zip(time_ticks, local_ticks)],
            fontsize=15,
        )

        cur_xlim = ax.get_xlim()
        if not np.isfinite(cur_xlim).all():
            ax.set_xlim(datetime_utc.iloc[0], datetime_utc.iloc[-1])

        # Top axis: lat/lon at the same tick positions
        ax_top2 = ax.twiny()
        ax_top2.set_xlim(ax.get_xlim())

        lat = np.asarray(lat)
        lon = np.asarray(lon)
        indices = np.asarray(indices, dtype=int)

        # Guard against out-of-range indices
        n = min(len(lat), len(lon))
        indices = np.clip(indices, 0, max(n - 1, 0))

        latlon_labels = []
        for la, lo in zip(lat[indices], lon[indices]):
            if not np.isfinite(la) or not np.isfinite(lo):
                latlon_labels.append("")
                continue

            lat_hem = "S" if la < 0 else "N"
            lon_hem = "W" if lo < 0 else "E"
            latlon_labels.append(
                f"${abs(la):.1f}^\\circ${lat_hem}\n${abs(lo):.1f}^\\circ${lon_hem}"
            )

        ax_top2.set_xticks(time_ticks)
        ax_top2.set_xticklabels(latlon_labels, fontsize=15, ha="right")

        return ax_top2


    os.makedirs(output_dir, exist_ok=True)

    out_name = re.sub(
        r".*_(\d{8})T(\d{6})Z_(\d{8})T(\d{6})Z_([A-Za-z0-9]+).*",
        r"EC_FLEXPART_GFAS_\1_T\2Z_T\4Z_\5.png",
        os.path.basename(data["earthcare_file"]),
    )
    out_png = os.path.join(output_dir, out_name)

    if (not cfg.force_replot) and os.path.exists(out_png):
        print(f"Skipping {os.path.basename(data['earthcare_file'])} already plotted")
        return

    # fig = plt.figure(figsize=(36, 26))
    fig = plt.figure(figsize=(24, 19))

    gs = GridSpec(41, 9, figure=fig)
    plt.rcParams.update(
        {
            "axes.titlesize": 28,
            "axes.labelsize": 24,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 18,
            "figure.titlesize": 30,
        }
    )

    # --- Top map ---
    ax_top = fig.add_subplot(gs[0:9, :], projection=ccrs.PlateCarree())
    ax_top.set_aspect("auto")
    ax_top.set_facecolor("none")
    ax_top.coastlines(linewidth=0.8, color="black")
    ax_top.add_feature(cfeature.LAND.with_scale("50m"), facecolor="white", edgecolor="none", zorder=0)
    ax_top.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor=(0.6, 0.8, 1.0, 0.3), edgecolor="none", zorder=0)
    ax_top.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.5, edgecolor="black")

    gl = ax_top.gridlines(draw_labels=True, linestyle="--", linewidth=0.6, alpha=1)
    for attr in ("top_labels", "xlabels_top"):
        if hasattr(gl, attr):
            setattr(gl, attr, False)
    for attr in ("right_labels", "ylabels_right"):
        if hasattr(gl, attr):
            setattr(gl, attr, False)
    for attr in ("left_labels", "ylabels_left"):
        if hasattr(gl, attr):
            setattr(gl, attr, True)
    for attr in ("bottom_labels", "xlabels_bottom"):
        if hasattr(gl, attr):
            setattr(gl, attr, True)
    gl.xlabel_style = {"size": 14}
    gl.ylabel_style = {"size": 14}

    ax_top.set_extent([-180, 180, 30, 90], crs=ccrs.PlateCarree())
    ax_top.autoscale(False)

    threshold_map = float(cfg.flexpart_column_threshold_map)
    cmap_obj = plt.colormaps.get_cmap("plasma_r").copy()
    cmap_obj.set_under("white")

    mesh = ax_top.pcolormesh(
        data["col"]["longitude"],
        data["col"]["latitude"],
        data["col"].where(data["col"] >= threshold_map),
        transform=ccrs.PlateCarree(),
        shading="auto",
        cmap=cmap_obj,
        vmin=threshold_map,
    )

    frame = data["frame"]
    track_name = data["track_name"].get(frame, "Track")

    ax_top.plot(
        data["lon_ec"],
        data["lat_ec"],
        color=data["track_color"],
        linewidth=2,
        transform=ccrs.Geodetic(),
        label=f"EarthCARE Track ({track_name})",
    )
    ax_top.legend(loc="lower right", fontsize=15)
    ax_top.set_title(
        f"Comparison of FLEXPART Black Carbon, GFAS FRP, and EarthCARE ATL Products on {data['date_str']}",
        fontweight="bold",
        fontsize=28,
    )

    gfas = data.get("gfas")
    if gfas is not None:
        vals = gfas["vals"]
        lon_flat = gfas["lon_flat"]
        lat_flat = gfas["lat_flat"]
        bins = gfas["bins"]
        colors = gfas["colors"]
        labels = gfas["labels"]

        mask_valid = vals > float(cfg.gfas_min_frp)
        for j in range(len(bins) - 1):
            mask_bin = (vals > bins[j]) & (vals <= bins[j + 1]) & mask_valid
            if np.any(mask_bin):
                ax_top.scatter(
                    lon_flat[mask_bin],
                    lat_flat[mask_bin],
                    color=colors[j],
                    s=10,
                    label=labels[j],
                    transform=ccrs.PlateCarree(),
                    zorder=3,
                    linewidths=0,
                )

        earthcare_handle = Line2D([0], [0], color=data["track_color"], linewidth=2, label=f"EarthCARE Track {track_name}")
        spacer_handle = Line2D([], [], linestyle="none", marker=None, label="")
        frp_title_handle = Line2D([0], [0], linestyle="none", marker=None, label="FRP (W m$^{-2}$)")
        bin_handles = [
            Line2D([0], [0], linestyle="none", marker="o", markersize=6, color=colors[j], label=labels[j])
            for j in range(len(labels))
        ]
        ordered_handles = [earthcare_handle, spacer_handle, frp_title_handle] + bin_handles
        legend2 = ax_top.legend(
            handles=ordered_handles,
            loc="lower right",
            fontsize=14,
            frameon=True,
            title=None,
            handlelength=2,
            scatterpoints=1,
            borderpad=0.8,
            labelspacing=0.5,
        )
        legend2.get_frame().set_alpha(0.9)

    cax_top = fig.add_axes([ax_top.get_position().x1 + 0.01, ax_top.get_position().y0, 0.015, ax_top.get_position().height])
    cbar_top = plt.colorbar(mesh, cax=cax_top)
    cbar_top.set_label(
        "[μg m⁻²]\n \n \n" + r"$\mathbf{FLEXPART}$ $\mathbf{Total}$ " + "\n" + r"$\mathbf{Column}$ $\mathbf{BC}$",
        labelpad=10,
        fontsize=24,
    )

    # --- FLEXPART cross-section ---
    ax1 = fig.add_subplot(gs[12:21, :])
    cmap_fp = plt.get_cmap("plasma_r")
    cmap_fp.set_bad("white")
    norm_fp = Normalize(cfg.flexpart_conc_norm[0], cfg.flexpart_conc_norm[1])

    h_for_plot = data["fp_plot_height"]
    fp_for_plot = data["fp_interp"]
    dt_for_plot = data["fp_plot_time"]

    # ensure increasing height for pcolormesh
    try:
        if np.nanmean(np.diff(h_for_plot, axis=1)) < 0:
            h_for_plot = h_for_plot[:, ::-1]
            fp_for_plot = fp_for_plot[:, ::-1]
    except Exception:
        pass

    im_fp = ax1.pcolormesh(dt_for_plot, h_for_plot, fp_for_plot, cmap=cmap_fp, norm=norm_fp, shading="auto")
    ax1.fill_between(data["fp_dt_1d"], 0, data["fp_elev"], color="black", zorder=2)
    ax1.plot(data["fp_dt_1d"], data["fp_elev"], color="black", linewidth=1.5, label="Surface Elevation")
    ax1.plot(data["fp_dt_1d"], data["fp_tropo"], color="black", linestyle="--", linewidth=1.5, label="Tropopause Height")

    # FLEXPART SPH (strongest layer)
    if data.get("sph_flex_time") is not None:
        if data.get("sph_flex_ext_km") is not None:
            ax1.plot(data["sph_flex_time"], data["sph_flex_ext_km"], color="black", marker="o", markersize=4, linestyle="None", label="sph_ext")
        if data.get("sph_flex_top_km") is not None:
            ax1.plot(data["sph_flex_time"], data["sph_flex_top_km"], color="#FF0000", marker="o", markersize=4, linestyle="None", label="sph_top", zorder=5)
        if data.get("sph_flex_bottom_km") is not None:
            ax1.plot(data["sph_flex_time"], data["sph_flex_bottom_km"], color="green", marker="o", markersize=4, linestyle="None", label="sph_bottom", zorder=5)

    # FLEXPART multilayer curves
    t = data.get("sph_flex_time")
    if t is not None:
        for key, color in [("sph_flex_ext_layers", "black"), ("sph_flex_top_layers", "#FF0000"), ("sph_flex_bottom_layers", "green")]:
            arr = data.get(key)
            if arr is None:
                continue
            arr = np.asarray(arr)
            for j in range(arr.shape[1]):
                ax1.plot(t, arr[:, j], color=color, marker="o", markersize=4, linestyle="None", alpha=0.9)

    ax1.set_ylabel("Altitude [km]")
    ax1.set_ylim(0, 20)
    ax1.legend(loc="upper right", fontsize=18)
    ax1.tick_params(labelbottom=False)

    forced_lat_min = 35.0
    track_mask = np.asarray(data["lat_ec"]) >= forced_lat_min

    if np.any(track_mask):
        dt_m  = np.asarray(data["dt"])[track_mask]
        lt_m  = np.asarray(data["lt"])[track_mask]
        lat_m = np.asarray(data["lat_ec"])[track_mask]
        lon_m = np.asarray(data["lon_ec"])[track_mask]
    else:
        dt_m  = np.asarray(data["dt"])
        lt_m  = np.asarray(data["lt"])
        lat_m = np.asarray(data["lat_ec"])
        lon_m = np.asarray(data["lon_ec"])


    # limit view
    if dt_m.size:
        ax1.set_xlim(dt_m[0], dt_m[-1])

    # build UTC ticks in the visible window
    ticks = pd.date_range(pd.to_datetime(dt_m[0]).ceil("1min"),
                        pd.to_datetime(dt_m[-1]).floor("1min"),
                        freq="1min")

    # map each tick -> nearest index in dt_m
    dt_m_pd = pd.to_datetime(dt_m)
    indices = np.clip(np.searchsorted(dt_m_pd, ticks), 1, len(dt_m_pd) - 1)
    indices -= (ticks - dt_m_pd[indices - 1] < dt_m_pd[indices] - ticks)

    # LT ticks aligned to the same indices (THIS is the key)
    lticks = pd.to_datetime(lt_m)[indices]

    setup_ax_labels(ax1, ticks, lticks, dt_m_pd, lat_m, lon_m, indices)






    pos1 = ax1.get_position()
    cax1 = fig.add_axes([pos1.x1 + 0.01, pos1.y0, 0.015, pos1.height])
    cbar1 = plt.colorbar(im_fp, cax=cax1)
    cbar1.set_label("[ng m⁻³]\n \n \n \n" + r"$\mathbf{FLEXPART}$ $\mathbf{BC}$" + "\n" + r" $\mathbf{Concentration}$", labelpad=10, fontsize=24)
    cbar1.ax.tick_params(labelsize=19)

    # --- ATL_EBD panel ---
    ax2 = fig.add_subplot(gs[22:31, :], sharex=ax1)
    if (data.get("ebd_ext") is not None) and (data.get("ebd_height_km") is not None) and (data.get("ebd_dt") is not None):
        ebd_ext_log = np.ma.masked_where(
            (data["ebd_ext"] <= 0) | (data["ebd_ext"] > 9.9) | ~np.isfinite(data["ebd_ext"]),
            data["ebd_ext"],
        )
        im_ebd = ax2.pcolormesh(
            data["ebd_dt"].to_numpy(),
            data["ebd_height_km"],
            ebd_ext_log.T,
            shading="auto",
            cmap="jet",
            norm=LogNorm(vmin=1e-6, vmax=1e-3),
        )
        ax2.fill_between(data["dt"], 0, data["elev"], color="black", zorder=2)
        ax2.plot(data["dt"], data["elev"], color="black", linewidth=1.5, label="Surface Elevation")
        ax2.plot(data["dt"], data["tropo"], color="black", linestyle="--", linewidth=1.5, label="Tropopause Height")

        # EBD SPH (strongest layer)
        if data.get("sph_ext_time") is not None and data.get("sph_ext_km") is not None:
            ax2.plot(data["sph_ext_time"], data["sph_ext_km"], color="black", marker="o", markersize=3, linestyle="None", alpha=0.9, label="sph_ext")# linestyle="-", linewidth=1.5, label="sph_ext"),  
        if data.get("sph_top_time") is not None and data.get("sph_top_km") is not None:
            ax2.plot(data["sph_top_time"], data["sph_top_km"], color="#FF0000", marker="o", markersize=3, linestyle="None", alpha=0.9, label="sph_top")# linestyle="-", linewidth=1.5, label="sph_top")
        if data.get("sph_bottom_time") is not None and data.get("sph_bottom_km") is not None:
            ax2.plot(data["sph_bottom_time"], data["sph_bottom_km"], color="green", marker="o", markersize=3, linestyle="None", alpha=0.9, label="sph_bottom")#, linestyle="-", linewidth=1.5, label="sph_bottom")

        # EBD multilayer
        if cfg.plot_sph_ext and data.get("sph_ext_layers_ebd") is not None:
            t = data["ebd_dt"]
            arr = np.asarray(data["sph_ext_layers_ebd"])
            for j in range(arr.shape[1]):
                ax2.plot(t, arr[:, j], color="black",  marker="o", markersize=3, linestyle="None", alpha=0.9)#  linestyle="-", linewidth=1.0, alpha=0.9)
        if cfg.plot_sph_top and data.get("sph_top_layers_ebd") is not None:
            t = data["ebd_dt"]
            arr = np.asarray(data["sph_top_layers_ebd"])
            for j in range(arr.shape[1]):
                ax2.plot(t, arr[:, j], color="#FF0000",  marker="o", markersize=3, linestyle="None", alpha=0.9)#  linestyle="-", linewidth=1.0, alpha=0.9)
        if cfg.plot_sph_bottom and data.get("sph_bottom_layers_ebd") is not None:
            t = data["ebd_dt"]
            arr = np.asarray(data["sph_bottom_layers_ebd"])
            for j in range(arr.shape[1]):
                ax2.plot(t, arr[:, j], color="green",  marker="o", markersize=3, linestyle="None", alpha=0.9)# , linestyle="-", linewidth=1.0, alpha=0.9)

        # ATL_ALD heights (optional)
        if cfg.plot_atl_ald_heights:
            ald_time = data.get("ald_time")
            ald_top = data.get("ald_top_height_km")
            ald_bottom = data.get("ald_bottom_height_km")
            ald_layer_number = data.get("ald_layer_number")

            if ald_time is not None and ald_layer_number is not None and ald_top is not None and ald_bottom is not None:
                ald_time_np = np.asarray(ald_time.to_numpy() if hasattr(ald_time, "to_numpy") else ald_time)
                n_layers = ald_top.shape[1]
                for j in range(n_layers):
                    ax2.plot(ald_time_np, ald_top[:, j], linestyle="-", linewidth=3, color="magenta", label="ATL_ALD top" if j == 0 else None, zorder=10)
                    ax2.plot(ald_time_np, ald_bottom[:, j], linestyle="--", linewidth=3, color="darkblue", label="ATL_ALD bottom" if j == 0 else None, zorder=10)

        ax2.legend(loc="upper right", fontsize=18)
        ax2.set_ylabel("Altitude [km]")
        ax2.set_ylim(0, 20)

        pos2 = ax2.get_position()
        cax2 = fig.add_axes([pos2.x1 + 0.01, pos2.y0, 0.015, pos2.height])
        cbar2 = plt.colorbar(im_ebd, cax=cax2)
        cbar2.set_label(r"[m$^{-1}$]" + "\n \n \n \n" + r"$\mathbf{Extinction}$ " + "\n" + r" $\mathbf{Coefficient}$", labelpad=10, fontsize=24)
        cbar2.ax.tick_params(labelsize=19)
    else:
        print(f"ebc_ext is: {data.get("ebd_ext")}")
        print(f"ebc_height_km is: {data.get("ebd_height_km")}")
        print(f"ebc_dt is: {data.get("ebd_dt")}")
        ax2.text(0.5, 0.5, "No EBD available", transform=ax2.transAxes, ha="center", va="center", fontsize=18)
        ax2.set_ylim(0, 20)
        ax2.set_ylabel("Altitude [km]")

    # --- ATL_TC classification ---
    ax3 = fig.add_subplot(gs[32:42, :], sharex=ax1)

    if cfg.reduce_classifications:
        TC_CLASSES = {
            -1: ("Missing data", "#c4c4c4"),
            -2: ("Surface or sub-surface", "#000000"),
            0: ("Clear", "#ffffff"),
            1: ("Clouds", "#5555fe"),
            4: ("Aerosols", "#a22f2e"),
        }
    else:
        TC_CLASSES = {
            -3: ("Missing data", "#f5f5dc"),
            -2: ("Surface or sub-surface", "#000000"),
            -1: ("Noise in Mie & Ray channels", "#c4c4c4"),
            0: ("Clear", "#ffffff"),
            1: ("(Warm) Liquid cloud", "#5555fe"),
            2: ("(Supercooled) Liquid cloud", "#0018e0"),
            3: ("Ice cloud", "#009bce"),
            10: ("Dust", "#a22f2e"),
            11: ("Sea salt", "#acd6e2"),
            12: ("Continental pollution", "#02fd7f"),
            13: ("Smoke", "#2f504e"),
            14: ("Dusty smoke", "#996633"),
            15: ("Dusty mix", "#e0b98b"),
            20: ("STS", "#ff00ff"),
            21: ("NAT", "#8f37df"),
            22: ("Stratospheric ice", "#4a0061"),
            25: ("Stratospheric ash", "#fece05"),
            26: ("Stratospheric sulfate", "#fdfb0d"),
            27: ("Stratospheric smoke", "#c3b27f"),
        }

    tc_class_for_plot = data["class_data"]   # already smoke-masked upstream when only_smoke_for_tc_sph=True

    tc_remapped, keep_classes, class_cmap, class_norm = _tc_plot_classes_and_remap(
        tc_class_for_plot,
        tc_classes=TC_CLASSES,
        only_smoke=bool(cfg.only_smoke_for_tc_sph),
        smoke_ids=tuple(cfg.tc_smoke_class_ids),
        include_clear=True,
        include_surface=True,
        include_missing=False,
    )

    im_ec = ax3.pcolormesh(
        data["dt_ext"],
        data["height"],
        tc_remapped,
        cmap=class_cmap,
        norm=class_norm,
        shading="auto",
    )

    ax3.plot(data["dt"], data["elev"], color="black", linewidth=1.5, label="Surface Elevation")
    ax3.plot(data["dt"], data["tropo"], color="black", linestyle="--", linewidth=1.5, label="Tropopause Height")

    if data.get("sph_tc_time") is not None and data.get("sph_tc_km") is not None:
        ax3.plot(data["sph_tc_time"], data["sph_tc_km"], color="black", linestyle="--", linewidth=1.5, label="sph_ext (ATL_TC)")

    ax3.set_ylabel("Altitude [km]")
    ax3.set_ylim(0, 20)
    ax3.legend(loc="upper right", fontsize=18)
    ax3.tick_params(labelbottom=True)

    for ax in (ax1, ax2, ax3):
        ax.label_outer()
    ax3.set_xlabel("Time [UTC / LT]", fontsize=18, labelpad=8, fontweight="bold")

    pos3 = ax3.get_position()
    cax3 = fig.add_axes([pos3.x1 + 0.01, pos3.y0, 0.015, pos3.height])

    cbar3 = plt.colorbar(im_ec, cax=cax3, spacing="uniform")
    cbar3.set_ticks(np.arange(len(keep_classes)))
    cbar3.set_ticklabels([TC_CLASSES[cid][0] for cid in keep_classes])


    cbar3.ax.tick_params(labelsize=14, pad=2)
    for lbl in cbar3.ax.get_yticklabels():
        lbl.set_verticalalignment("center")
    cbar3.set_label(r"$\mathbf{Target}$ " + "\n" + r" $\mathbf{classification}$", fontweight="bold", fontsize=24)

    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Matching / driver
# -----------------------------
def _normalize_day_string(d: str) -> Optional[str]:
    s = str(d).strip().replace("-", "")
    return s if len(s) == 8 and s.isdigit() else None


def match_and_plot_flexpart_earthcare(
    *,
    flexpart_folder: str,
    atl_tc_folder: str,
    atl_ebd_folder: Optional[str],
    output_dir: str,
    atl_ald_folder: Optional[str] = None,
    gfas_folder: Optional[str] = None,
    gfas_variant: str = "01",
    config: PlotConfig = PlotConfig(),
) -> None:
    def _days_in_flexpart_file(nc_path: str):
        try:
            ds = xr.open_dataset(nc_path, decode_timedelta=True)
            if ("time" in ds.coords) or ("time" in ds.dims):
                t = pd.to_datetime(ds["time"].values)
                days = sorted(set(pd.to_datetime(t).strftime("%Y%m%d")))
                ds.close()
                return days
            ds.close()
        except Exception:
            pass
        return []

    def _in_filters(day_yyyymmdd: str) -> bool:
        # ONLY_DATES
        if config.only_dates:
            only = set(filter(None, (_normalize_day_string(x) for x in config.only_dates)))
            if only and day_yyyymmdd not in only:
                return False

        # DATE_RANGE
        if config.date_range and len(config.date_range) == 2:
            start_day = _normalize_day_string(config.date_range[0])
            end_day = _normalize_day_string(config.date_range[1])
            if start_day and end_day and not (start_day <= day_yyyymmdd <= end_day):
                return False

        return True

    def index_earthcare(folder: Optional[str]):
        ts_to_file = {}
        date_to_ts = defaultdict(list)
        if not folder:
            return ts_to_file, date_to_ts

        pat_ec_full = re.compile(r"(?P<start>\d{8}T\d{6}Z)_\d{8}T\d{6}Z")
        files = glob.glob(os.path.join(folder, "*.h5"))
        print(f"Indexing {len(files)} files in {folder} by start timestamp and date...")
        for f in files:
            m = pat_ec_full.search(os.path.basename(f))
            if not m:
                continue
            ts = m.group("start")
            if ts not in ts_to_file:
                ts_to_file[ts] = f
                date_to_ts[ts[:8]].append(ts)
        for d in date_to_ts:
            date_to_ts[d].sort()
        return ts_to_file, date_to_ts

    os.makedirs(output_dir, exist_ok=True)

    tc_ts_to_file, tc_date_to_ts = index_earthcare(atl_tc_folder)
    ebd_ts_to_file, _ = index_earthcare(atl_ebd_folder) if atl_ebd_folder else ({}, defaultdict(list))
    ald_ts_to_file, _ = index_earthcare(atl_ald_folder) if atl_ald_folder else ({}, defaultdict(list))

    # FLEXPART files
    if os.path.isfile(flexpart_folder):
        flexpart_files = [flexpart_folder]
    else:
        all_files = set(glob.glob(os.path.join(flexpart_folder, "grid_conc_*.nc")))
        nest_files = set(glob.glob(os.path.join(flexpart_folder, "grid_conc_*_nest.nc")))
        flexpart_files = sorted(all_files - nest_files)
        # preserve your “2025 only” filter
        flexpart_files = [f for f in flexpart_files if "2025" in os.path.basename(f)]

    print(f"Found {len(flexpart_files)} FLEXPART NetCDF file(s) in {flexpart_folder}")

    pat_flex = re.compile(r"grid_conc_(\d{8})\d{6}\.nc$")

    for flex_file in flexpart_files:
        fname = os.path.basename(flex_file)
        days_from_file = _days_in_flexpart_file(flex_file)

        if days_from_file:
            days_iter = [d for d in days_from_file if _in_filters(d)]
        else:
            m = pat_flex.search(fname)
            if not m:
                print(f"Skipping FLEXPART file with no recognizable date: {fname}")
                continue
            day = m.group(1)
            if not _in_filters(day):
                print(f"Skipping FLEXPART {fname} outside filters")
                continue
            days_iter = [day]

        if not days_iter:
            print(f"No matching days (after filters) in {fname}")
            continue

        for day in days_iter:
            tc_timestamps = tc_date_to_ts.get(day, [])
            if not tc_timestamps:
                continue

            for ts in tc_timestamps:
                tc_file = tc_ts_to_file.get(ts)
                if not tc_file:
                    continue

                ebd_file = ebd_ts_to_file.get(ts) if atl_ebd_folder else None
                ald_file = ald_ts_to_file.get(ts) if atl_ald_folder else None

                out_name = re.sub(
                    r".*_(\d{8})T(\d{6})Z_(\d{8})T(\d{6})Z_([A-Za-z0-9]+).*",
                    r"EC_FLEXPART_GFAS_\1_T\2Z_T\4Z_\5.png",
                    os.path.basename(tc_file),
                )
                out_png = os.path.join(output_dir, out_name)
                if (not config.force_replot) and os.path.exists(out_png):
                    print(f"Already plotted {day} for {os.path.basename(tc_file)} — skipping")
                    continue

                print(f"   {os.path.basename(tc_file)} matched with {fname}")
                try:
                    data = prepare_data_flexpart_earthcare_gfas(
                        flexpart_nc=flex_file,
                        earthcare_file=tc_file,
                        ebd_file=ebd_file,
                        ald_file=ald_file,
                        gfas_folder=gfas_folder,
                        gfas_variant=gfas_variant,
                        config=config,
                    )

                    if data.get("skipped", False):
                        continue

                    dfm = data.get("gfas_fire_matches")
                    if dfm is not None and isinstance(dfm, pd.DataFrame) and not dfm.empty:
                        out_csv = os.path.join(
                            output_dir,
                            os.path.basename(out_png).replace(".png", "_GFAS_EC_height_comparison.csv")
                        )
                        dfm.to_csv(out_csv, index=False)
                        print(f"      Saved height comparison table: {os.path.basename(out_csv)}")


                    plot_flexpart_earthcare_from_data(data, output_dir)
                except Exception as e:
                    print(f"Failed for {os.path.basename(tc_file)} with {fname}: {e}")


# -----------------------------
# User script section
# -----------------------------
flexpart_folder = "/xnilu_wrk/users/ne/FORWARD_RUNS/BC_2025/OUT_BB_irene/"
plot_dir_ebd_flex_earthcare = "/xnilu_wrk2/projects/NEVAR/Irene/FLEXPART_EarthCARE_GFAS_BA"
ATL_TC_folder = "/xnilu_wrk2/projects/NEVAR/Irene/data/ATL_TC_20250510_20250620"
ATL_EBD_folder = "/xnilu_wrk2/projects/NEVAR/Irene/data/ATL_EBD_20250510_20250620"
ATL_ALD_folder = "/xnilu_wrk2/projects/NEVAR/Irene/data/ATL_ALD_20250510_20250620"
GFAS_DIR = "/xnilu_wrk/flex_wrk/ECMWF_DATA/GFAS"


# Equivalent of your old globals
config = PlotConfig(
    only_dates=None,
    # date_range=("2025-05-25", "2025-06-15"),
    date_range=("2025-06-01", "2025-06-15"),
    force_replot=True,
    quality_flags=(0, 1),
    resolution="low",
    sph_smooth_sigma=None,
    max_layers=4,

    sph_threshold_ext=1e-6,     
    sph_threshold_flex=SPH_THRESHOLD,

    # GFAS proximity filters (both only_overpasses_near_gfas and use_canada_only needs to be true!!)
    only_overpasses_near_gfas=False,
    use_canada_only=False,
    gfas_max_distance_km=100.0,
    gfas_min_frp=1,                                       # <-- rerun with this!!!, previously set to 0.1
    canada_bbox=dict(lat_min=41.6, lat_max=83.1, lon_min=-141.0, lon_max=-52.6),

    plot_atl_ald_heights=False,
    plot_sph_ext=True,
    plot_sph_top=True,
    plot_sph_bottom=True,

    plot_only_TC_and_extinction_alone=False,

    # plot only smoke classes in TC
    only_smoke_for_tc_sph=False,
    tc_smoke_class_ids=(13, 14, 27),

    reduce_classifications=False,
    cloud_pad_time=1,
    cloud_pad_z=1,

    remove_pixel_groups_below=10,
    remove_pixel_groups_connectivity=2,

    filter_out_ebd_values_above=None,
    no_clouds_in_flexpart=False,
    set_flexpart_res_to_original=True,
)


if config.no_clouds_in_flexpart:
    plot_dir_ebd_flex_earthcare = "/xnilu_wrk2/projects/NEVAR/Irene/FLEXPART_EarthCARE_GFAS_no_clouds_in_flexpart"

if config.only_smoke_for_tc_sph and config.no_clouds_in_flexpart == False:
    plot_dir_ebd_flex_earthcare = "/xnilu_wrk2/projects/NEVAR/Irene/FLEXPART_EarthCARE_only_smoke_TC"
    os.makedirs(plot_dir_ebd_flex_earthcare, exist_ok=True)

if config.plot_only_TC_and_extinction_alone:
    plot_dir_ebd_flex_earthcare = "/xnilu_wrk2/projects/NEVAR/Irene/FLEXPART_EarthCARE_only_TC_and_extinction"

if config.only_overpasses_near_gfas:
    plot_dir_ebd_flex_earthcare += f"_GFASprox_{int(config.gfas_max_distance_km)}km"

if config.reduce_classifications and config.plot_only_TC_and_extinction_alone == False and config.only_overpasses_near_gfas == False and config.only_smoke_for_tc_sph == False:
    plot_dir_ebd_flex_earthcare = "/xnilu_wrk2/projects/NEVAR/Irene/FLEXPART_EarthCARE_GFAS_few_classifications_no_cloud/2_layers"
    os.makedirs(plot_dir_ebd_flex_earthcare, exist_ok=True)

if __name__ == "__main__":
    match_and_plot_flexpart_earthcare(
        flexpart_folder=flexpart_folder,
        atl_tc_folder=ATL_TC_folder,
        atl_ebd_folder=ATL_EBD_folder,
        output_dir=plot_dir_ebd_flex_earthcare,
        atl_ald_folder=ATL_ALD_folder,
        gfas_folder=GFAS_DIR,
        gfas_variant="01",
        config=config,
    )