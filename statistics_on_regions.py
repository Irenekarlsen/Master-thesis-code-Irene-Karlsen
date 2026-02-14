from __future__ import annotations
import os
import re
import glob
import json
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import subplot_with_plume_heights_threshold as ec  # provides prepare_data_flexpart_earthcare_gfas

# ============================================================
# USER SETTINGS
# ============================================================

# Region boxes (lon_min, lon_max, lat_min, lat_max)
CANADA_BOX   = (-140.0, -55.0, 40.0, 65.0)
ATLANTIC_BOX = ( -55.0, -10.0, 40.0, 65.0)
EUROPE_BOX   = ( -10.0,  40.0, 40.0, 65.0)
POLAR_BOX    = (-140.0,  40.0, 65.0, 85.0)

START_DATE = "2025-05-25"
END_DATE   = "2025-06-15"

REGION_CSVS = {
    "CANADA":   "/xnilu_wrk2/projects/NEVAR/Irene/data/region_lists_20250510_20250620/CANADA_20250510_20250620.csv",
    "ATLANTIC": "/xnilu_wrk2/projects/NEVAR/Irene/data/region_lists_20250510_20250620/ATLANTIC_20250510_20250620.csv",
    "EUROPE":   "/xnilu_wrk2/projects/NEVAR/Irene/data/region_lists_20250510_20250620/EUROPE_20250510_20250620.csv",
    "POLAR":    "/xnilu_wrk2/projects/NEVAR/Irene/data/region_lists_20250510_20250620/POLAR_20250510_20250620.csv",
} 

FLEXPART_NC    = "/xnilu_wrk/users/ne/FORWARD_RUNS/BC_2025/OUT_BB_irene/grid_conc_20250101000000.nc"
ATL_TC_FOLDER  = "/xnilu_wrk2/projects/NEVAR/Irene/data/ATL_TC_20250510_20250620"
ATL_EBD_FOLDER = "/xnilu_wrk2/projects/NEVAR/Irene/data/ATL_EBD_20250510_20250620"
ATL_ALD_FOLDER = "/xnilu_wrk2/projects/NEVAR/Irene/data/ATL_ALD_20250510_20250620"
GFAS_DIR       = "/xnilu_wrk/flex_wrk/ECMWF_DATA/GFAS"

# Which altitude definition to use:
COMPARE_MODE = "ext"  # "ext", "top", or "bottom"
only_smoke_for_tc_sph = True 
no_clouds_in_flexpart = True 


COMPARE_AGL = True
USE_MULTILAYERS = True

# Plot switches
MAKE_DAILY_PLOTS   = False
MAKE_PERIOD_PLOTS  = True
MAKE_OVERVIEW_PLOT = True
MAKE_PERIOD_MAPS   = True

# Pairing settings (for scatter later): nearest EC-at-FP-time match must be within this tolerance
PAIR_MAX_ABS_DT_SECONDS = 5  # [seconds]

# Optional: only process these track IDs
DEBUG_TRACKS: Optional[List[str]] = None #["05730C"] #["05740D"] #["05730C", "05732B", "05731C", "05731B", "05740D", "05732C", "05753C", "05740D"]

# True means isolate outputs + prefix filenames
DEBUG_TRACK = DEBUG_TRACKS is not None and len(DEBUG_TRACKS) > 0

# Output root (will create subfolders: flexpart/, earthcare/, combined/, files/)
OUTDIR = "/xnilu_wrk2/projects/NEVAR/Irene/region_stats"

if only_smoke_for_tc_sph:
    OUTDIR = "/xnilu_wrk2/projects/NEVAR/Irene/region_stats/smoke_only_for_tc_sph"

if no_clouds_in_flexpart:
    OUTDIR = "/xnilu_wrk2/projects/NEVAR/Irene/region_stats/no_clouds_in_flexpart"

# ============================================================
# INTERNAL CONSTANTS / HELPERS
# ============================================================

REGION_BOXES = {
    "CANADA": CANADA_BOX,
    "ATLANTIC": ATLANTIC_BOX,
    "EUROPE": EUROPE_BOX,
    "POLAR": POLAR_BOX,
}

EC_NAME_PAT = re.compile(r".*_(\d{8})T(\d{6})Z_(\d{8})T(\d{6})Z_([A-Za-z0-9]+)\.h5$")


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def make_run_tag(debug: bool, debug_tracks: Optional[List[str]]) -> str:
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    if not debug:
        return ts
    if debug_tracks:
        tstr = "-".join(debug_tracks[:4])
        return f"debug_track_{tstr}_{ts}"
    return f"debug_track_{ts}"


def outpath(base_dir: str, fname: str, prefix: str = "") -> str:
    if prefix and not fname.startswith(prefix):
        fname = prefix + fname
    return os.path.join(base_dir, fname)


def wrap_lon(lon):
    lon = np.asarray(lon, dtype=float)
    return ((lon + 180.0) % 360.0) - 180.0


def in_box_mask(lon, lat, box):
    lon_min, lon_max, lat_min, lat_max = box
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)

    m = np.isfinite(lon) & np.isfinite(lat)
    out = np.zeros_like(lon, dtype=bool)
    if not np.any(m):
        return out

    lat_ok = (lat[m] >= lat_min) & (lat[m] <= lat_max)

    if lon_min <= lon_max:
        lon_ok = (lon[m] >= lon_min) & (lon[m] <= lon_max)
    else:
        lon_ok = (lon[m] >= lon_min) | (lon[m] <= lon_max)

    out[m] = lon_ok & lat_ok
    return out


def index_earthcare_by_track(tc_folder: str) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for f in glob.glob(os.path.join(tc_folder, "*.h5")):
        base = os.path.basename(f)
        m = EC_NAME_PAT.match(base)
        if not m:
            continue
        start_ymd, start_hms = m.group(1), m.group(2)
        track_id = m.group(5)
        start_dt = pd.to_datetime(start_ymd + start_hms, format="%Y%m%d%H%M%S")
        out.setdefault(track_id, []).append({
            "tc_file": f,
            "start_dt": start_dt,
            "day_ymd": start_ymd,
            "ts_start": f"{start_ymd}T{start_hms}Z",
            "track_id": track_id,
        })
    for k in out:
        out[k] = sorted(out[k], key=lambda r: r["start_dt"])
    return out


def index_earthcare_by_ts(folder: Optional[str]) -> Dict[str, str]:
    if not folder:
        return {}
    ts_map: Dict[str, str] = {}
    pat = re.compile(r"(?P<start>\d{8}T\d{6}Z)_\d{8}T\d{6}Z")
    for f in glob.glob(os.path.join(folder, "*.h5")):
        m = pat.search(os.path.basename(f))
        if m:
            ts_map[m.group("start")] = f
    return ts_map


def _try_get_cartopy():
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        return ccrs, cfeature
    except Exception:
        return None, None


def break_line_on_dateline(lon_wrapped, lat, jump_deg=180.0):
    lon_wrapped = np.asarray(lon_wrapped, dtype=float)
    lat = np.asarray(lat, dtype=float)

    m = np.isfinite(lon_wrapped) & np.isfinite(lat)
    lon = lon_wrapped[m]
    la = lat[m]

    if lon.size < 2:
        return lon, la

    dlon = np.abs(np.diff(lon))
    breaks = np.where(dlon > jump_deg)[0]

    if breaks.size == 0:
        return lon, la

    lon_out = []
    lat_out = []
    start = 0
    for i in breaks:
        lon_out.append(lon[start:i+1])
        lat_out.append(la[start:i+1])
        lon_out.append(np.array([np.nan]))
        lat_out.append(np.array([np.nan]))
        start = i + 1

    lon_out.append(lon[start:])
    lat_out.append(la[start:])

    return np.concatenate(lon_out), np.concatenate(lat_out)


def plot_track_with_wrap_copies(ax, lon, lat, plot_kwargs, extent=(-180, 180, 0, 90), jump_deg=180.0):
    lon_min, lon_max, lat_min, lat_max = extent

    lon0 = wrap_lon(np.asarray(lon, dtype=float))
    lat0 = np.asarray(lat, dtype=float)

    m = np.isfinite(lon0) & np.isfinite(lat0) & (lat0 >= lat_min) & (lat0 <= lat_max)
    if m.sum() < 2:
        return

    lon0 = lon0[m]
    lat0 = lat0[m]

    for shift in (-360.0, 0.0, 360.0):
        lon_s = lon0 + shift
        pad = 5.0
        mm = (lon_s >= lon_min - pad) & (lon_s <= lon_max + pad)
        if mm.sum() < 2:
            continue

        lon_use = lon_s[mm]
        lat_use = lat0[mm]
        lon_plot, lat_plot = break_line_on_dateline(lon_use, lat_use, jump_deg=jump_deg)
        ax.plot(lon_plot, lat_plot, **plot_kwargs)


def plot_day_region_map_only(
    out_png: str,
    region_name: str,
    box: Tuple[float, float, float, float],
    tracks: List[dict],
    day: pd.Timestamp,
    period_start: Optional[pd.Timestamp] = None,
    period_end: Optional[pd.Timestamp] = None,
):
    lon_min, lon_max, lat_min, lat_max = box
    pad_lon = max(5.0, 0.15 * (lon_max - lon_min))
    pad_lat = max(5.0, 0.15 * (lat_max - lat_min))
    extent = [lon_min - pad_lon, lon_max + pad_lon, lat_min - pad_lat, lat_max + pad_lat]

    if period_start is not None and period_end is not None:
        t0 = pd.to_datetime(period_start)
        t1 = pd.to_datetime(period_end)
    else:
        day0 = pd.to_datetime(day).normalize()
        t0 = day0
        t1 = day0 + pd.Timedelta(days=1)

    def _fmt(ts: pd.Timestamp) -> str:
        return ts.strftime("%Y-%m-%d %H:%M")

    period_str = f"{_fmt(t0)} → {_fmt(t1)}"

    ccrs, cfeature = _try_get_cartopy()
    fig = plt.figure(figsize=(10, 8))

    if ccrs is not None:
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.4)

        gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4)
        gl.top_labels = False
        gl.right_labels = False
        gl.x_inline = False
        gl.y_inline = False

        plot_kwargs = dict(transform=ccrs.PlateCarree())
    else:
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.grid(True, linewidth=0.4, alpha=0.4)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.tick_params(top=False, labeltop=False, right=False, labelright=False)
        plot_kwargs = {}

    rect_lons = [lon_min, lon_max, lon_max, lon_min, lon_min]
    rect_lats = [lat_min, lat_min, lat_max, lat_max, lat_min]
    ax.plot(rect_lons, rect_lats, linewidth=2.5, **plot_kwargs)

    for tr in tracks:
        if ccrs is not None:
            ax.plot(
                tr["lon_raw"],
                tr["lat"],
                transform=ccrs.Geodetic(),
                linewidth=0.9,
                alpha=0.6,
            )
        else:
            lon = wrap_lon(tr["lon_raw"])
            m = np.isfinite(lon) & np.isfinite(tr["lat"])
            lon_plot, lat_plot = break_line_on_dateline(lon[m], np.asarray(tr["lat"])[m], jump_deg=180.0)
            ax.plot(lon_plot, lat_plot, linewidth=0.9, alpha=0.6)

    ax.set_title(
        f"{region_name}\n"
        f"{period_str}\n"
        f"Tracks intersecting box: {len(tracks)}"
    )

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)


def _box_to_lonlat_poly(box):
    lon_min, lon_max, lat_min, lat_max = box
    lons = [lon_min, lon_max, lon_max, lon_min, lon_min]
    lats = [lat_min, lat_min, lat_max, lat_max, lat_min]
    return np.asarray(lons, float), np.asarray(lats, float)


def plot_nh_rectangular_tracks_and_boxes(
    out_png: str,
    tracks: list[dict],
    region_boxes: dict[str, tuple[float, float, float, float]],
    title: str,
    lat_min: float = 40.0,
    lon_min: float = -180.0,
    lon_max: float = 180.0,
):
    ccrs, cfeature = _try_get_cartopy()
    fig = plt.figure(figsize=(12, 9))

    if ccrs is not None:
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.set_extent([lon_min, lon_max, lat_min, 90], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.4)

        gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4)
        gl.top_labels = False
        gl.right_labels = False
        gl.x_inline = False
        gl.y_inline = False

        plot_kwargs = dict(transform=ccrs.PlateCarree())
        text_kwargs = dict(transform=ccrs.PlateCarree())
    else:
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, 90)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, linewidth=0.3, alpha=0.4)
        plot_kwargs = {}
        text_kwargs = {}

    for tr in tracks:
        lon_raw = tr.get("lon_raw", None)
        lat = tr.get("lat", None)
        if lon_raw is None or lat is None:
            continue

        lon_raw = np.asarray(lon_raw, dtype=float)
        lat = np.asarray(lat, dtype=float)

        if ccrs is not None:
            m = np.isfinite(lon_raw) & np.isfinite(lat) & (lat >= lat_min) & (lat <= 90)
            if m.sum() < 2:
                continue
            ax.plot(lon_raw[m], lat[m], transform=ccrs.Geodetic(), linewidth=0.9, alpha=0.6)
        else:
            lon = wrap_lon(lon_raw)
            m = np.isfinite(lon) & np.isfinite(lat) & (lat >= lat_min) & (lat <= 90)
            if m.sum() < 2:
                continue
            lon_plot, lat_plot = break_line_on_dateline(lon[m], lat[m], jump_deg=180.0)
            ax.plot(lon_plot, lat_plot, linewidth=0.9, alpha=0.6)

    for name, box in region_boxes.items():
        lons, lats = _box_to_lonlat_poly(box)
        ax.plot(lons, lats, linewidth=2.6, alpha=0.95, **plot_kwargs)

        lon0, lon1, la0, la1 = box
        x = 0.5 * (lon0 + lon1)
        y = 0.5 * (la0 + la1)

        ax.text(
            x, y, name,
            fontsize=10,
            ha="center", va="center",
            fontweight="bold",
            **text_kwargs
        )

    ax.set_title(f"{title}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def summarize_alts_basic(x: np.ndarray) -> dict:
    """
    Minimal summary: mean/median/std/p25/p75/p90 + n.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(n=0, mean=np.nan, median=np.nan, std=np.nan, p25=np.nan, p75=np.nan, p90=np.nan)
    return dict(
        n=int(x.size),
        mean=float(np.nanmean(x)),
        median=float(np.nanmedian(x)),
        std=float(np.nanstd(x, ddof=1)) if x.size >= 2 else np.nan,
        p25=float(np.nanpercentile(x, 25)),
        p75=float(np.nanpercentile(x, 75)),
        p90=float(np.nanpercentile(x, 90)),
    )



def get_native_series(data: dict, mode: str):
    t_ec = None
    t_fp = None
    y_ec = None
    y_fp = None

    # Normalize mode so accidental values don't break logic
    mode = (mode or "").strip().lower()
    if mode.startswith("sph_"):
        mode = mode.replace("sph_", "", 1)

    if mode == "ext":
        t_ec = data.get("sph_ext_time", None)
        y_ec = data.get("sph_ext_km_multi", data.get("sph_ext_km", None))

        t_fp = data.get("sph_flex_time", None)
        y_fp = data.get("sph_flex_ext_km_multi", data.get("sph_flex_ext_km", None))

    elif mode == "top":
        t_ec = data.get("sph_top_time", None)
        y_ec = data.get("sph_top_km_multi", data.get("sph_top_km", None))

        t_fp = data.get("sph_flex_time", None)
        y_fp = data.get("sph_flex_top_km_multi", data.get("sph_flex_top_km", None))

    elif mode == "bottom":
        t_ec = data.get("sph_bottom_time", None)
        y_ec = data.get("sph_bottom_km_multi", data.get("sph_bottom_km", None))

        t_fp = data.get("sph_flex_time", None)
        y_fp = data.get("sph_flex_bottom_km_multi", data.get("sph_flex_bottom_km", None))

    else:
        raise ValueError(f"mode must be one of: ext, top, bottom (got {mode!r})")

    # Convert times
    if t_ec is not None:
        t_ec = np.asarray(t_ec).astype("datetime64[ns]")
    if t_fp is not None:
        t_fp = np.asarray(t_fp).astype("datetime64[ns]")

    # Convert y
    y_ec = np.asarray(y_ec, dtype=float) if y_ec is not None else None
    y_fp = np.asarray(y_fp, dtype=float) if y_fp is not None else None

    return (t_ec, y_ec), (t_fp, y_fp)





def nearest_index_by_time(src_t: np.ndarray, q_t: np.ndarray) -> np.ndarray:
    """
    For each query time q_t, return index of nearest src_t.
    src_t and q_t must be datetime64[ns].
    """
    src_t = np.asarray(src_t).astype("datetime64[ns]")
    q_t = np.asarray(q_t).astype("datetime64[ns]")

    if src_t.size == 0 or q_t.size == 0:
        return np.full(q_t.shape, -1, dtype=int)

    order = np.argsort(src_t)
    src_t_sorted = src_t[order]
    src_ns = src_t_sorted.astype("int64")
    q_ns = q_t.astype("int64")

    pos = np.searchsorted(src_ns, q_ns, side="left")
    pos0 = np.clip(pos - 1, 0, src_ns.size - 1)
    pos1 = np.clip(pos,     0, src_ns.size - 1)

    d0 = np.abs(src_ns[pos0] - q_ns)
    d1 = np.abs(src_ns[pos1] - q_ns)
    use1 = d1 < d0
    best_pos = np.where(use1, pos1, pos0)

    # map back to original indices
    return order[best_pos]


def nearest_values_by_time(src_t: np.ndarray, src_y: np.ndarray, q_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each q_t, return nearest src_y and abs dt seconds.
    Filters NaNs in src_y.
    """
    src_t = np.asarray(src_t).astype("datetime64[ns]")
    q_t = np.asarray(q_t).astype("datetime64[ns]")
    src_y = np.asarray(src_y, dtype=float)

    m = np.isfinite(src_y)
    src_t = src_t[m]
    src_y = src_y[m]
    if src_t.size == 0 or q_t.size == 0:
        return np.full(q_t.shape, np.nan), np.full(q_t.shape, np.nan)

    order = np.argsort(src_t)
    src_t = src_t[order]
    src_y = src_y[order]

    src_ns = src_t.astype("int64")
    q_ns = q_t.astype("int64")

    pos = np.searchsorted(src_ns, q_ns, side="left")
    pos0 = np.clip(pos - 1, 0, src_ns.size - 1)
    pos1 = np.clip(pos,     0, src_ns.size - 1)

    d0 = np.abs(src_ns[pos0] - q_ns)
    d1 = np.abs(src_ns[pos1] - q_ns)
    use1 = d1 < d0
    best = np.where(use1, pos1, pos0)

    y = src_y[best]
    dt_sec = np.minimum(d0, d1) / 1e9
    return y, dt_sec


# ============================================================
# MAIN
# ============================================================

def main():
    run_tag = make_run_tag(DEBUG_TRACK, DEBUG_TRACKS)

    if DEBUG_TRACK:
        OUTROOT = os.path.join(OUTDIR, "debug_track", run_tag)
        file_prefix = "debug_track_"
    else:
        OUTROOT = OUTDIR
        file_prefix = ""

    out_flex  = os.path.join(OUTROOT, "flexpart")
    out_ec    = os.path.join(OUTROOT, "earthcare")
    out_comb  = os.path.join(OUTROOT, "combined")
    out_files = os.path.join(OUTROOT, "files")

    for p in [OUTROOT, out_flex, out_ec, out_comb, out_files]:
        ensure_dir(p)

    plots_flex = os.path.join(out_flex, "plots")
    plots_ec   = os.path.join(out_ec, "plots")
    plots_comb = os.path.join(out_comb, "plots")
    for p in [plots_flex, plots_ec, plots_comb]:
        ensure_dir(p)

    maps_flex = os.path.join(plots_flex, "daily_maps")
    maps_ec   = os.path.join(plots_ec, "daily_maps")
    maps_comb = os.path.join(plots_comb, "daily_maps")
    for p in [maps_flex, maps_ec, maps_comb]:
        ensure_dir(p)

    start_dt = pd.to_datetime(START_DATE)
    end_dt   = pd.to_datetime(END_DATE)

    manifest = {
        "START_DATE": START_DATE,
        "END_DATE": END_DATE,
        "COMPARE_MODE": COMPARE_MODE,
        "USE_MULTILAYERS": USE_MULTILAYERS,
        "REGION_CSVS": REGION_CSVS,
        "FLEXPART_NC": FLEXPART_NC,
        "ATL_TC_FOLDER": ATL_TC_FOLDER,
        "ATL_EBD_FOLDER": ATL_EBD_FOLDER,
        "ATL_ALD_FOLDER": ATL_ALD_FOLDER,
        "GFAS_DIR": GFAS_DIR,
        "DEBUG_TRACKS": DEBUG_TRACKS,
        "REGION_BOXES": REGION_BOXES,
        "DEBUG_TRACK": DEBUG_TRACK,
        "OUTROOT": OUTROOT,
        "RUN_TAG": run_tag,
        "PAIR_MAX_ABS_DT_SECONDS": PAIR_MAX_ABS_DT_SECONDS,
        "NOTE": "No interpolation. EC in-box uses native SPH times mapped to nearest track lat/lon by time. FP in-box uses native FP lat/lon.",
    }
    with open(outpath(out_files, "run_manifest.json", file_prefix), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    region_tracks = {name: set() for name in REGION_CSVS}
    for name, path in REGION_CSVS.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            if "track_id" in df.columns:
                region_tracks[name] = set(df["track_id"].astype(str))
            elif "filename" in df.columns:
                region_tracks[name] = set(df["filename"].astype(str).apply(lambda s: os.path.splitext(os.path.basename(s))[0].split("_")[-1]))
            else:
                raise ValueError(f"{path} must contain column 'track_id' or 'filename'")

    tc_by_track = index_earthcare_by_track(ATL_TC_FOLDER)
    ebd_by_ts   = index_earthcare_by_ts(ATL_EBD_FOLDER)
    ald_by_ts   = index_earthcare_by_ts(ATL_ALD_FOLDER)

    if DEBUG_TRACKS:
        tracks_to_process = list(DEBUG_TRACKS)
        print(f"DEBUG_TRACKS active -> processing: {tracks_to_process}")
    else:
        tracks_to_process = sorted(set().union(*region_tracks.values())) if region_tracks else []
        print(f"Processing tracks from region CSVs: {len(tracks_to_process)}")

    agg: Dict[Tuple[pd.Timestamp, str], dict] = {}

    # global paired rows for scatter CSV
    paired_rows: List[dict] = []

    for track_id in tracks_to_process:
        recs = tc_by_track.get(track_id, [])
        if not recs:
            print(f"[SKIP] No ATL_TC record found for track_id={track_id}")
            continue

        for rec in recs:
            if not (start_dt <= rec["start_dt"] <= (end_dt + pd.Timedelta(days=1))):
                continue

            tc_file  = rec["tc_file"]
            ebd_file = ebd_by_ts.get(rec["ts_start"])
            ald_file = ald_by_ts.get(rec["ts_start"])

            try:
                data = ec.prepare_data_flexpart_earthcare_gfas(
                    flexpart_nc=FLEXPART_NC,
                    earthcare_file=tc_file,
                    ebd_file=ebd_file,
                    ald_file=ald_file,
                    gfas_folder=GFAS_DIR,
                    gfas_variant="01",
                )
            except Exception as e:
                print(f"[FAIL] prepare_data failed for {track_id} {rec['ts_start']}: {e}")
                continue

            # Track geometry/time (EarthCARE ATL_TC)
            lat_track = np.asarray(data.get("lat_ec", []), dtype=float)
            lon_track = np.asarray(data.get("lon_ec_wrapped", wrap_lon(data.get("lon_ec", []))), dtype=float)
            dt_track = pd.to_datetime(data.get("dt", []))
            if len(dt_track) == 0 or lat_track.size == 0:
                print(f"[SKIP] Missing EarthCARE track geometry for {track_id} {rec['ts_start']}")
                continue

            # --- Smoke classifier from ATL_TC classes (per time index) ---
            smoke_ids = (13, 14, 27)  # Smoke, dusty smoke, stratospheric smoke
            class_tc = data.get("class_data_raw", data.get("class_data", None))
            class_tc = np.asarray(class_tc) if class_tc is not None else None

            have_tc_classes = (
                class_tc is not None and class_tc.ndim == 2 and class_tc.shape[0] == len(dt_track)
            )


            # FLEXPART geometry/time (native FP track)
            fp_lat = np.asarray(data.get("fp_lat", []), dtype=float)
            fp_lon = np.asarray(data.get("fp_lon", []), dtype=float)  
            fp_dt_geom = data.get("fp_dt_1d", None)
            fp_dt_geom = np.asarray(pd.to_datetime(fp_dt_geom)).astype("datetime64[ns]") if fp_dt_geom is not None else None

            # Native time/altitude series
            (ec_t, ec_y), (fp_t, fp_y) = get_native_series(data, COMPARE_MODE)

            # --- Optional: convert plume heights to AGL (above local ground) ---
            if COMPARE_AGL:
                elev_track = np.asarray(data.get("elev", []), dtype=float)
                if ec_t is not None and ec_y is not None and elev_track.size == lat_track.size:
                    idx_ec = nearest_index_by_time(np.asarray(dt_track).astype("datetime64[ns]"), ec_t)
                    idx_ec = np.clip(idx_ec, 0, lat_track.size - 1)
                    ec_y = ec_y - elev_track[idx_ec]

                # FLEXPART: helper already provides fp_elev on the same 1D track as fp_lat/fp_lon (km)
                fp_elev = np.asarray(data.get("fp_elev", []), dtype=float)
                if fp_y is not None and fp_elev.size:
                    n = min(fp_y.size, fp_elev.size, fp_lat.size, fp_lon.size)
                    fp_y = fp_y[:n] - fp_elev[:n]
                    if fp_t is not None: fp_t = fp_t[:n]
                    fp_lat = fp_lat[:n]
                    fp_lon = fp_lon[:n]
            if (ec_t is None or ec_y is None) and (fp_t is None or fp_y is None):
                print(f"[SKIP] Missing both EC and FP SPH series for mode={COMPARE_MODE} track={track_id} ts={rec['ts_start']}")
                continue

            day = pd.to_datetime(rec["start_dt"].date())

            if fp_t is not None and fp_dt_geom is not None and fp_t.size != fp_dt_geom.size:
                n = min(fp_t.size, fp_lat.size, fp_lon.size)
                fp_t = fp_t[:n]
                fp_y = fp_y[:n] if fp_y is not None else None
                fp_lat = fp_lat[:n]
                fp_lon = fp_lon[:n]

            # For each region, gather points inside box
            for region_name, box in REGION_BOXES.items():
                key = (day, region_name)
                if key not in agg:
                    agg[key] = {
                        "ec_vals": [],       # list of arrays (native EC SPH values in box)
                        "fp_vals": [],       # list of arrays (native FP SPH values in box)
                        "pairs_fp": [],      # list of arrays (fp values in box that have EC match)
                        "pairs_ec": [],      # list of arrays (ec values nearest to fp times)
                        "pairs_dtsec": [],   # list of arrays (abs dt seconds for pairing)
                        "tracks": {},        # for plotting
                    }

                # ---- EarthCARE in-box (native SPH points) ----
                if ec_t is not None and ec_y is not None and ec_y.size > 0:
                    # map each EC SPH time to a lon/lat via nearest track time
                    idx = nearest_index_by_time(np.asarray(dt_track).astype("datetime64[ns]"), ec_t)
                    # idx should be valid indices into lat_track/lon_track
                    idx = np.clip(idx, 0, lat_track.size - 1)
                    lat_ec_pts = lat_track[idx]
                    lon_ec_pts = lon_track[idx]

                    m_ec = in_box_mask(lon_ec_pts, lat_ec_pts, box) & np.isfinite(ec_y)
                    if np.any(m_ec):
                        agg[key]["ec_vals"].append(ec_y[m_ec])

                # ---- FLEXPART in-box (native SPH points) ----
                if fp_t is not None and fp_y is not None and fp_y.size > 0 and fp_lat.size == fp_y.size:
                    m_fp = in_box_mask(fp_lon, fp_lat, box) & np.isfinite(fp_y)
                    if np.any(m_fp):
                        fp_in = fp_y[m_fp]
                        fp_times_in = fp_t[m_fp]
                        agg[key]["fp_vals"].append(fp_in)

                        # ---- Pairing for scatter: EC nearest-at-FP-time ----
                        if ec_t is not None and ec_y is not None and ec_y.size > 0:
                            ec_at_fp, dtsec = nearest_values_by_time(ec_t, ec_y, fp_times_in)
                            ok = np.isfinite(fp_in) & np.isfinite(ec_at_fp) & np.isfinite(dtsec) & (dtsec <= PAIR_MAX_ABS_DT_SECONDS)
                            if np.any(ok):
                                fp_ok = fp_in[ok]
                                ec_ok = ec_at_fp[ok]
                                dt_ok = dtsec[ok]
                                fp_times_ok = fp_times_in[ok]

                                agg[key]["pairs_fp"].append(fp_ok)
                                agg[key]["pairs_ec"].append(ec_ok)
                                agg[key]["pairs_dtsec"].append(dt_ok)

                                # --- is_smoke per paired point (based on nearest ATL_TC time row) ---
                                if have_tc_classes:
                                    idx_tc = nearest_index_by_time(np.asarray(dt_track).astype("datetime64[ns]"), fp_times_ok)
                                    idx_tc = np.clip(idx_tc, 0, class_tc.shape[0] - 1)

                                    # any smoke class at that time (across height bins)
                                    is_smoke_arr = np.array(
                                        [np.any(np.isin(class_tc[i, :], smoke_ids)) for i in idx_tc],
                                        dtype=bool,
                                    )
                                else:
                                    is_smoke_arr = np.zeros(fp_ok.shape, dtype=bool)

                                # store row-wise for global scatter CSV
                                for a, b, dts, smk in zip(fp_ok, ec_ok, dt_ok, is_smoke_arr):
                                    paired_rows.append({
                                        "day": day.strftime("%Y-%m-%d"),
                                        "region": region_name,
                                        "track_id": track_id,
                                        "mode": COMPARE_MODE,
                                        "fp_km": float(a),
                                        "ec_km_at_fp_time": float(b),
                                        "diff_fp_minus_ec_km": float(a - b),
                                        "abs_dt_seconds": float(dts),
                                        "is_smoke": bool(smk), 
                                    })


                # store track geometry for plotting (one per day/region/track)
                agg[key]["tracks"][track_id] = {
                    "track_id": track_id,
                    "lon_raw": np.asarray(data.get("lon_ec", []), dtype=float),
                    "lat": lat_track,
                }

            print(f"Processed: {track_id} {rec['ts_start']}")

    if not agg:
        print("No intersections found between tracks and boxes in the selected period.")
        return

    # ============================================================
    # DAILY STATS (point-based means) + optional maps
    # ============================================================

    daily_rows_fp = []
    daily_rows_ec = []
    daily_rows_pairs = []

    for (day, region_name), d in sorted(agg.items(), key=lambda x: (x[0][0], x[0][1])):
        tracks_list = list(d["tracks"].values())

        # EarthCARE daily stats
        if d["ec_vals"]:
            ec_all = np.concatenate(d["ec_vals"]).astype(float)
            ec_all = ec_all[np.isfinite(ec_all)]
            s = summarize_alts_basic(ec_all)
            daily_rows_ec.append({
                "day": day.strftime("%Y-%m-%d"),
                "region": region_name,
                "mode": COMPARE_MODE,
                "n_tracks_with_any_data": len(tracks_list),
                "n_ec_points_in_box": s["n"],
                "ec_km_mean": s["mean"],
                "ec_km_median": s["median"],
                "ec_km_std": s["std"],
                "ec_km_p25": s["p25"],
                "ec_km_p75": s["p75"],
                "ec_km_p90": s["p90"],
            })

        # FLEXPART daily stats
        if d["fp_vals"]:
            fp_all = np.concatenate(d["fp_vals"]).astype(float)
            fp_all = fp_all[np.isfinite(fp_all)]
            s = summarize_alts_basic(fp_all)
            daily_rows_fp.append({
                "day": day.strftime("%Y-%m-%d"),
                "region": region_name,
                "mode": COMPARE_MODE,
                "n_tracks_with_any_data": len(tracks_list),
                "n_fp_points_in_box": s["n"],
                "fp_km_mean": s["mean"],
                "fp_km_median": s["median"],
                "fp_km_std": s["std"],
                "fp_km_p25": s["p25"],
                "fp_km_p75": s["p75"],
                "fp_km_p90": s["p90"],
            })

        # Paired daily stats (for scatter comparability)
        if d["pairs_fp"] and d["pairs_ec"]:
            fp_p = np.concatenate(d["pairs_fp"]).astype(float)
            ec_p = np.concatenate(d["pairs_ec"]).astype(float)
            dt_p = np.concatenate(d["pairs_dtsec"]).astype(float)

            diff = fp_p - ec_p
            s_diff = summarize_alts_basic(diff)
            daily_rows_pairs.append({
                "day": day.strftime("%Y-%m-%d"),
                "region": region_name,
                "mode": COMPARE_MODE,
                "n_tracks_with_any_data": len(tracks_list),
                "n_pairs_used": int(np.isfinite(diff).sum()),
                "diff_fp_minus_ec_km_mean": s_diff["mean"],
                "diff_fp_minus_ec_km_median": s_diff["median"],
                "diff_fp_minus_ec_km_std": s_diff["std"],
                "diff_fp_minus_ec_km_p25": s_diff["p25"],
                "diff_fp_minus_ec_km_p75": s_diff["p75"],
                "diff_fp_minus_ec_km_p90": s_diff["p90"],
                "median_abs_dt_seconds": float(np.nanmedian(dt_p)) if dt_p.size else np.nan,
            })

        # optional daily map(s)
        if MAKE_DAILY_PLOTS:
            # You can choose which tracks to visualize; currently it shows all tracks that contributed to this (day, region)
            tracks_for_plot = tracks_list
            if tracks_for_plot:
                out_png = outpath(maps_ec, f"{region_name}_{day.strftime('%Y%m%d')}_tracks.png", file_prefix)
                plot_day_region_map_only(out_png, region_name, REGION_BOXES[region_name], tracks_for_plot, day)

    # Save daily CSVs
    if daily_rows_fp:
        df = pd.DataFrame(daily_rows_fp)
        out_csv = outpath(out_flex, f"daily_box_flexpart_points_{COMPARE_MODE}_{START_DATE}_to_{END_DATE}.csv", file_prefix)
        df.to_csv(out_csv, index=False)
        print(f"Saved FLEXPART daily (point-based) stats: {out_csv}")

    if daily_rows_ec:
        df = pd.DataFrame(daily_rows_ec)
        out_csv = outpath(out_ec, f"daily_box_earthcare_points_{COMPARE_MODE}_{START_DATE}_to_{END_DATE}.csv", file_prefix)
        df.to_csv(out_csv, index=False)
        print(f"Saved EarthCARE daily (point-based) stats: {out_csv}")

    if daily_rows_pairs:
        df = pd.DataFrame(daily_rows_pairs)
        out_csv = outpath(out_comb, f"daily_box_paired_fp_minus_ec_{COMPARE_MODE}_{START_DATE}_to_{END_DATE}.csv", file_prefix)
        df.to_csv(out_csv, index=False)
        print(f"Saved paired daily stats (FP vs EC at FP time): {out_csv}")

    # Save paired scatter points (one file for whole run)
    if paired_rows:
        df = pd.DataFrame(paired_rows)
        out_csv = outpath(out_comb, f"paired_fp_ec_points_{COMPARE_MODE}_{START_DATE}_to_{END_DATE}.csv", file_prefix)
        df.to_csv(out_csv, index=False)
        print(f"Saved paired scatter points: {out_csv}")

    # ============================================================
    # PERIOD STATS (point-based, aggregating all days)
    # ============================================================

    pstart = start_dt.normalize()
    pend   = end_dt.normalize()

    period_agg: Dict[str, dict] = {}
    for (day, region_name), d in agg.items():
        if not (pstart <= day <= pend):
            continue
        if region_name not in period_agg:
            period_agg[region_name] = {
                "ec_vals": [],
                "fp_vals": [],
                "pairs_fp": [],
                "pairs_ec": [],
                "pairs_dtsec": [],
                "tracks": {},
            }
        period_agg[region_name]["ec_vals"].extend(d["ec_vals"])
        period_agg[region_name]["fp_vals"].extend(d["fp_vals"])
        period_agg[region_name]["pairs_fp"].extend(d["pairs_fp"])
        period_agg[region_name]["pairs_ec"].extend(d["pairs_ec"])
        period_agg[region_name]["pairs_dtsec"].extend(d["pairs_dtsec"])
        period_agg[region_name]["tracks"].update(d["tracks"])

    period_rows_fp = []
    period_rows_ec = []
    period_rows_pairs = []

    for region_name in sorted(period_agg.keys()):
        d = period_agg[region_name]
        tracks_list = list(d["tracks"].values())

        if d["ec_vals"]:
            ec_all = np.concatenate(d["ec_vals"]).astype(float)
            ec_all = ec_all[np.isfinite(ec_all)]
            s = summarize_alts_basic(ec_all)
            period_rows_ec.append({
                "period_start": pstart.strftime("%Y-%m-%d"),
                "period_end": pend.strftime("%Y-%m-%d"),
                "region": region_name,
                "mode": COMPARE_MODE,
                "n_tracks_with_any_data": len(tracks_list),
                "n_ec_points_in_box": s["n"],
                "ec_km_mean": s["mean"],
                "ec_km_median": s["median"],
                "ec_km_std": s["std"],
                "ec_km_p25": s["p25"],
                "ec_km_p75": s["p75"],
                "ec_km_p90": s["p90"],
            })

        if d["fp_vals"]:
            fp_all = np.concatenate(d["fp_vals"]).astype(float)
            fp_all = fp_all[np.isfinite(fp_all)]
            s = summarize_alts_basic(fp_all)
            period_rows_fp.append({
                "period_start": pstart.strftime("%Y-%m-%d"),
                "period_end": pend.strftime("%Y-%m-%d"),
                "region": region_name,
                "mode": COMPARE_MODE,
                "n_tracks_with_any_data": len(tracks_list),
                "n_fp_points_in_box": s["n"],
                "fp_km_mean": s["mean"],
                "fp_km_median": s["median"],
                "fp_km_std": s["std"],
                "fp_km_p25": s["p25"],
                "fp_km_p75": s["p75"],
                "fp_km_p90": s["p90"],
            })

        if d["pairs_fp"] and d["pairs_ec"]:
            fp_p = np.concatenate(d["pairs_fp"]).astype(float)
            ec_p = np.concatenate(d["pairs_ec"]).astype(float)
            dt_p = np.concatenate(d["pairs_dtsec"]).astype(float)
            diff = fp_p - ec_p
            s = summarize_alts_basic(diff)
            period_rows_pairs.append({
                "period_start": pstart.strftime("%Y-%m-%d"),
                "period_end": pend.strftime("%Y-%m-%d"),
                "region": region_name,
                "mode": COMPARE_MODE,
                "n_tracks_with_any_data": len(tracks_list),
                "n_pairs_used": int(np.isfinite(diff).sum()),
                "diff_fp_minus_ec_km_mean": s["mean"],
                "diff_fp_minus_ec_km_median": s["median"],
                "diff_fp_minus_ec_km_std": s["std"],
                "diff_fp_minus_ec_km_p25": s["p25"],
                "diff_fp_minus_ec_km_p75": s["p75"],
                "diff_fp_minus_ec_km_p90": s["p90"],
                "median_abs_dt_seconds": float(np.nanmedian(dt_p)) if dt_p.size else np.nan,
            })

        # Period maps (optional)
        if MAKE_PERIOD_PLOTS and MAKE_PERIOD_MAPS and tracks_list:
            out_dir = os.path.join(maps_ec, f"PERIOD_{pstart.strftime('%Y%m%d')}_{pend.strftime('%Y%m%d')}")
            ensure_dir(out_dir)
            out_png = outpath(out_dir, f"{region_name}_tracks_period.png", file_prefix)
            plot_day_region_map_only(
                out_png=out_png,
                region_name=f"{region_name} (Period)",
                box=REGION_BOXES[region_name],
                tracks=tracks_list,
                day=pstart,
                period_start=pstart,
                period_end=pend,
            )

    if period_rows_fp:
        df = pd.DataFrame(period_rows_fp)
        out_csv = outpath(out_flex, f"period_box_flexpart_points_{COMPARE_MODE}_{START_DATE}_to_{END_DATE}.csv", file_prefix)
        df.to_csv(out_csv, index=False)
        print(f"Saved FLEXPART period (point-based) stats: {out_csv}")

    if period_rows_ec:
        df = pd.DataFrame(period_rows_ec)
        out_csv = outpath(out_ec, f"period_box_earthcare_points_{COMPARE_MODE}_{START_DATE}_to_{END_DATE}.csv", file_prefix)
        df.to_csv(out_csv, index=False)
        print(f"Saved EarthCARE period (point-based) stats: {out_csv}")

    if period_rows_pairs:
        df = pd.DataFrame(period_rows_pairs)
        out_csv = outpath(out_comb, f"period_box_paired_fp_minus_ec_{COMPARE_MODE}_{START_DATE}_to_{END_DATE}.csv", file_prefix)
        df.to_csv(out_csv, index=False)
        print(f"Saved paired period stats: {out_csv}")

    # ============================================================
    # NORTHERN HEMISPHERE OVERVIEW (RECTANGULAR) + ALL 4 BOXES
    # ============================================================

    if MAKE_OVERVIEW_PLOT:
        tracks_union: Dict[str, dict] = {}
        for region_name, d in period_agg.items():
            for tid, tr in d["tracks"].items():
                tracks_union[tid] = tr

        tracks_union_list = list(tracks_union.values())
        if tracks_union_list:
            out_dir = os.path.join(OUTROOT, "overview_maps")
            ensure_dir(out_dir)
            out_png = outpath(out_dir, f"NH_rectangular_tracks_and_boxes_{START_DATE}_to_{END_DATE}.png", file_prefix)

            plot_nh_rectangular_tracks_and_boxes(
                out_png=out_png,
                tracks=tracks_union_list,
                region_boxes=REGION_BOXES,
                title=f"Boxes over the Northern Hemisphere with applied tracks from {START_DATE} to {END_DATE}",
                lat_min=35.0,
                lon_min=-160.0,
                lon_max=60.0,
            )
            print(f"Saved NH rectangular overview map: {out_png}")


if __name__ == "__main__":
    main()
