from __future__ import annotations
import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

# ============================================================
# USER SETTINGS
# ============================================================

OUTDIR = "/xnilu_wrk2/projects/NEVAR/Irene/region_stats"
only_smoke_for_tc_sph = False
no_clouds_in_flexpart = False

if only_smoke_for_tc_sph:
    OUTDIR = "/xnilu_wrk2/projects/NEVAR/Irene/region_stats/smoke_only_for_tc_sph"

if no_clouds_in_flexpart:
    OUTDIR = "/xnilu_wrk2/projects/NEVAR/Irene/region_stats/no_clouds_in_flexpart"

# Debug plotting mode (auto-load latest debug_track run)
DEBUG_TRACK = False
DEBUG_RUN_TAG = None # set to a specific folder name under OUTDIR/debug_track/
DEBUG_SOURCE_FILE = None # "EC_FLEXPART_GFAS_20250601_T051320Z_T170405Z_05730C_GFAS_EC_height_comparison.csv" # "EC_FLEXPART_GFAS_20250601_T204948Z_T175623Z_05740D_GFAS_EC_height_comparison.csv"

# These will be overridden from the manifest if DEBUG_TRACK=True
COMPARE_MODE = "top"  # "ext", "top", "bottom"
START_DATE = "2025-05-25"
END_DATE   = "2025-06-15"

# What to make
MAKE_DAILY_FP_EC_SUBPLOT = True       # two panels: FP daily + EC daily
MAKE_DAILY_PAIRED_DIFF   = False      # FP-EC at FP times (daily)
MAKE_PERIOD_SUMMARY_PLOT = False      # period summary by region
MAKE_SCATTER_PAIRED      = True       # uses paired_fp_ec_points_*.csv (best scatter)
MAKE_SCATTER_DAILY_AVG_2X2 = True     # daily mean/median per day per region scatter (EC x, FP y)
MAKE_DAILY_FP_EC_SUBPLOT_FROM_PAIRS = True  # True -> build daily FP/EC panels from paired points
MAKE_DAILY_FP_EC_SUBPLOT_FROM_PAIRS_JUST_ONE_N_POINTS_PANEL = True


# Daily plot style:
#   "mean_std"   -> mean line + shaded ±1 std
#   "median_iqr" -> median line + shaded p25–p75
DAILY_STYLE = "mean_std"

# y-limits (set None to auto)
YLIM_ALT_KM  = (0, 16)      # for FP/EC altitudes
YLIM_DIFF_KM = (-5, 5)      # for paired FP-EC differences

REGION_COLORS = {
    "ATLANTIC": "tab:blue",
    "CANADA":   "tab:orange",
    "EUROPE":   "tab:green",
    "POLAR":    "tab:red",
}


# =================================================================
# GFAS ↔ EarthCARE comparison plots
# =================================================================

PLOT_GFAS_EC = True
GFAS_EC_MATCH_DIR = "/xnilu_wrk2/projects/NEVAR/Irene/FLEXPART_EarthCARE_GFAS_BA_GFASprox_100km" # "/xnilu_wrk2/projects/NEVAR/Irene/FLEXPART_EarthCARE_GFAS_few_classifications_no_cloud/2_layers" #

if only_smoke_for_tc_sph:
    GFAS_EC_MATCH_DIR = "/xnilu_wrk2/projects/NEVAR/Irene/FLEXPART_EarthCARE_only_smoke_TC_GFASprox_100km"

if no_clouds_in_flexpart:
    GFAS_EC_MATCH_DIR = "/xnilu_wrk2/projects/NEVAR/Irene/FLEXPART_EarthCARE_GFAS_no_clouds_in_flexpart"


# Options in match CSVs include:
#   "d_top_minus_apt"
#   "d_bottom_minus_apb"
#   "d_ext_minus_mami_assume_amsl"
#   "d_top_minus_injh_assume_amsl"

GFAS_EC_DIFF_COL = "d_top_minus_apt"

# If you have region info per overpass in the match CSVs, leave True.
# If not, everything will be aggregated as region="all".
USE_REGION_MAPPING = False  # kept for compatibility; not used unless you extend mapping logic


def _pick_latest_debug_outroot(outdir: str) -> str:
    dbg_root = os.path.join(outdir, "debug_track")
    if not os.path.isdir(dbg_root):
        raise FileNotFoundError(f"No debug_track folder found: {dbg_root}")

    subdirs = [
        os.path.join(dbg_root, d)
        for d in os.listdir(dbg_root)
        if os.path.isdir(os.path.join(dbg_root, d))
    ]
    if not subdirs:
        raise FileNotFoundError(f"No debug runs found under: {dbg_root}")

    # newest by mtime
    subdirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return subdirs[0]


def _load_manifest_if_present(outroot: str, file_prefix: str) -> dict | None:
    man = os.path.join(outroot, "files", f"{file_prefix}run_manifest.json")
    if not os.path.exists(man):
        return None
    import json
    with open(man, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# FILE HELPERS
# ============================================================

OUTROOT = OUTDIR        # will be updated in main() if DEBUG_TRACK
FILE_PREFIX = ""        # will be "debug_track_" in debug mode

def f_daily_fp() -> str:
    return os.path.join(
        OUTROOT, "flexpart",
        f"{FILE_PREFIX}daily_box_flexpart_points_{COMPARE_MODE}_{START_DATE}_to_{END_DATE}.csv"
    )

def f_daily_ec() -> str:
    return os.path.join(
        OUTROOT, "earthcare",
        f"{FILE_PREFIX}daily_box_earthcare_points_{COMPARE_MODE}_{START_DATE}_to_{END_DATE}.csv"
    )

def f_daily_paired_diff() -> str:
    return os.path.join(
        OUTROOT, "combined",
        f"{FILE_PREFIX}daily_box_paired_fp_minus_ec_{COMPARE_MODE}_{START_DATE}_to_{END_DATE}.csv"
    )

def f_pairs_points() -> str:
    return os.path.join(
        OUTROOT, "combined",
        f"{FILE_PREFIX}paired_fp_ec_points_{COMPARE_MODE}_{START_DATE}_to_{END_DATE}.csv"
    )

def f_period_fp() -> str:
    return os.path.join(
        OUTROOT, "flexpart",
        f"{FILE_PREFIX}period_box_flexpart_points_{COMPARE_MODE}_{START_DATE}_to_{END_DATE}.csv"
    )

def f_period_ec() -> str:
    return os.path.join(
        OUTROOT, "earthcare",
        f"{FILE_PREFIX}period_box_earthcare_points_{COMPARE_MODE}_{START_DATE}_to_{END_DATE}.csv"
    )

def f_period_paired_diff() -> str:
    return os.path.join(
        OUTROOT, "combined",
        f"{FILE_PREFIX}period_box_paired_fp_minus_ec_{COMPARE_MODE}_{START_DATE}_to_{END_DATE}.csv"
    )

def out_plot_dir() -> str:
    p = os.path.join(OUTROOT, "combined", "plots", "summary")
    os.makedirs(p, exist_ok=True)
    return p


def _load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "day" in df.columns:
        df["day"] = pd.to_datetime(df["day"], errors="coerce")
    return df


# ============================================================
# COLUMN NORMALIZATION
# ============================================================

def _infer_prefix(df: pd.DataFrame) -> str:
    for c in df.columns:
        if c.endswith("_km_mean"):
            return c[:-len("_km_mean")]
    raise ValueError(f"Could not infer prefix from columns: {list(df.columns)}")

def _get_cols(prefix: str, style: str):
    if style == "median_iqr":
        return {
            "center": f"{prefix}_km_median",
            "lo":     f"{prefix}_km_p25",
            "hi":     f"{prefix}_km_p75",
        }
    elif style == "mean_std":
        return {
            "center": f"{prefix}_km_mean",
            "std":    f"{prefix}_km_std",
        }
    else:
        raise ValueError("style must be 'mean_std' or 'median_iqr'")

def _find_count_col(df: pd.DataFrame, prefix: str) -> str | None:
    cand = f"n_{prefix}_points_in_box"
    if cand in df.columns:
        return cand
    for c in df.columns:
        if c.startswith("n_") and c.endswith("_points_in_box"):
            return c
    return None


# ============================================================
# PLOTTING UTILITIES
# ============================================================

def _savefig(path: str):
    # plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    

def _plot_daily_panel(ax, df: pd.DataFrame, title: str, ylabel: str, style: str, ylim=None):
    if df.empty:
        ax.set_title(title + " (no data)")
        ax.grid(True, alpha=0.3)
        return

    prefix = _infer_prefix(df)
    cols = _get_cols(prefix, style)
    count_col = _find_count_col(df, prefix)

    for region, g in df.groupby("region"):
        g = g.sort_values("day")
        if "day" not in g.columns:
            continue

        y = pd.to_numeric(g.get(cols["center"], np.nan), errors="coerce").to_numpy()

        (line,) = ax.plot(g["day"], y, linewidth=1.6, label=str(region))
        color = line.get_color()

        ax.scatter(g["day"], y, s=25, color=color, alpha=0.85)
        

        if style == "median_iqr":
            lo = pd.to_numeric(g.get(cols["lo"], np.nan), errors="coerce").to_numpy()
            hi = pd.to_numeric(g.get(cols["hi"], np.nan), errors="coerce").to_numpy()
            ax.fill_between(g["day"].values, lo, hi, color=color, alpha=0.18)
        else:
            std = pd.to_numeric(g.get(cols.get("std", ""), np.nan), errors="coerce").to_numpy()
            ax.fill_between(g["day"].values, y - std, y + std, color=color, alpha=0.18)

    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.grid(True, alpha=0.3)
    if ylim is not None:
        ax.set_ylim(*ylim)


def _daily_stats_from_pairs(
    df_pairs: pd.DataFrame,
    *,
    fp_col: str = "fp_km",
    ec_col: str = "ec_km_at_fp_time",
    use_smoke_only: bool = False,
    smoke_col: str = "is_smoke",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build daily stats tables (same schema as your existing daily_box_* CSVs)
    from the paired points file, so FP and EC have identical sample sizes per (day, region).

    Returns:
      df_daily_fp_like, df_daily_ec_like
    """
    df = df_pairs.copy()

    # day / region required
    if "day" not in df.columns:
        raise ValueError("paired points df is missing 'day'")
    if "region" not in df.columns:
        # keep consistent with your code; you can also choose to set region="all"
        df["region"] = "all"

    df["day"] = pd.to_datetime(df["day"], errors="coerce")
    df["fp_km"] = pd.to_numeric(df.get(fp_col, np.nan), errors="coerce")
    df["ec_km"] = pd.to_numeric(df.get(ec_col, np.nan), errors="coerce")

    df = df.dropna(subset=["day", "region", "fp_km", "ec_km"])

    # Optional smoke-only filter
    if use_smoke_only and (smoke_col in df.columns):
        df = df[df[smoke_col].astype(bool)].copy()

    # group key
    g_fp = df.groupby(["day", "region"])["fp_km"]
    g_ec = df.groupby(["day", "region"])["ec_km"]

    def _agg(g):
        out = g.agg(
            alt_km_mean="mean",
            alt_km_std="std",
            alt_km_median="median",
            alt_km_p25=lambda x: np.nanpercentile(x, 25),
            alt_km_p75=lambda x: np.nanpercentile(x, 75),
            n_alt_points_in_box="count",
        ).reset_index()
        return out

    df_fp = _agg(g_fp)
    df_ec = _agg(g_ec)

    return df_fp, df_ec



def _plot_counts_panel(ax, df: pd.DataFrame, ec_or_fp=None, title: str | None = None, logy: bool = False):
    """
    Plot n_*_points_in_box as a time series per region.
    Uses the same region grouping as _plot_daily_panel.
    """
    if df.empty:
        ax.set_title((title or "Counts") + " (no data)")
        ax.grid(True, alpha=0.3)
        return

    prefix = _infer_prefix(df)
    count_col = _find_count_col(df, prefix)

    if count_col is None or count_col not in df.columns:
        ax.set_title((title or "Counts") + " (missing count column)")
        ax.grid(True, alpha=0.3)
        return

    for region, g in df.groupby("region"):
        g = g.sort_values("day")
        if "day" not in g.columns:
            continue
        n = pd.to_numeric(g[count_col], errors="coerce").to_numpy()
        n = np.where(np.isfinite(n) & (n >= 0), n, np.nan)

        ax.plot(g["day"], n, linewidth=1.2, label=str(region))

    if title:
        ax.set_title(title, fontweight="bold")

    ax.set_ylabel("# points", fontweight="bold")

    if logy:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)


# ============================================================
# PERIOD SUMMARY
# ============================================================

def _plot_period_summary(df_fp: pd.DataFrame, df_ec: pd.DataFrame, df_paired: pd.DataFrame | None):
    plotdir = out_plot_dir()

    def pick_center(df: pd.DataFrame):
        p = _infer_prefix(df)
        c = (
            f"{p}_km_mean" if f"{p}_km_mean" in df.columns
            else (f"{p}_km_median" if f"{p}_km_median" in df.columns else None)
        )
        n = _find_count_col(df, p)
        return p, c, n

    fp_p, fp_c, fp_n = pick_center(df_fp)
    ec_p, ec_c, ec_n = pick_center(df_ec)

    keep_fp = ["region"] + ([fp_c] if fp_c else []) + ([fp_n] if fp_n else [])
    keep_ec = ["region"] + ([ec_c] if ec_c else []) + ([ec_n] if ec_n else [])
    d = pd.merge(df_fp[keep_fp], df_ec[keep_ec], on="region", how="outer", suffixes=("_fp", "_ec"))

    fig, ax = plt.subplots(figsize=(10, 5))

    regions = sorted([r for r in d["region"].dropna().unique()])
    x = np.arange(len(regions))

    fp_vals = np.array([d.loc[d["region"] == r, fp_c].values[0] if fp_c and (d["region"] == r).any() else np.nan for r in regions], float)
    ec_vals = np.array([d.loc[d["region"] == r, ec_c].values[0] if ec_c and (d["region"] == r).any() else np.nan for r in regions], float)

    ax.scatter(x - 0.1, fp_vals, s=60, alpha=0.9, label="FLEXPART (period)")
    ax.scatter(x + 0.1, ec_vals, s=60, alpha=0.9, label="EarthCARE (period)")

    ax.set_xticks(x)
    ax.set_xticklabels(regions)
    ax.set_ylabel("Altitude (km)", fontweight="bold")
    ax.set_title(f"Period summary for {START_DATE} to {END_DATE} with sph_{COMPARE_MODE}", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()

    out_png = os.path.join(plotdir, f"period_summary_FP_EC_{COMPARE_MODE}_{START_DATE}_{END_DATE}.png")
    _savefig(out_png)
    print(f"Saved: {out_png}")

    if df_paired is not None and not df_paired.empty:
        p = _infer_prefix(df_paired)
        c = f"{p}_km_mean" if f"{p}_km_mean" in df_paired.columns else None
        if c is not None and "region" in df_paired.columns:
            fig, ax = plt.subplots(figsize=(10, 5))
            regions2 = sorted(df_paired["region"].dropna().unique())
            x2 = np.arange(len(regions2))
            vals = np.array([df_paired.loc[df_paired["region"] == r, c].values[0] for r in regions2], float)

            ax.bar(x2, vals, alpha=0.85)
            ax.axhline(0, linewidth=1)
            ax.set_xticks(x2)
            ax.set_xticklabels(regions2)
            ax.set_ylabel("FP - EC (km)", fontweight="bold")
            ax.set_title(f"Period paired FP-EC difference\n{START_DATE} → {END_DATE} | sph_{COMPARE_MODE}", fontweight="bold")
            ax.grid(True, alpha=0.3)

            out_png = os.path.join(plotdir, f"period_paired_fp_minus_ec_{COMPARE_MODE}_{START_DATE}_{END_DATE}.png")
            _savefig(out_png)
            print(f"Saved: {out_png}")


# ============================================================
# SCATTER FROM PAIRED POINTS  (FIXED FOR YOUR COLUMNS)
# ============================================================

def _plot_scatter_from_pairs(df_pairs: pd.DataFrame):
    plotdir = out_plot_dir()

    # Expected paired file columns (example):
    # ['day','region','track_id','mode','fp_km','ec_km_at_fp_time','diff_fp_minus_ec_km','abs_dt_seconds']
    fp_col = "fp_km" if "fp_km" in df_pairs.columns else None
    ec_col = "ec_km_at_fp_time" if "ec_km_at_fp_time" in df_pairs.columns else None

    if fp_col is None or ec_col is None:
        raise ValueError(
            "paired points file missing required columns.\n"
            "Need fp_km and one of ec_km_at_fp_time/ec_km_at_fp/ec_alt_at_fp_km/ec_km.\n"
            f"Columns: {list(df_pairs.columns)}"
        )

    df = df_pairs.copy()
    df["fp_km"] = pd.to_numeric(df[fp_col], errors="coerce")
    df["ec_km"] = pd.to_numeric(df[ec_col], errors="coerce")
    df = df.dropna(subset=["fp_km", "ec_km"])

    fig, ax = plt.subplots(figsize=(7.5, 7.5))

    if "region" in df.columns:
        # for region, g in df.groupby("region"):
        #     ax.scatter(g["fp_km"], g["ec_km"], s=10, alpha=0.35, label=str(region))
        for region, g in df.groupby("region"):
            if "is_smoke" in g.columns:
                smoke = g["is_smoke"].astype(bool)
                ax.scatter(g.loc[~smoke, "ec_km"], g.loc[~smoke, "fp_km"],
                        s=10, alpha=0.25, color="0.6", linewidths=0)
                ax.scatter(g.loc[smoke, "ec_km"], g.loc[smoke, "fp_km"],
                        s=10, alpha=0.75, color="tab:red", linewidths=0, label="smoke")
            else:
                ax.scatter(g["ec_km"], g["fp_km"], s=10, alpha=0.35, label=str(region), linewidths=0)


        ax.legend(fontsize=10, loc="lower right")
    else:
        ax.scatter(df["fp_km"], df["ec_km"], s=10, alpha=0.35)

    vmin = float(np.nanmin([df["fp_km"].min(), df["ec_km"].min()]))
    vmax = float(np.nanmax([df["fp_km"].max(), df["ec_km"].max()]))
    ax.plot([vmin, vmax], [vmin, vmax], linestyle="--", linewidth=1)

    ax.set_xlabel("FLEXPART altitude at FP times [km]")
    ax.set_ylabel("EarthCARE altitude at same FP times [km]")
    ax.set_title(
        f"Paired point scatter for sph_{COMPARE_MODE} between {START_DATE} to {END_DATE} \n"
        f"N={len(df)} paired points",
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)

    out_png = os.path.join(plotdir, f"scatter_paired_points_fp_vs_ec_{COMPARE_MODE}_{START_DATE}_{END_DATE}.png")
    _savefig(out_png)
    print(f"Saved: {out_png}")



def _plot_scatter_from_pairs_2x2_by_region(
    df_pairs: pd.DataFrame,
    regions: list[str] | None = None,
    max_regions: int = 4,
    share_limits: bool = True,
):
    """
    Make a 2x2 panel scatter: EC-at-FP-time (x) vs FP (y), one region per axis.

    If regions=None:
      - picks up to max_regions regions with the most paired points.

    share_limits=True:
      - uses common x/y limits across all panels, so regions are directly comparable.
    """
    plotdir = out_plot_dir()

    fp_col = "fp_km" if "fp_km" in df_pairs.columns else None
    ec_col = "ec_km_at_fp_time" if "ec_km_at_fp_time" in df_pairs.columns else None

    if fp_col is None or ec_col is None:
        raise ValueError(
            "paired points file missing required columns.\n"
            "Need fp_km and ec_km_at_fp_time.\n"
            f"Columns: {list(df_pairs.columns)}"
        )

    df = df_pairs.copy()
    df["fp_km"] = pd.to_numeric(df[fp_col], errors="coerce")
    df["ec_km"] = pd.to_numeric(df[ec_col], errors="coerce")

    # Require region for 2x2-per-region plotting
    if "region" not in df.columns:
        raise ValueError("paired points file has no 'region' column; cannot do per-region panels.")

    df = df.dropna(subset=["fp_km", "ec_km", "region"])

    # Choose regions
    if regions is None:
        counts = df["region"].value_counts()
        regions = list(counts.index[:max_regions])
    else:
        regions = regions[:max_regions]

    if len(regions) == 0:
        raise ValueError("No regions found after filtering.")

    # Compute shared limits (optional)
    if share_limits:
        dsub = df[df["region"].isin(regions)]
        vmin = float(np.nanmin([dsub["fp_km"].min(), dsub["ec_km"].min()]))
        vmax = float(np.nanmax([dsub["fp_km"].max(), dsub["ec_km"].max()]))

        pad = 0.03 * (vmax - vmin) if np.isfinite(vmax - vmin) and vmax > vmin else 0.5
        lims = (vmin - pad, vmax + pad)
    else:
        lims = None

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))#, sharex=share_limits, sharey=share_limits)
    axes = axes.ravel()

    for i, region in enumerate(regions):
        ax = axes[i]
        g = df[df["region"] == region]

        color = REGION_COLORS.get(str(region), "0.3")
        print(f"Region {region}: plotting {color} ")

        # EarthCARE on X, FLEXPART on Y
        if "is_smoke" in g.columns:
            smoke = g["is_smoke"].astype(bool)
            # ax.scatter(g.loc[~smoke, "ec_km"], g.loc[~smoke, "fp_km"],
            #         s=10, alpha=1, color=color, linewidths=0)
            ax.scatter(g.loc[smoke, "ec_km"], g.loc[smoke, "fp_km"],
                    s=10, alpha=1, color=color, linewidths=0)
            
            if only_smoke_for_tc_sph:
                n = int(g["is_smoke"].astype(bool).sum())
                print(f"Region {region}: {n} smoke points out of {len(g)} total points")
            else:
                n = len(g)
        else:
            ax.scatter(g["ec_km"], g["fp_km"], s=10, alpha=0.35, color=color, linewidths=0)
            n = len(g)


        # 1:1 line
        if lims is not None:
            ax.plot([lims[0], lims[1]], [lims[0], lims[1]], linestyle="--", linewidth=1)
            ax.set_xlim(*lims)
            ax.set_ylim(*lims)
        else:
            vmin_r = float(np.nanmin([g["fp_km"].min(), g["ec_km"].min()]))
            vmax_r = float(np.nanmax([g["fp_km"].max(), g["ec_km"].max()]))
            ax.plot([vmin_r, vmax_r], [vmin_r, vmax_r], linestyle="--", linewidth=1)

        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.yaxis.set_major_locator(MultipleLocator(2))

        ax.set_title(f"{region} (N={n})", fontweight="bold")
        ax.grid(True, alpha=0.3)

    # Turn off unused panels if <4 regions
    for j in range(len(regions), 4):
        axes[j].axis("off")

    fig.supxlabel("EarthCARE altitude at matched FP times [km]")
    fig.supylabel("FLEXPART altitude [km]")

    fig.suptitle(
        f"Paired scatter by region for sph_{COMPARE_MODE} during the period {START_DATE} to {END_DATE}",
        fontweight="bold",
        y=0.98,
    )

    out_png = os.path.join(
        plotdir,
        f"scatter_paired_points_ec_vs_fp_2x2_by_region_{COMPARE_MODE}_{START_DATE}_{END_DATE}.png",
    )

    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")





def _plot_scatter_daily_avg_2x2_by_region(
    df_daily_ec: pd.DataFrame,
    df_daily_fp: pd.DataFrame,
    regions: list[str] | None = None,
    max_regions: int = 4,
    share_limits: bool = True,
    style: str | None = None,   # default: uses global DAILY_STYLE
):
    """
    2x2 panel scatter using DAILY aggregated values:
      x = EarthCARE daily center (mean/median depending on style)
      y = FLEXPART daily center (mean/median depending on style)

    One point per (day, region), after merging EC and FP daily tables on (day, region).
    """
    plotdir = out_plot_dir()
    if style is None:
        style = DAILY_STYLE

    if df_daily_ec.empty or df_daily_fp.empty:
        raise ValueError("Daily EC/FP dataframe is empty; cannot make daily-average scatter.")

    # Infer which columns to use as the daily 'center' statistic
    ec_prefix = _infer_prefix(df_daily_ec)
    fp_prefix = _infer_prefix(df_daily_fp)

    ec_cols = _get_cols(ec_prefix, style)
    fp_cols = _get_cols(fp_prefix, style)

    ec_center = ec_cols["center"]
    fp_center = fp_cols["center"]

    keep_ec = ["day", "region", ec_center]
    keep_fp = ["day", "region", fp_center]

    d_ec = df_daily_ec[keep_ec].copy()
    d_fp = df_daily_fp[keep_fp].copy()

    d_ec["day"] = pd.to_datetime(d_ec["day"], errors="coerce")
    d_fp["day"] = pd.to_datetime(d_fp["day"], errors="coerce")

    d_ec[ec_center] = pd.to_numeric(d_ec[ec_center], errors="coerce")
    d_fp[fp_center] = pd.to_numeric(d_fp[fp_center], errors="coerce")

    d = pd.merge(
        d_ec,
        d_fp,
        on=["day", "region"],
        how="inner",
        suffixes=("_ec", "_fp"),
    ).dropna(subset=[ec_center, fp_center, "region"])

    # Require region for 2x2 per-region plotting
    if "region" not in d.columns or d["region"].isna().all():
        raise ValueError("Merged daily table has no valid 'region' values; cannot do per-region panels.")

    # Choose regions (top N by number of matched daily points)
    if regions is None:
        counts = d["region"].value_counts()
        regions = list(counts.index[:max_regions])
    else:
        regions = regions[:max_regions]

    if len(regions) == 0:
        raise ValueError("No regions found after filtering for daily-average scatter.")

    dsub = d[d["region"].isin(regions)].copy()

    # Shared limits (optional)
    if share_limits:
        vmin = float(np.nanmin([dsub[ec_center].min(), dsub[fp_center].min()]))
        vmax = float(np.nanmax([dsub[ec_center].max(), dsub[fp_center].max()]))
        pad = 0.03 * (vmax - vmin) if np.isfinite(vmax - vmin) and vmax > vmin else 0.5
        lims = (vmin - pad, vmax + pad)
    else:
        lims = None


    fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=share_limits, sharey=share_limits)
    axes = axes.ravel()

    for i, region in enumerate(regions):
        ax = axes[i]
        g = dsub[dsub["region"] == region].sort_values("day")

        color = REGION_COLORS.get(str(region), "0.3")
        ax.scatter(g[ec_center], g[fp_center], s=18, alpha=0.55, color=color)

        # 1:1 line (since both are altitudes)
        if lims is not None:
            ax.plot([lims[0], lims[1]], [lims[0], lims[1]], linestyle="--", linewidth=1)
            ax.set_xlim(*lims)
            ax.set_ylim(*lims)
        else:
            vmin_r = float(np.nanmin([g[ec_center].min(), g[fp_center].min()]))
            vmax_r = float(np.nanmax([g[ec_center].max(), g[fp_center].max()]))
            ax.plot([vmin_r, vmax_r], [vmin_r, vmax_r], linestyle="--", linewidth=1)

        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(1))

        ax.set_title(f"{region}", fontweight="bold")
        ax.grid(True, alpha=0.3)

    # Turn off unused panels if <4 regions
    for j in range(len(regions), 4):
        axes[j].axis("off")

    fig.supxlabel("EarthCARE daily averaged altitude [km]")
    fig.supylabel("FLEXPART daily averaged altitude [km]")

    fig.suptitle(
        f"Daily-average scatter by region for sph_{COMPARE_MODE} during {START_DATE} to {END_DATE}",
        fontweight="bold",
        y=0.98,
    )

    out_png = os.path.join(
        plotdir,
        f"scatter_daily_avg_ec_vs_fp_2x2_by_region_{COMPARE_MODE}_{START_DATE}_{END_DATE}.png",
    )
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")



# ============================================================
# GFAS ↔ EarthCARE COMPARISON
# ============================================================

def _find_gfas_ec_match_csvs(folder: str):
    pat = os.path.join(folder, "*_GFAS_EC_height_comparison.csv")
    return sorted(glob.glob(pat))

def _date_from_earthcare_filename(name: str) -> pd.Timestamp:
    """
    Extract YYYYMMDD from something like EC_FLEXPART_GFAS_20250601_T....
    Falls back to NaT.
    """
    m = re.search(r"EC_FLEXPART_GFAS_(\d{8})_", name)
    if m:
        return pd.to_datetime(m.group(1), format="%Y%m%d")
    return pd.NaT

def load_all_gfas_ec_matches(match_dir: str) -> pd.DataFrame:
    files = _find_gfas_ec_match_csvs(match_dir)
    if not files:
        raise FileNotFoundError(f"No *_GFAS_EC_height_comparison.csv found in {match_dir}")

    dfs = []
    for f in files:
        if os.path.getsize(f) == 0:
            print(f"[WARN] Skipping empty match CSV: {f}")
            continue
        try:
            df = pd.read_csv(f)
        except pd.errors.EmptyDataError:
            print(f"[WARN] Skipping unreadable/empty CSV: {f}")
            continue

        # Add day column (prefer explicit date_ymd if present)
        if "date_ymd" in df.columns:
            df["day"] = pd.to_datetime(df["date_ymd"].astype(str), format="%Y%m%d", errors="coerce")
        else:
            df["day"] = _date_from_earthcare_filename(os.path.basename(f))

        df["source_file"] = os.path.basename(f)
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"All match CSVs were empty/unreadable in {match_dir}")

    return pd.concat(dfs, ignore_index=True)


def daily_stats_from_matches(df: pd.DataFrame, diff_col: str, region_col: str = "region") -> pd.DataFrame:
    """
    Daily median/IQR/mean/std of chosen diff column.
    If region_col not present, creates region='all'.
    """
    if region_col not in df.columns:
        df = df.copy()
        df[region_col] = "all"

    d = df.copy()
    d["day"] = pd.to_datetime(d["day"], errors="coerce")
    d[diff_col] = pd.to_numeric(d[diff_col], errors="coerce")
    d = d.dropna(subset=["day", diff_col])

    g = d.groupby(["day", region_col])[diff_col]

    stats = g.agg(
        diff_km_median="median",
        diff_km_p25=lambda x: np.nanpercentile(x, 25),
        diff_km_p75=lambda x: np.nanpercentile(x, 75),
        diff_km_mean="mean",
        diff_km_std="std",
        n="count",
    ).reset_index()

    # Standardize region column name for plotting helpers
    stats = stats.rename(columns={region_col: "region"})
    return stats

def plot_daily_diff_panel(ax, df_daily: pd.DataFrame, title: str,
                          ylabel: str = "Δheight (km)", style: str = "median_iqr"):
    if df_daily.empty:
        ax.set_title(title + " (no data)")
        ax.grid(True, alpha=0.3)
        return

    for region, g in df_daily.groupby("region"):
        g = g.sort_values("day")

        if style == "median_iqr":
            (line,) = ax.plot(g["day"], g["diff_km_median"], linewidth=1.5, label=str(region))
            color = line.get_color()
            ax.scatter(g["day"], g["diff_km_median"], color=color, s=35, alpha=0.9)
            ax.fill_between(g["day"].values, g["diff_km_p25"].values, g["diff_km_p75"].values, alpha=0.2, color=color)
        else:
            (line,) = ax.plot(g["day"], g["diff_km_mean"], linewidth=1.5, label=str(region))
            color = line.get_color()
            ax.scatter(g["day"], g["diff_km_mean"], color=color, s=35, alpha=0.9)
            ax.fill_between(
                g["day"].values,
                (g["diff_km_mean"] - g["diff_km_std"]).values,
                (g["diff_km_mean"] + g["diff_km_std"]).values,
                alpha=0.2,
                color=color,
            )

    ax.axhline(0.0, linewidth=1.0, linestyle="--")
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold", fontsize=12)
    ax.grid(True, alpha=0.3)

def plot_diff_histogram(df: pd.DataFrame, diff_col: str, out_png: str, title: str):
    x = pd.to_numeric(df[diff_col], errors="coerce").dropna().values
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(x, bins=40, alpha=0.85)
    ax.axvline(0.0, linestyle="--", linewidth=1.0)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Δheight (km)")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    _savefig(out_png)


def plot_ec_vs_gfas_scatter(
    df: pd.DataFrame,
    ec_col: str,
    gfas_col: str,
    out_png: str,
    title: str,
    *,
    smoke_col: str = "is_smoke",
):
    """Scatter with EarthCARE on x-axis and GFAS on y-axis.

    If `smoke_col` exists (boolean-ish), smoke points are drawn in a different color.
    """
    x = pd.to_numeric(df[ec_col], errors="coerce")
    y = pd.to_numeric(df[gfas_col], errors="coerce")
    m = np.isfinite(x) & np.isfinite(y)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    if smoke_col in df.columns:
        is_smoke = df[smoke_col].astype(bool).to_numpy()
        m_smoke = m & is_smoke
        m_other = m & (~is_smoke)

        # Non-smoke first so smoke sits on top
        if m_other.any():
            ax.scatter(x[m_other], y[m_other], s=18, alpha=0.55, color="0.5", label="non-smoke")
        if m_smoke.any():
            ax.scatter(x[m_smoke], y[m_smoke], s=18, alpha=0.85, color="tab:red", label="smoke")
        if (m_smoke.any() or m_other.any()):
            ax.legend(frameon=False)
    else:
        ax.scatter(x[m], y[m], s=18, alpha=0.7)

    if m.any():
        vmin = float(np.nanmin([x[m].min(), y[m].min()]))
        vmax = float(np.nanmax([x[m].max(), y[m].max()]))
        ax.plot([vmin, vmax], [vmin, vmax], linestyle="--", linewidth=1)

    ax.set_xlabel("EarthCARE SPH height [km, AMSL]")
    ax.set_ylabel("GFAS height [km, AMSL]")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, alpha=0.3)
    _savefig(out_png)



def _run_gfas_ec_plots():
    dfm = load_all_gfas_ec_matches(GFAS_EC_MATCH_DIR)
    dfm = dfm[dfm["matched"] == True]

    # date window
    t0 = pd.to_datetime(START_DATE)
    t1 = pd.to_datetime(END_DATE)
    dfm["day"] = pd.to_datetime(dfm["day"], errors="coerce")
    dfm = dfm[(dfm["day"] >= t0) & (dfm["day"] <= t1)].copy()

    if DEBUG_SOURCE_FILE is not None:
        dfm = dfm[dfm["source_file"] == DEBUG_SOURCE_FILE].copy()
        print("[DEBUG] rows after source_file filter:", len(dfm))

    # Ensure a region column exists for daily stats
    if "region" not in dfm.columns:
        dfm["region"] = "all"

    plot_dir_gfas = os.path.join(OUTROOT, "gfas_ec_summary")
    os.makedirs(plot_dir_gfas, exist_ok=True)

    pretty_names = {
        "d_top_minus_apt": "sph_top - APT",
        "d_bottom_minus_apb": "sph_bottom - APB",
        "d_ext_minus_mami_assume_agl": "sph_ext - MAMI",
        "d_top_minus_injh_assume_agl": "sph_top - INJH",
    }

    # Map diff column -> (EarthCARE col, GFAS col, label tuple)
    scatter_map = {
        "d_top_minus_apt": ("ec_sph_top_km", "apt_km_agl", ("sph_top", "apt")),
        "d_bottom_minus_apb": ("ec_sph_bottom_km", "apb_km_agl", ("sph_bottom", "apb")),
        "d_ext_minus_mami_assume_agl": ("ec_sph_ext_km", "mami_km_amsl_assume_agl", ("sph_ext", "mami")),
        "d_top_minus_injh_assume_agl": ("ec_sph_top_km", "injh_km_amsl_assume_agl", ("sph_top", "injh")),  
    }

    diff_cols = [
        "d_top_minus_apt",
        "d_bottom_minus_apb",
        "d_ext_minus_mami_assume_agl",
        "d_top_minus_injh_assume_agl",
    ]

    # -----------------------------
    # DAILY DIFF SUBPLOTS: 4x1
    # -----------------------------
    fig1, axes1 = plt.subplots(len(diff_cols), 1, figsize=(8, 9), sharex=True)

    for i, diff_col in enumerate(diff_cols):
        ax = axes1[i]

        print(dfm.columns)

        if diff_col not in dfm.columns:
            ax.set_title(f"{diff_col} (missing column)", fontweight="bold", fontsize=14)
            ax.grid(True, alpha=0.3)
            continue

        df_daily = daily_stats_from_matches(dfm, diff_col, region_col="region")
        ylabel = "EarthCARE - GFAS [km, AGL]"
        title = f"Δheight: {pretty_names.get(diff_col, diff_col)}"

        plot_daily_diff_panel(
            ax,
            df_daily,
            title=title,
            # ylabel=ylabel,
            ylabel=None,
            style="mean_std",  # or "median_iqr"
        )

        # keep the legend only once if you have many regions
        if i != 0 and ax.get_legend() is not None:
            ax.get_legend().remove()

        ax.set_ylim(-2, 12)

    # add the super ylabel (in figure coords)
    fig1.text(
            0.015,          # x position (0 = far left, 1 = far right)
            0.5,            # y position (0 = bottom, 1 = top)
            "EarthCARE - GFAS [km, AGL]",
            va="center",
            rotation="vertical",
            fontweight="bold",
            # fontsize=16,
        )

    # reserve space on the left for that label
    fig1.tight_layout(rect=[0.03, 0.04, 1, 0.95])

    fig1.suptitle(
        # f"GFAS vs EarthCARE daily Δheight during the period {START_DATE} to {END_DATE}",
        f"Daily mean (and std) of the height difference per GFAS and EarthCARE match", #\n during the period {START_DATE} to {END_DATE}",
        fontweight="bold",
        # fontsize=18,
        y=0.995,
    )

    out1 = os.path.join(plot_dir_gfas, f"DAILY_DIFF_ALL_4_{START_DATE}_{END_DATE}.png")
    _savefig(out1)
    print(f"Saved: {out1}")

    # -----------------------------
    # SCATTER SUBPLOTS: 2x2
    # -----------------------------
    fig2, axes2 = plt.subplots(2, 2, figsize=(8, 8))
    axes2 = axes2.ravel()

    # choose whether to filter to matched-only
    df_scatter_base = dfm.copy()
    if "matched" in df_scatter_base.columns:
        df_scatter_base = df_scatter_base[df_scatter_base["matched"] == True].copy()

    for i, diff_col in enumerate(diff_cols):
        ax = axes2[i]

        if diff_col not in scatter_map:
            ax.set_title(f"{diff_col} (no scatter mapping)", fontweight="bold")
            ax.grid(True, alpha=0.3)
            continue

        ec_col, gfas_col, (ec_name, gfas_name) = scatter_map[diff_col]

        if ec_col not in df_scatter_base.columns or gfas_col not in df_scatter_base.columns:
            ax.set_title(f"{diff_col} (missing {ec_col} or {gfas_col})", fontweight="bold")
            ax.grid(True, alpha=0.3)
            continue

        # EarthCARE on X, GFAS on Y
        x = pd.to_numeric(df_scatter_base[ec_col], errors="coerce")
        y = pd.to_numeric(df_scatter_base[gfas_col], errors="coerce")
        m = np.isfinite(x) & np.isfinite(y)

        ax.scatter(x[m], y[m], s=10, alpha=0.35)

        if m.any():
            vmin = float(np.nanmin([x[m].min(), y[m].min()]))
            vmax = float(np.nanmax([x[m].max(), y[m].max()]))
            ax.plot([vmin, vmax], [vmin, vmax], linestyle="--", linewidth=1)

        ax.set_title(pretty_names.get(diff_col, diff_col), fontweight="bold")
        ax.set_xlabel(f"EarthCARE {ec_name} [km, AGL]")
        ax.set_ylabel(f"GFAS {gfas_name} [km, AGL]")

        
        ax.grid(True, alpha=0.3)

        ax.set_xlim(-0.5, 13)
        ax.set_ylim(-0.5, 13)

    fig2.suptitle(
        f"GFAS vs EarthCARE in the period {START_DATE} to {END_DATE}",
        fontweight="bold",
        # fontsize=18,
        y=0.995,
    )
    fig2.tight_layout()
    out2 = os.path.join(plot_dir_gfas, f"SCATTER_ALL_4_{START_DATE}_{END_DATE}.png")
    fig2.savefig(out2)
    plt.close(fig2)
    print(f"Saved: {out2}")

    print(f"Saved GFAS↔EarthCARE multi-panel plots to: {plot_dir_gfas}")




# ============================================================
# MAIN
# ============================================================

def main():
    global OUTROOT, FILE_PREFIX, START_DATE, END_DATE, COMPARE_MODE

    if DEBUG_TRACK:
        FILE_PREFIX = "debug_track_"

        if DEBUG_RUN_TAG:
            OUTROOT = os.path.join(OUTDIR, "debug_track", DEBUG_RUN_TAG)
        else:
            OUTROOT = _pick_latest_debug_outroot(OUTDIR)

        manifest = _load_manifest_if_present(OUTROOT, FILE_PREFIX)
        if manifest is not None:
            # auto-sync plot settings to the debug run
            START_DATE = manifest.get("START_DATE", START_DATE)
            END_DATE = manifest.get("END_DATE", END_DATE)
            COMPARE_MODE = manifest.get("COMPARE_MODE", COMPARE_MODE)

        print(f"[DEBUG_TRACK] Using OUTROOT: {OUTROOT}")
        print(f"[DEBUG_TRACK] Using prefix: {FILE_PREFIX}")
        print(f"[DEBUG_TRACK] Window: {START_DATE} → {END_DATE} | mode={COMPARE_MODE}")
    else:
        OUTROOT = OUTDIR
        FILE_PREFIX = ""

    plotdir = out_plot_dir()

    df_daily_fp = _load_csv(f_daily_fp())
    df_daily_ec = _load_csv(f_daily_ec())

    print(f"Loaded daily FP: {len(df_daily_fp)} rows")
    print(f" |     daily EC: {len(df_daily_ec)} rows")
    print(f"daily FP columns: {df_daily_fp.columns.tolist()}")
    print(f"daily EC columns: {df_daily_ec.columns.tolist()}") 

    df_daily_paired = _load_csv(f_daily_paired_diff()) if os.path.exists(f_daily_paired_diff()) else pd.DataFrame()

    df_period_fp = _load_csv(f_period_fp()) if os.path.exists(f_period_fp()) else pd.DataFrame()
    df_period_ec = _load_csv(f_period_ec()) if os.path.exists(f_period_ec()) else pd.DataFrame()
    df_period_paired = _load_csv(f_period_paired_diff()) if os.path.exists(f_period_paired_diff()) else pd.DataFrame()

    if MAKE_SCATTER_DAILY_AVG_2X2:
        _plot_scatter_daily_avg_2x2_by_region(
            df_daily_ec=df_daily_ec,
            df_daily_fp=df_daily_fp,
            regions=None,
            max_regions=4,
            share_limits=True,
            style=DAILY_STYLE,   # uses mean if "mean_std", median if "median_iqr"
        )

        # Load pairs once if needed
    df_pairs = pd.DataFrame()
    if (MAKE_DAILY_FP_EC_SUBPLOT_FROM_PAIRS or MAKE_SCATTER_PAIRED) and os.path.exists(f_pairs_points()):
        df_pairs = _load_csv(f_pairs_points())

    # ---------------------------------------------------------
    # If requested: build "daily FP/EC subplot" from paired points
    # ---------------------------------------------------------
    if MAKE_DAILY_FP_EC_SUBPLOT_FROM_PAIRS:
        if df_pairs.empty:
            raise FileNotFoundError(f"Pairs file missing/empty: {f_pairs_points()}")

        df_daily_fp_pairs, df_daily_ec_pairs = _daily_stats_from_pairs(
            df_pairs,
            fp_col="fp_km",
            ec_col="ec_km_at_fp_time",
            use_smoke_only=only_smoke_for_tc_sph,  # or False if you want paired-subplot always include all
            smoke_col="is_smoke",
        )

        # Now plot using the SAME code path as before,
        # just swapping in our computed daily frames:
        fig = plt.figure(figsize=(8, 10))

        outer = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.25)
        gs_fp = outer[0].subgridspec(2, 1, height_ratios=[3.0, 1.0], hspace=0.05)
        gs_ec = outer[1].subgridspec(2, 1, height_ratios=[3.0, 1.0], hspace=0.05)

        ax_fp   = fig.add_subplot(gs_fp[0, 0])
        ax_fp_n = fig.add_subplot(gs_fp[1, 0], sharex=ax_fp)
        ax_ec   = fig.add_subplot(gs_ec[0, 0], sharex=ax_fp)
        ax_ec_n = fig.add_subplot(gs_ec[1, 0], sharex=ax_fp)

        _plot_daily_panel(
            ax_fp, df_daily_fp_pairs,
            title=f"FLEXPART daily average (+ std) \n for sph_{COMPARE_MODE} and sample counts",
            ylabel="Altitude [km]",
            style=DAILY_STYLE,
            ylim=YLIM_ALT_KM,
        )
        _plot_daily_panel(
            ax_ec, df_daily_ec_pairs,
            title=f"EarthCARE daily average (+ std) \n for sph_{COMPARE_MODE} at matched FP time and sample counts",
            ylabel="Altitude [km]",
            style=DAILY_STYLE,
            ylim=YLIM_ALT_KM,
        )

        _plot_counts_panel(ax_fp_n, df_daily_fp_pairs, title=None, logy=False)
        _plot_counts_panel(ax_ec_n, df_daily_ec_pairs, title=None, logy=False)

        fig.align_ylabels([ax_fp, ax_fp_n, ax_ec, ax_ec_n])
        ax_fp.tick_params(axis="x", labelbottom=False)
        ax_ec.tick_params(axis="x", labelbottom=False)

        handles, labels = ax_fp.get_legend_handles_labels()
        if handles:
            ax_fp.legend(handles, labels, ncol=4, fontsize=10, loc="upper center")

        out_png = os.path.join(
            plotdir,
            f"daily_FP_EC_FROM_PAIRS_WITH_COUNTS_{COMPARE_MODE}_{START_DATE}_{END_DATE}.png",
        )
        _savefig(out_png)
        print(f"Saved: {out_png}")



    if MAKE_DAILY_FP_EC_SUBPLOT_FROM_PAIRS_JUST_ONE_N_POINTS_PANEL:
        if df_pairs.empty:
            raise FileNotFoundError(f"Pairs file missing/empty: {f_pairs_points()}")

        df_daily_fp_pairs, df_daily_ec_pairs = _daily_stats_from_pairs(
            df_pairs,
            fp_col="fp_km",
            ec_col="ec_km_at_fp_time",
            use_smoke_only=only_smoke_for_tc_sph,
            smoke_col="is_smoke",
        )

        # --- Figure layout ---
        fig = plt.figure(figsize=(8, 10))

        # 3 rows now: FP alt, EC alt, shared counts
        gs = fig.add_gridspec(
            3, 1,
            height_ratios=[3.0, 3.0, 1.2],
            hspace=0.17,
        )

        ax_fp   = fig.add_subplot(gs[0, 0])
        ax_ec   = fig.add_subplot(gs[1, 0], sharex=ax_fp)
        ax_n    = fig.add_subplot(gs[2, 0], sharex=ax_fp)

        # --- Altitude panels ---
        _plot_daily_panel(
            ax_fp,
            df_daily_fp_pairs,
            title=f"FLEXPART daily average (+ std) for sph_{COMPARE_MODE}",
            ylabel="Altitude [km]",
            style=DAILY_STYLE,
            ylim=YLIM_ALT_KM,
        )

        _plot_daily_panel(
            ax_ec,
            df_daily_ec_pairs,
            title=f"EarthCARE daily average (+ std) for sph_{COMPARE_MODE} at matched FP time",
            ylabel="Altitude [km]",
            style=DAILY_STYLE,
            ylim=YLIM_ALT_KM,
        )

        # --- Single shared counts panel ---
        # Use FP counts (should be identical to EC if paired correctly)
        _plot_counts_panel(
            ax_n,
            df_daily_fp_pairs,
            title=None,
            logy=False,
        )

        # --- Formatting ---
        fig.align_ylabels([ax_fp, ax_ec, ax_n])

        ax_fp.tick_params(axis="x", labelbottom=False)
        ax_ec.tick_params(axis="x", labelbottom=False)

        handles, labels = ax_fp.get_legend_handles_labels()
        if handles:
            ax_fp.legend(handles, labels, ncol=4, fontsize=10, loc="upper center")

        out_png = os.path.join(
            plotdir,
            f"daily_FP_EC_FROM_PAIRS_WITH_COUNTS_one_number_of_points_panel_{COMPARE_MODE}_{START_DATE}_{END_DATE}.png",
        )
        _savefig(out_png)
        print(f"Saved: {out_png}")


    # ---------------------------------------------------------
    # Otherwise plot the original daily CSV version
    # ---------------------------------------------------------
    if MAKE_DAILY_FP_EC_SUBPLOT:
        fig = plt.figure(figsize=(8, 10))

        outer = fig.add_gridspec(
            2, 1,
            height_ratios=[1, 1],
            hspace=0.25  # <-- BIG gap between FP and EC blocks
        )

        gs_fp = outer[0].subgridspec(
            2, 1,
            height_ratios=[3.0, 1.0],
            hspace=0.05  # <-- SMALL gap between FP altitude and FP counts
        )

        gs_ec = outer[1].subgridspec(
            2, 1,
            height_ratios=[3.0, 1.0],
            hspace=0.05  # <-- gap between EC altitude and EC counts
        )

        ax_fp   = fig.add_subplot(gs_fp[0, 0])
        ax_fp_n = fig.add_subplot(gs_fp[1, 0], sharex=ax_fp)
        ax_ec   = fig.add_subplot(gs_ec[0, 0], sharex=ax_fp)
        ax_ec_n = fig.add_subplot(gs_ec[1, 0], sharex=ax_fp)


        # --- altitude panels (unchanged logic)
        _plot_daily_panel(
            ax_fp, df_daily_fp,
            title=f"FLEXPART daily averaged sph_{COMPARE_MODE} altitudes and sample counts",
            ylabel="Altitude [km]",
            style=DAILY_STYLE,
            ylim=YLIM_ALT_KM,
        )
        _plot_daily_panel(
            ax_ec, df_daily_ec,
            title=f"EarthCARE daily averaged sph_{COMPARE_MODE} altitudes and sample counts",
            ylabel="Altitude [km]",
            style=DAILY_STYLE,
            ylim=YLIM_ALT_KM,
        )

        # --- count panels
        _plot_counts_panel(ax_fp_n, df_daily_fp, ec_or_fp="fp", title=None, logy=False)
        _plot_counts_panel(ax_ec_n, df_daily_ec, ec_or_fp="fp", title=None, logy=False)

        # Align y-labels and remove x-labels from upper panels
        fig.align_ylabels([ax_fp, ax_fp_n, ax_ec, ax_ec_n])
        ax_fp.tick_params(axis="x", labelbottom=False)
        ax_ec.tick_params(axis="x", labelbottom=False)


        # One legend for the whole figure (use the altitude panel handles)
        handles, labels = ax_fp.get_legend_handles_labels()
        if handles:
            ax_fp.legend(handles, labels, ncol=4, fontsize=10, loc="upper center")


        out_png = os.path.join(plotdir, f"daily_FP_EC_point_based_WITH_COUNTS_{COMPARE_MODE}_{START_DATE}_{END_DATE}.png")
        _savefig(out_png)
        print(f"Saved: {out_png}")




    if MAKE_DAILY_PAIRED_DIFF and not df_daily_paired.empty:
        fig, ax = plt.subplots(figsize=(13, 4.8))
        _plot_daily_panel(
            ax, df_daily_paired,
            title="Daily paired difference (FP - EC at FP times)",
            ylabel="FP - EC (km)",
            style=DAILY_STYLE,
            ylim=YLIM_DIFF_KM,
        )
        ax.legend(ncol=4, fontsize=10, loc="upper center")
        out_png = os.path.join(plotdir, f"daily_paired_fp_minus_ec_{COMPARE_MODE}_{START_DATE}_{END_DATE}.png")
        _savefig(out_png)
        print(f"Saved: {out_png}")

    if MAKE_PERIOD_SUMMARY_PLOT and (not df_period_fp.empty) and (not df_period_ec.empty):
        _plot_period_summary(df_period_fp, df_period_ec, df_period_paired if not df_period_paired.empty else None)

    if MAKE_SCATTER_PAIRED and os.path.exists(f_pairs_points()):
        df_pairs = _load_csv(f_pairs_points())
        _plot_scatter_from_pairs_2x2_by_region(df_pairs, regions=None, max_regions=4, share_limits=True)


    # ============================================================
    # GFAS ↔ EarthCARE comparison plots 
    # ============================================================
    if PLOT_GFAS_EC:
        _run_gfas_ec_plots()


if __name__ == "__main__":
    main()