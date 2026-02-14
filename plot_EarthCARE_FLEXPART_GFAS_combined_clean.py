import os, glob, re, warnings, h5py
import numpy as np
import xarray as xr
import pandas as pd
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from pyproj import Geod
from netCDF4 import num2date
from cartopy import crs as ccrs
from cartopy.io import DownloadWarning
from collections import defaultdict
from scipy.interpolate import interp1d
from matplotlib.colors import LogNorm, Normalize, ListedColormap, BoundaryNorm
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

warnings.simplefilter("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="facecolor will have no effect*", category=UserWarning)
warnings.filterwarnings("ignore", message="The input coordinates to pcolormesh*", category=UserWarning)
warnings.filterwarnings("ignore", message="set_ticklabels() should only be used*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*set_ticklabels.*FixedLocator.*")
warnings.filterwarnings("ignore", category=DownloadWarning)
plt.ioff()


def _days_in_flexpart_file(nc_path):
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

def format_ticks(datetime, localtime):
    ticks = pd.date_range(datetime.min().ceil('1min'), datetime.max().floor('1min'), freq='1min')
    lticks = pd.date_range(localtime.min().ceil('1min'), localtime.max().floor('1min'), freq='1min')
    indices = np.clip(np.searchsorted(datetime, ticks), 1, len(datetime) - 1)
    indices -= (ticks - datetime[indices - 1] < datetime[indices] - ticks)
    return ticks, lticks, indices

def setup_ax_labels(ax, time_ticks, local_ticks, datetime, lat, lon, indices, y_offset):
    ax.set_xticks(time_ticks)
    ax.set_xticklabels([f"{t:%H:%M}\n{lt:%H:%M}" for t, lt in zip(time_ticks, local_ticks)], fontsize=15)
    ax.set_xlim(datetime[0], datetime[-1])
    ax11 = ax.twiny()
    ax11.set_xlim(ax.get_xlim())
    latlon = [f"${la:.1f}^\\circ${'S' if la<0 else 'N'}\n${lo:.1f}^\\circ${'E' if lo<0 else 'E'}"
              for la, lo in zip(lat[indices], lon[indices])]

    ax11.set_xticks(time_ticks)
    ax11.set_xticklabels(latlon, fontsize=15, ha='right')

def _list_groups_with_var(nc_path, varname="spec001_mr"):
    groups = []
    with h5py.File(nc_path, "r") as f:
        def _visit(name, obj):
            if isinstance(obj, h5py.Dataset) and name.endswith("/" + varname):
                grp = "/" + "/".join(name.split("/")[:-1]).strip("/")
                groups.append(grp if grp != "/" else "")
        f.visititems(lambda n, o: _visit(n, o))
    # unique, preserve order
    seen, uniq = set(), []
    for g in groups:
        if g not in seen:
            uniq.append(g); seen.add(g)
    return uniq

def _median_timestep_hours(ds):
    if "time" not in ds:
        return float("inf")
    t = pd.to_datetime(ds["time"].values)
    if t.size < 2:
        return float("inf")
    dt = (np.diff(t).astype("timedelta64[s]").astype(float)) / 3600.0
    return float(np.median(dt))

def open_flexpart_dataset(nc_path, preferred=("3h","3hour","3H")):
    # legacy single-file with variables at top level
    try:
        ds_try = xr.open_dataset(nc_path, decode_timedelta=True)
        if "spec001_mr" in ds_try:
            return ds_try
        ds_try.close()
    except Exception:
        pass

    # Find groups containing 'spec001_mr'
    groups = _list_groups_with_var(nc_path, "spec001_mr")
    if not groups:
        raise RuntimeError(f"No groups with 'spec001_mr' found in {nc_path}")

    # pick preferred group if present
    grp = None
    if preferred:
        for cand in (preferred if isinstance(preferred, (list, tuple)) else (preferred,)):
            for g in groups:
                if cand.lower() in g.lower():
                    grp = g; break
            if grp: break

    # otherwise pick group with timestep closest to 3h
    if not grp:
        best = (None, float("inf"))
        for g in groups:
            try:
                dsg = xr.open_dataset(nc_path, group=g, decode_timedelta=True)
                score = abs(_median_timestep_hours(dsg) - 3.0)
                if score < best[1]:
                    best = (g, score)
                dsg.close()
            except Exception:
                pass
        grp = best[0] or groups[0]

    ds_fp = xr.open_dataset(nc_path, group=grp, decode_timedelta=True)

    # light renames if needed
    rename_map = {}
    if "lon" in ds_fp and "longitude" not in ds_fp: rename_map["lon"] = "longitude"
    if "lat" in ds_fp and "latitude" not in ds_fp: rename_map["lat"] = "latitude"
    if rename_map:
        ds_fp = ds_fp.rename(rename_map)
    return ds_fp



def plot_subplotted_flexpart_earthcare_class_ebd_and_GFAS(
                        flexpart_nc,          
                        earthcare_file,
                        output_dir,
                        ebd_file=None,
                        gfas_folder=None,     
                        gfas_variant="01"
                        ):
    os.makedirs(output_dir, exist_ok=True)

    # --- Skipping if PNG already exists ---
    short_name_fp = os.path.splitext(os.path.basename(flexpart_nc))[0]
    short_name_ec = os.path.splitext(os.path.basename(earthcare_file))[0]
    # out_name = f"combined_earthcare_flexpart_{short_name_fp}__{short_name_ec}.png"
    out_name = re.sub(r".*_(\d{8})T(\d{6})Z_(\d{8})T(\d{6})Z_([A-Za-z0-9]+).*",
                      r"EC_FLEXPART_GFAS_\1_T\2Z_T\4Z_\5.png",
                      os.path.basename(earthcare_file)
                    )  

    out_png = os.path.join(output_dir, out_name)
    if not globals().get("FORCE_REPLOT", True) and os.path.exists(out_png): 
        print(f"Skipping {short_name_ec} already plotted")
        return
    

    # --- Load FLEXPART ---
    try:
        ds_fp = xr.open_dataset(flexpart_nc, decode_timedelta=True).squeeze(drop=True)
    except Exception as e:
        print(f"Could not open FLEXPART NetCDF {os.path.basename(flexpart_nc)}: {e}")
        return

    z_agl_m = ds_fp["height"]

    # --- Layer thickness (htdiff) from Sabine ---
    # Prefer FLEXPART's layer interfaces if available; otherwise fall back to mid-point edges
    if "outheight" in ds_fp:
        # outheight is expected to be the interfaces [m AGL], length = nlev + 1
        oh = ds_fp["outheight"]
        dim_oh = oh.dims[0]
        dz = oh.diff(dim_oh)  # thickness per layer [m]
        # Make sure dz uses the same dim/coords as the mid-level "height"
        dz = dz.rename({dim_oh: "height"}).assign_coords(height=ds_fp["height"])
    else:
        # Fallback: build edges from mid-levels, then take diff (what you had)
        z_agl_m = ds_fp["height"]
        z_agl_m = z_agl_m - 500        # xxx er dette for cheecky??

    # Keep all FLEXPART times; just wrap longitudes
    lon_wrapped = ((ds_fp["longitude"] + 180) % 360) - 180
    c_all = ds_fp["spec001_mr"].assign_coords(longitude=lon_wrapped).sortby("longitude")

    # -------- Load EarthCARE ATL_TC  ----------
    with h5py.File(earthcare_file, "r") as f:
        ds = f['ScienceData']
        lat_ec = np.array(ds['latitude'])
        lon_ec = np.array(ds['longitude'])
        lon_ec_wrapped = ((lon_ec + 180) % 360) - 180

        dt = pd.to_datetime(
            num2date(ds['time'], 'seconds since 2000-01-01').astype('datetime64[us]')
        )
        lt = dt + pd.to_timedelta(np.round(lon_ec / 15 * 3600).astype(int), unit='s')

        # elev = 0.001 * (np.array(ds['elevation']) - ds['geoid_offset'][()])  # km AMSL surface
        # --- Surface height from ECMWF geopotential along the track (replaces EarthCARE 'elev') ---
        # NOTE: 'z' in geop_surf_05.nc is geopotential [m^2 s^-2]; height [m] = z / g0
        geop_nc = r"\\extfile.nilu.no\xnilu_wrk\users\sec\kleinprojekte\kerstin_wildfire\geop_surf_05.nc"
        try:
            ds_geop = xr.open_dataset(geop_nc).squeeze(drop=True)
            # wrap longitudes to [-180, 180] to match the rest
            lon_geop_wrapped = ((ds_geop["longitude"] + 180) % 360) - 180
            ds_geop = ds_geop.assign_coords(longitude=lon_geop_wrapped).sortby("longitude")

            # geopotential -> height
            inv_g = 1.0 / 9.80665  # m s^2 / m^2  (multiply geopotential by this to get meters)
            h_surf_m = (ds_geop["z"] * inv_g).interp(
                latitude=xr.DataArray(lat_ec, dims=["track"]),
                longitude=xr.DataArray(lon_ec_wrapped, dims=["track"]),
                method="nearest"
            ).to_numpy()

            elev = 0.001 * h_surf_m.astype(float)  # km AMSL surface, 1D along track (size = n_track)

        except Exception as e:
            print(f"Warning: could not read/convert geopotential from {geop_nc}: {e}")
            elev = np.zeros_like(lat_ec, dtype=float)  # safe fallback: 0 km



        tropo = 0.001 * np.array(ds['tropopause_height'])
        geoid_offset = np.array(ds['geoid_offset'])
        height = 0.001 * (np.array(ds['height'])[::-1] - geoid_offset[:, np.newaxis])  # km AMSL, shape (time, z)
        dt_ext = np.tile(dt.to_numpy()[:, None], (1, height.shape[1]))
        class_data = np.array(ds['classification_low_resolution'])


    # -------- make a coarse surface for shifting FLEXPART -------- xxx double check if this is correct
    dlat = float(np.median(np.diff(ds_fp["latitude"].values)))
    dlon = float(np.median(np.diff(ds_fp["longitude"].values)))
    mean_lat = float(np.nanmean(lat_ec))
    fp_dx_km = 111.0 * np.hypot(dlat, dlon * np.cos(np.deg2rad(mean_lat)))
    geod = Geod(ellps="WGS84")
    seg_km = np.r_[0.0,
                geod.inv(lon_ec[:-1], lat_ec[:-1],
                            lon_ec[ 1:], lat_ec[ 1:])[2] / 1000.0]
    km_per_sample = max(1e-6, float(np.median(seg_km[1:])))

    # choose a smoothing length comparable to the FLEXPART grid
    window_km = max(2.0 * fp_dx_km, 80.0)   
    win = max(3, int(round(window_km / km_per_sample)))
    box = np.ones(win, dtype=float) / win

    # coarse surface used for shifting
    elev_coarse = np.convolve(elev, box, mode="same")


    # Use the EarthCARE track midpoint as the overpass time
    overpass_time = pd.to_datetime(dt.to_numpy()[len(dt)//2])
    c_3d = c_all.sel(time=overpass_time, method='nearest') if 'time' in c_all.dims else c_all

    # If spec001_mr already has no 'height' dimension, treat it as a ready-made column
    if 'height' in c_3d.dims:
        col = (c_3d * z_agl_m).sum("height") / 1_000.0
        col.attrs["units"] = "μg m⁻²"
    else:
        col = c_3d  # already a 2-D column (lat, lon)


    # Parse baseline code + convenient date strings
    ec_base = os.path.basename(earthcare_file)
    baseline = ec_base[6:8]
    frame = ec_base[-4]

    # Use EarthCARE overpass time as the reference date
    ts = pd.to_datetime(overpass_time)
    date = f"{ts.day}-{ts.strftime('%b').lower()}-{ts.year}"
    date_ymd = ts.strftime("%Y%m%d")


    # -------- Sample FLEXPART 3D field along EarthCARE track ----------
    track = xr.DataArray(np.arange(len(lat_ec)), dims=["track"])
    lat_da = xr.DataArray(lat_ec, dims=["track"])
    lon_da = xr.DataArray(lon_ec_wrapped, dims=["track"])

    c_track = c_3d.interp(latitude=lat_da, longitude=lon_da, method="linear")
    fp_at_earthcare_track = np.asarray(c_track)
    fp_at_earthcare_track = np.ma.masked_where(~np.isfinite(fp_at_earthcare_track) | (fp_at_earthcare_track <= 50),
                                               fp_at_earthcare_track)


    # -------- Vertical interpolation onto EC height grid ----------
    z_agl_km = (z_agl_m.values.astype(float)) / 1000.0
    n_track = len(lat_ec)
    fp_interp = np.full_like(height, np.nan)
    for i in range(n_track):
        h_ec = height[i, :]
        shifted_altitudes = z_agl_km + elev_coarse[i]
        order = np.argsort(shifted_altitudes)
        fz = interp1d(shifted_altitudes[order],
                      fp_at_earthcare_track[order, i],
                      bounds_error=False, fill_value=np.nan)
        fp_interp[i, :] = fz(h_ec)
    fp_interp = np.ma.masked_where((fp_interp < 50) | ~np.isfinite(fp_interp), fp_interp)


    # -------- EBD extinction ----------
    ebd_ext = ebd_height_km = ebd_dt = None
    
    with h5py.File(ebd_file, "r") as f_ebd:
        ds_ebd = f_ebd['ScienceData']
        time_1d = np.array(ds_ebd['time'])
        ebd_dt = pd.to_datetime(
            num2date(time_1d, 'seconds since 2000-01-01').astype('datetime64[us]')
        )

        # --- choose variable by desired RESOLUTION (with graceful fallback) ---
        def _choose_ebd_variable(ds_ebd, resolution):
            candidates_by_res = {
                "high":   ["particle_extinction_coefficient_355nm",
                            "particle_extinction_coefficient_355nm_high_resolution",
                            "extinction_high_resolution"],
                "medium": ["particle_extinction_coefficient_355nm_medium_resolution",
                            "extinction_medium_resolution"],
                "low":    ["particle_extinction_coefficient_355nm_low_resolution",
                            "extinction_low_resolution"],
            }
            search = candidates_by_res.get(str(resolution).lower(), [])
            search += candidates_by_res["high"] + candidates_by_res["medium"] + candidates_by_res["low"]
            seen = set()
            ordered = [v for v in search if not (v in seen or seen.add(v))]
            for var in ordered:
                if var in ds_ebd:
                    try:
                        return np.array(ds_ebd[var])
                    except Exception:
                        pass
            return None

        ext = _choose_ebd_variable(ds_ebd, globals().get("RESOLUTION", "high"))
        if ext is None:  # fallback to old default if present
            ext = np.array(ds_ebd['particle_extinction_coefficient_355nm'])

        # Build height grid
        if 'height' in ds_ebd and 'geoid_offset' in ds_ebd:
            geoid_offset_ebd = np.array(ds_ebd['geoid_offset'])
            height_2d = 0.001 * (np.array(ds_ebd['height'])[::-1] - geoid_offset_ebd[:, np.newaxis])
            ebd_height_km = np.mean(height_2d, axis=0)
        else:
            ebd_height_km = np.array(ds_ebd.get('JSG_height', [])) * 0.001

        # --- ensure extinction is oriented (time, z) ---
        if ext.ndim == 2 and ebd_height_km is not None and ebd_dt is not None:
            if ext.shape[0] == len(ebd_height_km) and ext.shape[1] == len(ebd_dt):
                ext = ext.T  # make (time, z)

        # --- SIMPLE QUALITY FILTER USING 'quality_status' ---
        q_keep = set(globals().get("QUALITY_FLAGS", (0, 1)))
        qf = np.array(ds_ebd.get('quality_status', []))
        if qf.size:
            ext_shape = np.shape(ext)
            # adapt qf to ext shape
            if qf.shape == ext_shape[::-1]:
                qf = qf.T
            elif qf.shape == (ext_shape[0],):
                qf = qf[:, None]
            elif qf.shape == (ext_shape[1],):
                qf = qf[None, :]
            # if dtype not integer, cast
            if not np.issubdtype(qf.dtype, np.integer):
                qf = qf.astype(np.int32, copy=False)
            if qf.shape == ext.shape:
                keep = np.isin(qf, list(q_keep))
                n_total = np.isfinite(ext).sum()
                ext = np.where(keep, ext, np.nan)
                n_kept = np.isfinite(ext).sum()
            else:
                print("'quality_status' shape not compatible with extinction; skipping QA filter.")
        else:
            print("'quality_status' not found; skipping QA filter.")

        ebd_ext = np.ma.masked_invalid(ext)



    # ===================== PLOTTING =====================
    fig = plt.figure(figsize=(36, 26))
    gs = GridSpec(41, 9, figure=fig)

    plt.rcParams.update({
        "axes.titlesize": 28,
        "axes.labelsize": 24,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
        "figure.titlesize": 30
    })

    # --- FLEXPART map ---
    ax_top = fig.add_subplot(gs[0:10, :], projection=ccrs.PlateCarree())
    ax_top.set_aspect('auto')
    ax_top.set_facecolor("none")

    ax_top.coastlines(linewidth=0.8, color='black')
    ax_top.add_feature(
        cfeature.LAND.with_scale("50m"),
        facecolor="white", edgecolor='none', zorder=0
    )
    ax_top.add_feature(
        cfeature.OCEAN.with_scale("50m"),
        facecolor=(0.6, 0.8, 1.0, 0.3),
        edgecolor='none', zorder=0
    )
    ax_top.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.5, edgecolor="black")

    gl = ax_top.gridlines(draw_labels=True, linestyle='--', linewidth=0.6, alpha=1)
    for attr in ('top_labels', 'xlabels_top'):
        if hasattr(gl, attr): setattr(gl, attr, False)
    for attr in ('right_labels', 'ylabels_right'):
        if hasattr(gl, attr): setattr(gl, attr, False)
    for attr in ('left_labels', 'ylabels_left'):
        if hasattr(gl, attr): setattr(gl, attr, True)
    for attr in ('bottom_labels', 'xlabels_bottom'):
        if hasattr(gl, attr): setattr(gl, attr, True)
    gl.xlabel_style = {'size': 14}
    gl.ylabel_style = {'size': 14}

    ax_top.set_extent([-180, 180, 30, 90], crs=ccrs.PlateCarree())
    ax_top.autoscale(False)

    threshold_map = 200.0  # μg m^-2
    cmap_obj = plt.colormaps.get_cmap('plasma_r').copy()
    cmap_obj.set_under("white")

    mesh = ax_top.pcolormesh(
        col["longitude"], col["latitude"], col.where(col >= threshold_map),
        transform=ccrs.PlateCarree(), shading="auto",
        cmap=cmap_obj, vmin=threshold_map
    )
    
    # choose color based on frame
    frame_colors = {"D": "red", "B": "blue", "C": "black"}
    track_color = frame_colors.get(frame, "gray")  # default gray if unexpected letter
    track_name = {"D": "N -> S", "B": "S -> N", "C": "Cross Polar"}

    # plot with chosen color
    ax_top.plot(lon_ec, lat_ec, color=track_color, linewidth=2,
        transform=ccrs.Geodetic(), label=f"EarthCARE Track ({track_name[frame]})")
    ax_top.legend(loc='upper right')
    ax_top.set_title(f"Comparison of FLEXPART Black Carbon, GFAS FRP, and EarthCARE ATL Products on {date}", fontweight='bold', fontsize=28)

    # --- GFAS overlay (daily FRP) ---
    if gfas_folder is not None:
        gfas_path = os.path.join(gfas_folder, f"BC_GFAS_{gfas_variant}_{date_ymd}.nc")
        try:
            ds_gfas = xr.open_dataset(gfas_path, decode_times=True)  # daily file (time=1)

            # Wrap longitudes to [-180, 180] if needed
            if (ds_gfas['longitude'] >= 0).all():
                ds_gfas = ds_gfas.assign_coords(
                    longitude=((ds_gfas['longitude'] + 180) % 360) - 180
                ).sortby('longitude')

            # Subset to NH like before
            ds_gfas_sub = ds_gfas.sel(latitude=slice(90, 30), longitude=slice(-180, 180))

            # Daily FRP (drop singleton time to get 2D field)
            frp = ds_gfas_sub['frpfire']
            frp2d = frp.isel(time=0) if ('time' in frp.dims and frp.sizes.get('time', 1) == 1) else frp.squeeze()

            # Prepare for scatter like before
            vals = frp2d.values.ravel()
            lon2d, lat2d = np.meshgrid(ds_gfas_sub['longitude'].values,
                                    ds_gfas_sub['latitude'].values)
            lon_flat = lon2d.ravel(); lat_flat = lat2d.ravel()

            bins   = [0.1, 5, 10, 50, 100, np.inf]  # W m^-2
            colors = ["#ff6e6e", "#ff0000", "#b62020", "#800000", "#000000"]
            labels = ['0.1–5', '5–10', '10–50', '50–100', '>100']

            mask_valid = vals > 0.1
            for j in range(len(bins)-1):
                mask_bin = (vals > bins[j]) & (vals <= bins[j+1]) & mask_valid
                if np.any(mask_bin):
                    ax_top.scatter(lon_flat[mask_bin], lat_flat[mask_bin],
                                color=colors[j], s=10, label=labels[j],
                                transform=ccrs.PlateCarree(), zorder=3, linewidths=0)

            # Legend (simple daily label)
            earthcare_handle = Line2D([0], [0], color=track_color, linewidth=2, label=f'EarthCARE Track {track_name[frame]}')
            spacer_handle = Line2D([], [], linestyle='none', marker=None, label='')
            frp_title_handle = Line2D([0], [0], linestyle='none', marker=None, label='FRP (W m$^{-2}$)')
            bin_handles = [
                Line2D([0], [0], linestyle='none', marker='o', markersize=6,
                    color=colors[j], label=labels[j])
                for j in range(len(labels))
            ]
            ordered_handles = [earthcare_handle, spacer_handle, frp_title_handle] + bin_handles
            legend2 = ax_top.legend(handles=ordered_handles, loc='lower right',
                                    fontsize=18, frameon=True, title=None,
                                    handlelength=2, scatterpoints=1,
                                    borderpad=0.8, labelspacing=0.5)
            legend2.get_frame().set_alpha(0.9)

        except FileNotFoundError:
            print(f"No GFAS file for {date_ymd} at {gfas_path}; skipping GFAS dots.")
        except Exception as e:
            print(f"GFAS overlay failed for {date_ymd}: {e}")

    # Colorbar for top map
    cax_top = fig.add_axes([ax_top.get_position().x1 + 0.01,
                            ax_top.get_position().y0, 0.015,
                            ax_top.get_position().height])
    
    cbar_top = plt.colorbar(mesh, cax=cax_top)
    cbar_top.set_label("[μg m⁻²]" + "\n \n \n \n \n \n" + r"$\mathbf{FLEXPART}$ $\mathbf{Total}$ "+ "\n" + r"$\mathbf{Column}$ $\mathbf{BC}$", labelpad=10, fontsize=24)

    # --- Custom classification colormap/norm ---
    u = np.array([-3,-2,-1,0,1,2,3,10,11,12,13,14,15,20,21,22,25,26,27])
    bounds = np.concatenate(([u.min() - 1], u[:-1] + np.diff(u) / 2., [u.max() + 1]))
    class_norm = BoundaryNorm(bounds, len(bounds) - 1)
    tick_centers = 0.5 * (bounds[:-1] + bounds[1:])
    class_labels = [
        'Missing data', 'Surface or sub-surface', 'Noise in Mie & Ray channels', 'Clear',
        '(Warm) Liquid cloud', '(Supercooled) Liquid cloud', 'Ice cloud', 'Dust', 'Sea salt',
        'Continental pollution', 'Smoke', 'Dusty smoke', 'Dusty mix', 'STS', 'NAT',
        'Stratospheric ice', 'Stratospheric ash', 'Stratospheric sulfate', 'Stratospheric smoke'
    ]
    class_cmap = ListedColormap([
        "#f5f5dc", "#000000", "#c4c4c4", "#ffffff", "#5555fe", "#0018e0", "#009bce",
        "#a22f2e", "#acd6e2", "#02fd7f", "#2f504e", "#996633", "#e0b98b", "#ff00ff",
        "#8f37df", "#4a0061", "#fece05", "#fdfb0d", "#c3b27f"
    ])

    # --- FLEXPART cross-section  ---
    ax1 = fig.add_subplot(gs[12:21, :])
    cmap_fp = plt.get_cmap('plasma_r')
    cmap_fp.set_bad("white")
    norm_fp = Normalize(0, 1000)

    # --------- FLEXPART cross-section on the EarthCARE height grid ----------
    # Use the same (time, z) grid for X, Y, and C
    h_for_plot = height          # shape (time, z)
    fp_for_plot = fp_interp      # shape (time, z)
    dt_for_plot = dt_ext         # shape (time, z)

    # Ensure heights increase with z (per time), else flip both height and data
    try:
        if np.nanmean(np.diff(h_for_plot, axis=1)) < 0:
            h_for_plot = h_for_plot[:, ::-1]
            fp_for_plot = fp_for_plot[:, ::-1]
    except Exception:
        pass

    print(np.shape(dt_for_plot), np.shape(h_for_plot), np.shape(fp_for_plot))
    im_fp = ax1.pcolormesh(
        dt_for_plot, h_for_plot, fp_for_plot,
        cmap=cmap_fp, norm=norm_fp, shading='auto'
    )

    ax1.fill_between(dt, 0, elev, color='black', zorder=2)
    ax1.plot(dt, elev, color='black', linewidth=1.5, label='Surface Elevation')
    ax1.plot(dt, tropo, color='blue', linestyle='--', linewidth=1.5, label='Tropopause Height')
    ax1.set_ylabel("Altitude [km]")
    ax1.set_ylim(0, 20)
    ax1.legend(loc='upper right', fontsize=18)
    ax1.tick_params(labelbottom=False)
    ticks, lticks, indices = format_ticks(dt, lt)
    setup_ax_labels(ax1, ticks, lticks, dt, lat_ec, lon_ec, indices, y_offset=1.2)

    pos1 = ax1.get_position()
    cax1 = fig.add_axes([pos1.x1 + 0.01, pos1.y0, 0.015, pos1.height])
    cbar1 = plt.colorbar(im_fp, cax=cax1)
    cbar1.set_label("[ng m⁻³]" + "\n \n \n \n \n \n" + r"$\mathbf{FLEXPART}$ $\mathbf{BC}$"+ "\n" + r" $\mathbf{Concentration}$", labelpad=10, fontsize=24)
    cbar1.ax.tick_params(labelsize=19)

    # --- ATL_EBD extinction panel ---
    ax2 = fig.add_subplot(gs[22:31, :], sharex=ax1)
    ebd_ext_log = np.ma.masked_where((ebd_ext <= 0) | (ebd_ext > 9.9) | ~np.isfinite(ebd_ext), ebd_ext)
    im_ebd = ax2.pcolormesh(
        ebd_dt.to_numpy(),
        ebd_height_km,
        ebd_ext_log.T,
        shading='auto',
        norm=LogNorm(vmin=1e-6, vmax=1e-2)
    )
    
    ax2.fill_between(dt, 0, elev, color='black', zorder=2)
    ax2.plot(dt, elev, color='black', linewidth=1.5, label='Surface Elevation')
    ax2.plot(dt, tropo, color='blue', linestyle='--', linewidth=1.5, label='Tropopause Height')
    ax2.legend(loc='upper right', fontsize=18)
    ax2.set_ylabel("Altitude [km]")
    ax2.set_ylim(0, 20)

    pos2 = ax2.get_position()
    cax2 = fig.add_axes([pos2.x1 + 0.01, pos2.y0, 0.015, pos2.height])
    cbar2 = plt.colorbar(im_ebd, cax=cax2)
    cbar2.set_label(r"[m$^{-1}$]" + "\n \n \n \n \n \n" + r"$\mathbf{Extinction}$ "+ "\n" + r" $\mathbf{Coefficient}$", labelpad=10, fontsize=24)
    cbar2.ax.tick_params(labelsize=19)

    # --- (3) EarthCARE classification  ---
    ax3 = fig.add_subplot(gs[32:42, :], sharex=ax1)
    im_ec = ax3.pcolormesh(dt_ext, height, class_data, cmap=class_cmap, norm=class_norm, shading='auto')
    ax3.plot(dt, elev, color='black', linewidth=1.5, label='Surface Elevation')
    ax3.plot(dt, tropo, color='blue', linestyle='--', linewidth=1.5, label='Tropopause Height')
    ax3.set_ylabel("Altitude [km]")
    ax3.set_ylim(0, 20)
    ax3.legend(loc='upper right', fontsize=18)
    ax3.tick_params(labelbottom=True)
    for ax in (ax1, ax2, ax3):
        ax.label_outer()
    ax3.set_xlabel("Time [UTC / LT]", fontsize=18, labelpad=8, fontweight='bold')

    pos3 = ax3.get_position()
    cax3 = fig.add_axes([pos3.x1 + 0.01, pos3.y0, 0.015, pos3.height])
    cbar3 = plt.colorbar(im_ec, cax=cax3, spacing='uniform')
    cbar3.set_ticks(tick_centers)
    cbar3.set_ticklabels(class_labels)
    cbar3.set_label("Target \n Classification", fontweight="bold", fontsize=24)
    cbar3.ax.tick_params(labelsize=15)

    # Save
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)





def match_and_plot_flexpart_earthcare(flexpart_folder, 
                                      atl_tc_folder, 
                                      atl_ebd_folder,
                                      output_dir,
                                      plot_function,
                                      gfas_folder=None,
                                      gfas_variant="01"):
    """
    Pairs a single daily FLEXPART file (grid_conc_YYYYMMDDHHMMSS.nc) with
    ALL EarthCARE files (TC + optional EBD) from the same date (YYYYMMDD).

    For each TC file on that date, we call plot_function once, and try to attach
    EBD files that share the same start timestamp.

    If gfas_folder is provided, the plot_function is also handed:
      - gfas_folder (directory containing BC_GFAS_<variant>_YYYYMMDD.nc)
      - gfas_variant (default "01")
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- Patterns ---
    pat_ec_full = re.compile(r"(?P<start>\d{8}T\d{6}Z)_\d{8}T\d{6}Z")
    pat_flex = re.compile(r"grid_conc_(\d{8})\d{6}\.nc$")

     # --- Applying date filters (ONLY_DATES or DATE_RANGE) ---
    only_dates = set()
    if globals().get("ONLY_DATES"):
        for d in globals()["ONLY_DATES"]:
            s = str(d).replace("-", "")
            if len(s) == 8:
                only_dates.add(s)

    start_day = end_day = None
    if globals().get("DATE_RANGE") and len(globals()["DATE_RANGE"]) == 2:
        start_day = str(globals()["DATE_RANGE"][0]).replace("-", "")
        end_day   = str(globals()["DATE_RANGE"][1]).replace("-", "")

    def index_earthcare(folder):
        ts_to_file = {}
        date_to_ts = defaultdict(list)
        if not folder:
            return ts_to_file, date_to_ts

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
            else:
                if ts_to_file[ts] != f:
                    print(f"Duplicate EarthCARE start {ts}: keeping {os.path.basename(ts_to_file[ts])}, ignoring {os.path.basename(f)}")
        for d in date_to_ts:
            date_to_ts[d].sort()
        return ts_to_file, date_to_ts

    # Build indices per product
    tc_ts_to_file,  tc_date_to_ts  = index_earthcare(atl_tc_folder)
    ebd_ts_to_file, ebd_date_to_ts = index_earthcare(atl_ebd_folder) if atl_ebd_folder else ({}, defaultdict(list))

    # --- Scan FLEXPART inputs (single nested file OR many legacy files) ---
    if os.path.isfile(flexpart_folder):
        flexpart_files = [flexpart_folder]
    else:
        all_files  = set(glob.glob(os.path.join(flexpart_folder, "grid_conc_*.nc")))
        nest_files = set(glob.glob(os.path.join(flexpart_folder, "grid_conc_*_nest.nc")))
        flexpart_files = sorted(all_files - nest_files)
        flexpart_files = [f for f in flexpart_files if "2025" in os.path.basename(f)]


    print(f"Found {len(flexpart_files)} FLEXPART NetCDF file(s) in {flexpart_folder}")

    for flex_file in flexpart_files:
        fname = os.path.basename(flex_file)
        days_from_file = _days_in_flexpart_file(flex_file)

        # Applying ONLY_DATES / DATE_RANGE filters
        def _in_filters(d):
            if only_dates and d not in only_dates:
                return False
            if globals().get("DATE_RANGE") and len(globals()["DATE_RANGE"]) == 2:
                start_day = str(globals()["DATE_RANGE"][0]).replace("-", "")
                end_day   = str(globals()["DATE_RANGE"][1]).replace("-", "")
                if not (start_day <= d <= end_day):
                    return False
            return True

        # Decide which days to use for this file
        if days_from_file:
            days_iter = [d for d in days_from_file if _in_filters(d)]
        else:
            # fallback: try to get a day from the filename
            m = pat_flex.search(fname)
            if not m:
                print(f"Skipping FLEXPART file with no recognizable date: {fname}")
                continue
            d = m.group(1)
            if not _in_filters(d):
                print(f"Skipping FLEXPART {fname} outside filters")
                continue
            days_iter = [d]

        if not days_iter:
            print(f"No matching days (after filters) in {fname}")
            continue

        # For each day covered by this file, pair the matching EarthCARE timestamps
        for day in days_iter:
            tc_timestamps = tc_date_to_ts.get(day, [])
            if not tc_timestamps:
                # Quietly skip days that have no EarthCARE files
                continue

            for ts in tc_timestamps:
                tc_file  = tc_ts_to_file.get(ts)
                ebd_file = ebd_ts_to_file.get(ts) if atl_ebd_folder else None

                print(f"   {os.path.basename(tc_file)} matched with {fname}")

                if not globals().get("FORCE_REPLOT", True):
                    out_name = f"combined_earthcare_flexpart_{os.path.splitext(fname)[0]}__{os.path.splitext(os.path.basename(tc_file))[0]}.png"
                    out_png  = os.path.join(output_dir, out_name)
                    if os.path.exists(out_png):
                        print(f"Already plotted {day} for given EarthCARE {os.path.basename(tc_file)}")
                        continue

                try:
                    kwargs = {"ebd_file": ebd_file}
                    if gfas_folder is not None:
                        kwargs["gfas_folder"] = gfas_folder
                        kwargs["gfas_variant"] = gfas_variant

                    plot_function(
                        flex_file,
                        tc_file,
                        output_dir,
                        **kwargs
                    )
                except Exception as e:
                    print(f"Failed to plot for {os.path.basename(tc_file)} matched with {fname}: {e}")


plot_dir_ebd_flex_earthcare = "Z:\\projects\\NEVAR\\Irene\\FLEXPART_EarthCARE_GFAS"
flexpart_folder = "Y:\\users\\ne\\FORWARD_RUNS\\BC_2025\\OUT_BB_irene\\"
ATL_TC_folder = "C:\\Users\\ikar\\OneDrive - NILU\\Documents\\NEVAR\\data\\files_northern_hemisphere\\ATL_TC_20250509_20250612"
ATL_EBD_folder = "C:\\Users\\ikar\\OneDrive - NILU\\Documents\\NEVAR\\data\\files_northern_hemisphere\\ATL_EBD_20250509_20250612"
GFAS_DIR = "Y:\\flex_wrk\\ECMWF_DATA\\GFAS"  


ONLY_DATES = None                         # YYYY-MM-DD or None
DATE_RANGE = ["2025-06-01", "2025-06-15"] # (start, end) inclusive or None
FORCE_REPLOT = True                       # if False, skip already existing PNGs
QUALITY_FLAGS = (0, 1)                    # Plotting for only these quality_status values
RESOLUTION = "low"                        # "high" | "medium" | "low"



match_and_plot_flexpart_earthcare(
    flexpart_folder=flexpart_folder,
    atl_tc_folder=ATL_TC_folder,
    atl_ebd_folder=ATL_EBD_folder,
    output_dir=plot_dir_ebd_flex_earthcare,
    plot_function=plot_subplotted_flexpart_earthcare_class_ebd_and_GFAS,
    gfas_folder=GFAS_DIR,
    gfas_variant="01"
)

