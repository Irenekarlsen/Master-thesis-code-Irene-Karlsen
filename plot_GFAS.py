import os
import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sns
import dask.array as da
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.io.shapereader as shpreader
from matplotlib.colors import Normalize
from shapely.geometry import Point
from pathlib import Path


# === USER INPUTS ===
new_dates = False
data_type = "01"     # "01" for 0.1 degree resolution, "1" for 1 degree resolution, "01_VIIRS_MODIS", "01_VIIRS_obs"
start_date = "2025-05-10"
end_date   = "2025-06-20"
data_dir = "Y:/flex_wrk/ECMWF_DATA/GFAS/"
PLOTS_ROOT = Path("Z:/projects/NEVAR/Irene/plots_GFAS")
lat_range = (45, 70)
lon_range = (-130, -60)


# save plots as SVG instead of PNG
savefig_to_svg = False  

# Which FRP plots to make
plot_dotted_map = True
plot_dotted_map_daily = False
plot_daily_total = False
plot_daily_total_frp_and_bc = True 

# Which altitude plots to make
altitude_variable = "injh"  # "apt". "apb", "mami" or "injh"
plot_daily_altitude = False
plot_daily_altitude_boxplots = False
plot_altitude_dotted_map = False
plot_altitude_dotted_map_daily = False
plot_daily_violin = False
plot_daily_violin_all_altitudes = False
plot_daily_violin_with_bc = False
plot_daily_violin_with_bc_subpanel = True 
compute_altitude_stats = False

if compute_altitude_stats:
    CANADA_BOX = (-140.0, -50.0, 40.0, 70.0)  # (lon_min, lon_max, lat_min, lat_max)
    lat_range = (40, 70)
    lon_range = (-140, -50)
    RMSE_MAE_START = "2025-06-01"
    RMSE_MAE_END   = "2025-06-15"  # inclusive




ALT_MIN_M = 5  # drop everything below 5 m (set to None to disable)

# Helper for pretty variable names
alt_pretty = {"mami": "Mean Altitude of Maximum Injection", 
              "apt": "Plume Top Height Above Surface",
              "apb": "Plume Bottom Height Above Surface",
              "injh": "Injection Height"}

alt_short  = {"mami": "mami", 
              "apt": "apt",
              "apb": "apb",
              "injh": "injh"}



# === Creating area function per long/lat cell ===
def make_cell_area_da(ds, lat_name="latitude", lon_name="longitude"):
    R = 6367470.0  # m
    lat = ds[lat_name].values
    lon = ds[lon_name].values

    # infer step (ascending or descending)
    dlat = np.abs(np.median(np.diff(lat)))
    dlon = np.abs(np.median(np.diff(lon)))

    # lat edges for spherical quad area
    lat_sorted = np.sort(lat)
    lat_edges = np.concatenate(([lat_sorted[0] - dlat/2],
                                (lat_sorted[:-1] + lat_sorted[1:]) / 2,
                                [lat_sorted[-1] + dlat/2]))
    phi1 = np.deg2rad(lat_edges[:-1])
    phi2 = np.deg2rad(lat_edges[1:])
    dphi = np.deg2rad(dlat)
    dlmb = np.deg2rad(dlon)

    band_area_sorted = (R**2) * dlmb * (np.sin(phi2) - np.sin(phi1))  # (nlat,)

    # map band areas back to original lat order
    order = np.argsort(lat)
    invorder = np.argsort(order)
    band_area_orig = band_area_sorted[invorder]

    # broadcast to (lat, lon)
    area_2d = np.outer(band_area_orig, np.ones_like(lon))
    return xr.DataArray(
        area_2d,
        coords={lat_name: ds[lat_name], lon_name: ds[lon_name]},
        dims=(lat_name, lon_name),
        name="cell_area",
        attrs={"units": "m^2", "long_name": "grid-cell area"}
    )


# === Loading data ===
if new_dates:
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    datasets = []
    for date in dates:
        filename = data_dir + f"BC_GFAS_01_{date.strftime('%Y%m%d')}.nc"
        try:
            ds_i = xr.open_dataset(filename)
            datasets.append(ds_i)
        except FileNotFoundError:
            print(f"File not found: {filename}") 
            continue

    print(f"Loaded {len(datasets)} datasets from {start_date} to {end_date}")
    combined = xr.concat(datasets, dim='time')
    ds = combined  
    
    out_combined = f'C:/Users/ikar/OneDrive - NILU/Documents/NEVAR/data/GFAS/combined_BC_GFAS_{data_type}_resolution_{start_date}_{end_date}.nc'
    # out_combined = f'/xnilu_wrk2/projects/NEVAR/Irene/data/GFAS/combined_BC_GFAS_{data_type}_resolution_{start_date}_{end_date}.nc'

    ds.to_netcdf(out_combined)
else:
    ds = xr.open_dataset(f'C:/Users/ikar/OneDrive - NILU/Documents/NEVAR/data/GFAS/combined_BC_GFAS_{data_type}_resolution_{start_date}_{end_date}.nc')
    # ds = xr.open_dataset(f'/xnilu_wrk2/projects/NEVAR/Irene/data/GFAS/combined_BC_GFAS_{data_type}_resolution_{start_date}_{end_date}.nc')


# Setting values only above ALT_MIN_M
if ALT_MIN_M is not None:
    for _v in ["mami", "apt", "apb", "injh"]:
        if _v in ds:
            ds[_v] = ds[_v].where(ds[_v] >= ALT_MIN_M)



# Wrangle longitudes to -180 to 180 if needed
if (ds['longitude'] >= 0).all():
    ds = ds.assign_coords(longitude=((ds['longitude'] + 180) % 360) - 180).sortby('longitude')

# Subset to Canada (once, then reuse)
ds = ds.sel(latitude=slice(lat_range[1], lat_range[0]),  # 75 -> 40
            longitude=slice(lon_range[0], lon_range[1])) # -140 -> -50

# Attaching cell area (m^2) on the subset grid
cell_area = make_cell_area_da(ds, lat_name="latitude", lon_name="longitude")
ds = ds.assign(cell_area=cell_area)

# Always decode CF time and sort/drop duplicates once
ds = xr.decode_cf(ds).sortby('time')
_, uniq_idx = np.unique(ds['time'].values, return_index=True)
ds = ds.isel(time=np.sort(uniq_idx))



# === Sorting plot directories ===
def build_plot_dirs(data_type: str, alt_short: dict, var: str, root=PLOTS_ROOT):
    _slug = lambda s: str(s).strip().replace(" ", "_")
    base = root / _slug(data_type)
    var_key = _slug(alt_short.get(var, var))

    frp_base = base / "frp"
    alt_base = base / var_key

    dirs = {
        # FRP
        "frp_timeseries": frp_base,           # daily total FRP time series, etc.
        "frp_summary":    frp_base,           # e.g. dotted summary map for whole period
        "frp_daily":      frp_base / "daily", # per-day FRP maps

        # ALTITUDE (for selected variable)
        "alt_timeseries": alt_base,           # daily mean, boxplots
        "alt_summary":    alt_base,           # dotted altitude map (95th pct, etc.)
        "alt_daily":      alt_base / "daily"  # per-day altitude maps
    }

    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs

# Build once and reuse everywhere
DIRS = build_plot_dirs(data_type, alt_short, altitude_variable)

if savefig_to_svg:
    def savefig_to(dir_key: str, filename: str, tight = True, dpi: int = 150):
        outdir = DIRS[dir_key] / "svg"
        outdir.mkdir(exist_ok=True)  

        # ensure .svg extension
        if not filename.lower().endswith(".svg"):
            filename = filename.rsplit(".", 1)[0] + ".svg"

        outpath = outdir / filename
        plt.savefig(outpath, format="svg", bbox_inches='tight' if tight else None, dpi=dpi)
        return outpath

else:
    def savefig_to(dir_key: str, filename: str, dpi: int = 150):
        outdir = DIRS[dir_key]
        outpath = outdir / filename
        plt.savefig(outpath, dpi=dpi)
        return outpath




"""------------------------------------Plot daily total FRP -----------------------------------"""
if plot_daily_total:
    # Instantaneous total FRP (power): (W m^-2) * (m^2) -> W
    inst_W = (ds['frpfire'] * ds['cell_area']).sum(dim=['latitude', 'longitude'])

    # Daily mean power (typical for FRP)
    daily_power_W = inst_W.resample(time='1D').mean()

    # Convert to GW for readability
    daily_power_GW = (daily_power_W / 1e9).rename('FRP_daily_power_GW')

    # Plotting
    df = daily_power_GW.to_dataframe().reset_index()
    plt.figure(figsize=(10, 6))
    plt.bar(df['time'], df['FRP_daily_power_GW'], width=0.8, align='center', color='firebrick')

    locator = mdates.DayLocator(bymonthday=[10, 20, 30])
    formatter = mdates.DateFormatter('%d-%b')
    ax = plt.gca()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    plt.title("GFAS Daily Total Fire Radiative Power for Canada")
    plt.ylabel("Total Fire Radiative Power / GW", fontsize=12)
    plt.tight_layout()
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)

    outfile = savefig_to(
        "frp_timeseries",
        f"daily_total_frp_power_GW_{data_type}_{start_date}_{end_date}.png"
    )
    print(f"Saved: {outfile}")




"""-------------------------------------------Plot dotted map --------------------------------------------"""
if plot_dotted_map:
    var = "frpfire"
    frp_field = ds[var].max(dim="time")

    # Mask low values
    frp_masked = frp_field.where(frp_field > 0.1)

    lon, lat = np.meshgrid(ds['longitude'], ds['latitude'])
    lon = lon.flatten(); lat = lat.flatten()
    frp_vals = frp_masked.values.flatten()

    # Bins in W m^-2 (half-open intervals)
    bins   = [0.1, 5, 10, 50, 100]
    colors = ['#ffcccc', '#ff9999', '#ff6666', '#cc0000', '#800000']
    labels = ['0.1-5', '5-10', '10-50', '50-100', '≥100']  # W m^-2

    fig = plt.figure(figsize=(8, 6))
    proj = ccrs.NorthPolarStereo(central_longitude=-100)
    ax = plt.subplot(1, 1, 1, projection=proj)

    # Map extent
    ax.set_extent([lon_range[0], lon_range[1], lat_range[0], lat_range[1]], crs=ccrs.PlateCarree())

    # Base map
    ax.add_feature(cfeature.OCEAN, facecolor='#e8e8e8', zorder=0)
    ax.add_feature(cfeature.LAND, facecolor='white', edgecolor='none', zorder=0)
    ax.coastlines(resolution='50m', linewidth=0.4, color='black', zorder=1)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5, zorder=1)


    # --- Provinces / territories of Canada (admin-1), with labels ---
    shpfilename = shpreader.natural_earth(
        resolution='50m',
        category='cultural',
        name='admin_1_states_provinces'
    )
    reader = shpreader.Reader(shpfilename)

    # Pre-select FRP points (where we actually plot dots)
    frp_mask = np.isfinite(frp_vals) & (frp_vals >= 0.1)
    frp_lon = lon[frp_mask]
    frp_lat = lat[frp_mask]

    for rec in reader.records():
        if rec.attributes.get('admin') != 'Canada':
            continue

        geom = rec.geometry

        # Stable interior point for starting position
        rep = geom.representative_point()

        # ---- skip far-east provinces / territories ----
        if rep.x > -65:   # adjust to -75 if you also want to drop Québec
            continue
        # ------------------------------------------------

        # Draw border
        ax.add_geometries(
            [geom],
            crs=ccrs.PlateCarree(),
            edgecolor='dimgray',
            facecolor='none',
            linewidth=0.5,
            zorder=2.5
        )

        province_name = rec.attributes.get("name", "")
        step_deg = 1.0  # degrees to nudge labels by (try 0.5–1.0)
        candidate_offsets = [
            (0.0, 0.0),
            ( step_deg, 0.0),
            (-step_deg, 0.0),
            (0.0,  step_deg),
            (0.0, -step_deg),
            ( step_deg,  step_deg),
            ( step_deg, -step_deg),
            (-step_deg,  step_deg),
            (-step_deg, -step_deg),
        ]

        best_x, best_y = rep.x, rep.y
        best_score = -np.inf

        # If there are no FRP points, just use the representative point
        if frp_lon.size > 0:
            for dx, dy in candidate_offsets:
                x_try = rep.x + dx
                y_try = rep.y + dy
                pt = Point(x_try, y_try)

                # stay inside the province / territory
                if not geom.contains(pt):
                    continue

                # distance to nearest FRP point (squared)
                dist2 = (frp_lon - x_try) ** 2 + (frp_lat - y_try) ** 2
                min_dist2 = np.min(dist2)

                # we want the candidate that maximizes distance to any FRP dot
                if min_dist2 > best_score:
                    best_score = min_dist2
                    best_x, best_y = x_try, y_try
        # ---------------------------------------------------------

        # Add name label at the chosen position
        ax.text(
            best_x, best_y,
            province_name,
            transform=ccrs.PlateCarree(),
            fontsize=12,
            color="black",
            ha="center",
            va="center",
            zorder=5,
            bbox=dict(
                facecolor="white",
                alpha=0.6,
                edgecolor="none",
                boxstyle="round,pad=0.2"
            )
        )


    # --- Gridlines with labels OUTSIDE and horizontal ---
    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.5,
        color='gray',
        alpha=0.5,
        linestyle='--'
    )

    gl.top_labels = False
    gl.right_labels = False
    gl.bottom_labels = True
    gl.left_labels = True

    # Horizontal labels (not rotated!)
    gl.rotate_labels = False  
    gl.x_inline = False
    gl.y_inline = False

    # Fixed tick positions
    gl.xlocator = mticker.FixedLocator(np.arange(-140, -40, 10))
    gl.ylocator = mticker.FixedLocator(np.arange(40, 80, 5))

    # Label styling
    gl.xlabel_style = {"size": 12, "rotation": 0}
    gl.ylabel_style = {"size": 12, "rotation": 0}


    # interior bins: [low, high)
    for i in range(len(bins)-1):
        mask = (frp_vals >= bins[i]) & (frp_vals < bins[i+1])
        ax.scatter(
            lon[mask], lat[mask],
            color=colors[i],
            s=10,
            label=labels[i],
            transform=ccrs.PlateCarree(),
            zorder=3
        )
        print(f'values in bin {labels[i]}: {np.sum(mask)}')

    # last bin: ≥ last edge
    mask_last = frp_vals >= bins[-1]
    ax.scatter(
        lon[mask_last], lat[mask_last],
        color=colors[-1],
        s=10,
        label=labels[-1],
        transform=ccrs.PlateCarree(),
        zorder=3
    )


    print(f'values in bin {labels[-1]}: {np.sum(mask_last)}')

    legend = ax.legend(
        title='FRP (W m$^{-2}$)',
        loc='upper right',
        fontsize=12,
        title_fontsize=14,
        frameon=True
    )
    legend.get_frame().set_alpha(0.9)

    # plt.title(
    #     f"GFAS Total Fire Radiative Power (W m$^{{-2}}$): {start_date} to {end_date}",
    #     fontsize=16,
    #     pad=20
    # )
    plt.tight_layout()
    outfile = savefig_to(
        "frp_summary",
        f"dotted_frp_map_{data_type}_{start_date}_{end_date}.png"
    )
    print(f"Saved: {outfile}")



"""-------------------------------------------Plot dotted map daily --------------------------------------------"""
if plot_dotted_map_daily:
    # Precompute mesh
    lon2, lat2 = np.meshgrid(ds['longitude'], ds['latitude'])
    lon_flat = lon2.flatten(); lat_flat = lat2.flatten()

    for i in range(ds.sizes['time']):
        this_time = ds['time'].isel(time=i)
        date_str = this_time.dt.strftime("%Y-%m-%d").item()

        frp_day = ds['frpfire'].isel(time=i)  # W m^-2
        vals = frp_day.values.flatten()

        # Keep only ≥ 0.1 W m^-2 to declutter
        mask_valid = vals >= 0.1

        bins   = [0.1, 5, 10, 50, 100]
        colors = ['#ffcccc', '#ff9999', '#ff6666', '#cc0000', '#800000']
        labels = ['0.1–5', '5–10', '10–50', '50–100', '≥100']

        fig = plt.figure(figsize=(10, 10))
        proj = ccrs.NorthPolarStereo(central_longitude=-100)
        ax = plt.subplot(1, 1, 1, projection=proj)

        ax.set_extent([lon_range[0], lon_range[1], lat_range[0], lat_range[1]], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.OCEAN, facecolor='lightgrey')
        ax.add_feature(cfeature.LAND, facecolor='white', edgecolor='none')
        ax.coastlines(resolution='50m', linewidth=0.4, color='black')
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)

        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False; gl.right_labels = False
        gl.bottom_labels = True; gl.left_labels = True

        # interior bins: [low, high)
        for j in range(len(bins)-1):
            mask_bin = (vals >= bins[j]) & (vals < bins[j+1]) & mask_valid
            ax.scatter(lon_flat[mask_bin], lat_flat[mask_bin], color=colors[j], s=10, label=labels[j],
                       transform=ccrs.PlateCarree(), zorder=3)

        # last bin: ≥ last edge
        mask_last = (vals >= bins[-1]) & mask_valid
        ax.scatter(lon_flat[mask_last], lat_flat[mask_last], color=colors[-1], s=10, label=labels[-1],
                   transform=ccrs.PlateCarree(), zorder=3)

        legend = ax.legend(title='FRP (W m$^{-2}$)', loc='lower right',
                           fontsize=9, title_fontsize=10, frameon=True)
        legend.get_frame().set_alpha(0.9)
        plt.title(f"GFAS FRP (W m$^{{-2}}$): {date_str}", fontsize=13, pad=20)
        plt.tight_layout()

        outpath = savefig_to("frp_daily", f"GFAS_FRP_{date_str}.png")
        plt.close()
        print(f"Saved: {outpath}")




"""------------------------------------Daily total FRP + BC (2nd axis)-----------------------------------"""

if plot_daily_total_frp_and_bc:
    # --- FRP: instantaneous total power (W m^-2 * m^2 -> W) ---
    frp_inst_W = (ds["frpfire"] * ds["cell_area"]).sum(dim=["latitude", "longitude"])
    frp_daily_W = frp_inst_W.resample(time="1D").mean()
    frp_daily_GW = (frp_daily_W / 1e9).rename("FRP_daily_GW")

    # --- BC: instantaneous total emission rate (kg m^-2 s^-1 * m^2 -> kg s^-1) ---
    if "bcfire" not in ds:
        raise KeyError(f"'bcfire' not found. Available vars: {list(ds.data_vars)}")

    bc_inst_kgs = (ds["bcfire"] * ds["cell_area"]).sum(dim=["latitude", "longitude"])
    bc_daily_kgs = bc_inst_kgs.resample(time="1D").mean().rename("BC_daily_kg_s")

    # --- align days (in case of missing days) ---
    df = xr.merge([frp_daily_GW, bc_daily_kgs]).to_dataframe().reset_index()

    fig, ax1 = plt.subplots(figsize=(7, 4.5))

    # FRP on left axis (bars)
    ax1.bar(df["time"], df["FRP_daily_GW"], width=0.8, align="center", color="firebrick")
    ax1.set_ylabel("Total Fire Radiative Power [GW]") #, fontsize=12)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.5)

    # BC on right axis (line)
    ax2 = ax1.twinx()
    ax2.plot(df["time"], df["BC_daily_kg_s"], linewidth=1.8, color="black")
    ax2.set_ylabel(r"Total BC emission rate [kg s$^{-1}$]")#, fontsize=12)
    ax2.set_ylim(bottom=0)

    # X ticks like your style
    locator = mdates.DayLocator(bymonthday=[10, 20, 30])
    formatter = mdates.DateFormatter("%d-%b")
    ax1.xaxis.set_major_locator(locator)
    ax1.xaxis.set_major_formatter(formatter)

    ax1.set_title("GFAS Daily Total FRP with Daily Total BC Emissions for Canada in May/June 2025")

    plt.tight_layout()

    outfile = savefig_to(
        "frp_timeseries",
        f"daily_total_frp_GW_and_total_bc_kgs_{data_type}_{start_date}_{end_date}.png"
    )
    print(f"Saved: {outfile}")




# ============================ ALTITUDE PLOTS (mami / apt) ============================
# Bins for altitude in meters
alt_bins_m = [0, 500, 1000, 2000, 3000]
alt_colors = ['#cfe8ff', '#99ccff', '#66a3ff', '#1f78ff', '#0047b3']
alt_labels = ['0-500 m', '500-1000 m', '1000-2000 m', '2000-3000 m', '≥3000 m'] 

"""------------------------------------Daily area-weighted mean altitude -----------------------------------"""
if plot_daily_altitude:
    var = altitude_variable
    if var not in ds:
        raise KeyError(f"Altitude variable '{var}' not found in dataset. Available: {list(ds.data_vars)}")

    # weights only where altitude is valid
    valid_w = ds['cell_area'].where(ds[var].notnull())

    # instantaneous area-weighted mean over space
    weighted_inst = (ds[var] * valid_w).sum(dim=['latitude', 'longitude']) / valid_w.sum(dim=['latitude','longitude'])

    # daily mean of the instantaneous means
    daily_mean_m = weighted_inst.resample(time='1D').mean().rename('daily_altitude_m')

    df_alt = daily_mean_m.to_dataframe().reset_index()

    plt.figure(figsize=(10, 6))
    plt.bar(df_alt['time'], df_alt['daily_altitude_m'], width=0.8, align='center', color='#1f78ff')  # meters
    locator = mdates.DayLocator(bymonthday=[10, 20, 30])
    formatter = mdates.DateFormatter('%d-%b')
    ax = plt.gca()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    plt.title(f"{alt_pretty.get(var,var)}")
    plt.ylabel("Altitude [m]", fontsize=12)
    plt.ylim(0, 3500)   # <-- force axis to 0–3500
    plt.tight_layout()
    outfile = savefig_to(
        "alt_timeseries",
        f"daily_mean_altitude_{alt_short.get(altitude_variable, altitude_variable)}_{data_type}_{start_date}_{end_date}.png"
    )
    print(f"Saved: {outfile}")


""" -------------------------- Daily altitude boxplots (spatial distribution, min–max whiskers) --------------------------"""
if plot_daily_altitude_boxplots:
    var = altitude_variable
    daily_cell_mean = ds[var].resample(time='1D').mean()

    # Collecting spatial distribution across cells
    dates = []
    data_by_day = []
    for t in daily_cell_mean['time'].values:
        # Flatten latitude/longitude into a single dimension and drop NaNs
        vals = (daily_cell_mean.sel(time=t)
                              .stack(cell=('latitude', 'longitude'))
                              .values)
        vals = vals[np.isfinite(vals)]
        if vals.size > 0:
            dates.append(pd.to_datetime(t))
            data_by_day.append(vals)

    if not data_by_day:
        raise ValueError("No daily data available to plot boxplots.")

    # Plot box for each day with min/max whiskers
    plt.figure(figsize=(10, 6))
    positions = mdates.date2num(dates)
    plt.boxplot(
        data_by_day,
        positions=positions,
        widths=0.8,          # one day
        whis=(0, 100),       # whiskers at min/max
        showfliers=False,    
        manage_ticks=False,
        patch_artist=True,
        boxprops=dict(facecolor='#1f78ff')
    )

    ax = plt.gca()
    locator = mdates.DayLocator(bymonthday=[10, 20, 30])
    formatter = mdates.DateFormatter('%d-%b')
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    plt.title(f"{alt_pretty.get(var, var)}")
    plt.ylabel("Altitude [m]", fontsize=12)
    plt.margins(x=0.01)
    plt.tight_layout()

    outfile = savefig_to(
        "alt_timeseries",
        f"daily_boxplot_altitude_{alt_short.get(altitude_variable, altitude_variable)}_{data_type}_{start_date}_{end_date}.png"
    )
    print(f"Saved: {outfile}")



"""-------------------------------------------Plot dotted map --------------------------------------------"""
if plot_altitude_dotted_map:
    var = altitude_variable
    if var not in ds:
        raise KeyError(f"Altitude variable '{var}' not found in dataset. Available: {list(ds.data_vars)}")

    # High percentile over time to summarize the period 
    alt_field = ds[var].quantile(0.95, dim='time')

    lon, lat = np.meshgrid(ds['longitude'], ds['latitude'])
    lon = lon.flatten(); lat = lat.flatten()
    vals = alt_field.values.flatten()

    fig = plt.figure(figsize=(10, 10))
    proj = ccrs.NorthPolarStereo(central_longitude=-100)
    ax = plt.subplot(1, 1, 1, projection=proj)

    ax.set_extent([lon_range[0], lon_range[1], lat_range[0], lat_range[1]], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN, facecolor='lightgrey')
    ax.add_feature(cfeature.LAND, facecolor='white', edgecolor='none')
    ax.coastlines(resolution='50m', linewidth=0.4, color='black')
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False; gl.right_labels = False
    gl.bottom_labels = True; gl.left_labels = True

    # interior finite bins: [low, high)
    for i in range(len(alt_bins_m) - 1):
        low, high = alt_bins_m[i], alt_bins_m[i+1]
        mask = (vals >= low) & (vals < high)
        ax.scatter(lon[mask], lat[mask], color=alt_colors[i], s=10, label=alt_labels[i],
                transform=ccrs.PlateCarree(), zorder=3)

    # open-ended last bin: ≥ last edge
    mask_last = vals >= alt_bins_m[-1]  
    ax.scatter(lon[mask_last], lat[mask_last], color=alt_colors[-1], s=10, label=alt_labels[-1],
            transform=ccrs.PlateCarree(), zorder=3)


    legend = ax.legend(title=f'Altitude ({alt_short.get(var,var)}) [m]', loc='lower right',
                       fontsize=10, title_fontsize=12, frameon=True)
    legend.get_frame().set_alpha(0.9)

    plt.title(f"GFAS {alt_pretty.get(var,var)} (95th pct): {start_date} to {end_date}", fontsize=14, pad=20)
    plt.tight_layout()
    outfile = savefig_to(
        "alt_summary",
        f"dotted_altitude_map_{alt_short.get(altitude_variable, altitude_variable)}_{data_type}_{start_date}_{end_date}.png"
    )
    print(f"Saved: {outfile}")



"""-------------------------------------------Plot dotted map daily --------------------------------------------"""
if plot_altitude_dotted_map_daily:
    var = altitude_variable
    if var not in ds:
        raise KeyError(f"Altitude variable '{var}' not found in dataset. Available: {list(ds.data_vars)}")

    # Precompute mesh
    lon2, lat2 = np.meshgrid(ds['longitude'], ds['latitude'])
    lon_flat = lon2.flatten(); lat_flat = lat2.flatten()

    for i in range(ds.sizes['time']):
        this_time = ds['time'].isel(time=i)
        date_str = this_time.dt.strftime("%Y-%m-%d").item()

        alt_day = ds[var].isel(time=i)  # meters
        vals = alt_day.values.flatten()

        fig = plt.figure(figsize=(10, 10))
        proj = ccrs.NorthPolarStereo(central_longitude=-100)
        ax = plt.subplot(1, 1, 1, projection=proj)

        ax.set_extent([lon_range[0], lon_range[1], lat_range[0], lat_range[1]], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.OCEAN, facecolor='lightgrey')
        ax.add_feature(cfeature.LAND, facecolor='white', edgecolor='none')
        ax.coastlines(resolution='50m', linewidth=0.4, color='black')
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)

        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False; gl.right_labels = False
        gl.bottom_labels = True; gl.left_labels = True

        # interior finite bins: [low, high)
        for j in range(len(alt_bins_m) - 1):
            low, high = alt_bins_m[j], alt_bins_m[j+1]
            mask = (vals >= low) & (vals < high)
            ax.scatter(lon[mask], lat[mask], color=alt_colors[j], s=10, label=alt_labels[j],
                    transform=ccrs.PlateCarree(), zorder=3)

        # open-ended last bin: ≥ last edge
        mask_last = vals >= alt_bins_m[-1]  
        ax.scatter(lon[mask_last], lat[mask_last], color=alt_colors[-1], s=10, label=alt_labels[-1],
                transform=ccrs.PlateCarree(), zorder=3)


        legend = ax.legend(title=f'Altitude ({alt_short.get(var,var)}) [m]', loc='lower right',
                           fontsize=9, title_fontsize=10, frameon=True)
        legend.get_frame().set_alpha(0.9)
        plt.title(f"GFAS {alt_pretty.get(var,var)}: {date_str}", fontsize=13, pad=20)
        plt.tight_layout()

        outpath = savefig_to(
            "alt_daily",
            f"GFAS_altitude_{alt_short.get(altitude_variable, altitude_variable)}_{date_str}.png"
        )
        plt.close()
        print(f"Saved: {outpath}")





""""------------------------------------Daily altitude violin plots -----------------------------------"""
def plot_daily_violin_for_var(ds, var, title, ylabel, dir_key, fname_prefix, color):
    """
    Make a daily spatial-distribution violin plot for `var`.
    - ds: xarray.Dataset with dimensions (time, latitude, longitude)
    - var: name of variable in ds
    - title: plot title string
    - ylabel: y-axis label
    - dir_key: key in DIRS dict to decide where to save
    - fname_prefix: prefix for output filename
    """

    if var not in ds:
        print(f"Variable '{var}' not in dataset, skipping violin plot.")
        return

    # Daily mean per grid cell (same idea as boxplot)
    daily_cell_mean = ds[var].resample(time='1D').mean()

    dates = []
    data_by_day = []
    for t in daily_cell_mean['time'].values:
        vals = (daily_cell_mean.sel(time=t)
                                  .stack(cell=('latitude', 'longitude'))
                                  .values)
        vals = vals[np.isfinite(vals)]
        if vals.size > 0:
            dates.append(pd.to_datetime(t))
            data_by_day.append(vals)

    if not data_by_day:
        print(f"No daily data available for violin plot of '{var}'.")
        return

    # Convert dates to Matplotlib date numbers for x positions
    positions = mdates.date2num(dates)

    fig, ax = plt.subplots(figsize=(10, 6))

    vp = ax.violinplot(
        data_by_day,
        positions=positions,
        widths=0.8,          # one day
        showmeans=True,
        showmedians=False,
        showextrema=True
    )

    # Style violins
    for body in vp['bodies']:
        body.set_facecolor(color)
        body.set_alpha(0.6)
        body.set_edgecolor('black')
        body.set_linewidth(0.5)

    if 'cmedians' in vp:
        vp['cmedians'].set_linewidth(1.0)

    # X-axis formatting: dates as 10/20/30 of month
    locator = mdates.DayLocator(bymonthday=[10, 20, 30])
    formatter = mdates.DateFormatter('%d-%b')
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    plt.title(title)
    plt.ylabel(ylabel, fontsize=12)
    plt.margins(x=0.01)
    plt.tight_layout()

    outname = f"{fname_prefix}_{data_type}_{start_date}_{end_date}.png"
    outfile = savefig_to(dir_key, outname)
    print(f"Saved violin plot for {var}: {outfile}")


if plot_daily_violin:
    # 1) Violin for the current altitude variable
    if altitude_variable in ds:
        plot_daily_violin_for_var(
            ds,
            var=altitude_variable,
            title=alt_pretty.get(altitude_variable, altitude_variable),
            ylabel="Altitude [m]",
            dir_key="alt_timeseries",
            fname_prefix=f"daily_violin_altitude_{alt_short.get(altitude_variable, altitude_variable)}",
            color="#1f78ff"
        )



""""------------------------------------ Daily altitude violin plots all veriables -----------------------------------"""
def _make_daily_violin_df(ds, var):
    """
    Build a tidy DataFrame with columns ['date', 'value'] for seaborn violinplot,
    using daily mean per grid cell for variable `var`.
    """
    if var not in ds:
        print(f"Variable '{var}' missing in dataset, skipping.")
        return pd.DataFrame()

    daily = ds[var].resample(time='1D').mean()

    records = []
    for t in daily['time'].values:
        arr = (daily.sel(time=t)
                      .stack(cell=('latitude', 'longitude'))
                      .values)
        arr = arr[np.isfinite(arr)]
        if arr.size > 0:
            date_str = pd.to_datetime(t).strftime("%Y-%m-%d")
            for v in arr:
                records.append({"date": date_str, "value": v})

    if not records:
        print(f"No valid daily data for variable '{var}'.")
        return pd.DataFrame()

    return pd.DataFrame.from_records(records)



"""------------------------------------ Plot all altitude violins stacked (Option B: Matplotlib dates) -----------------------------------"""
def plot_all_altitude_violins(
    ds,
    vars_list=("apt", "apb", "mami"),
    ymax=9000.0,
    color="#1f78ff",
    widths=0.8,
):
    """
    Create one figure with stacked Matplotlib violin plots (one per altitude variable),
    using a TRUE datetime x-axis (mdates) so ticks/labels are correct.

    - Uses daily mean per grid cell (same aggregation as before)
    - X positions are Matplotlib date numbers (mdates.date2num)
    - Share x/y across subplots
    """

    # Defensive: ensure injh is never included even if passed in
    vars_list = tuple(v for v in vars_list if v != "injh")

    # ---- helper: daily mean per cell -> list of arrays per day + list of datetime days ----
    def _daily_arrays_and_dates(ds, var):
        if var not in ds:
            return [], []

        daily = ds[var].resample(time="1D").mean()

        dates = []
        data_by_day = []
        for t in daily["time"].values:
            vals = (
                daily.sel(time=t)
                .stack(cell=("latitude", "longitude"))
                .values
            )
            vals = vals[np.isfinite(vals)]
            if vals.size > 0:
                dates.append(pd.to_datetime(t))
                data_by_day.append(vals)

        return dates, data_by_day

    # ---- gather data for each variable ----
    series = []  # list of tuples: (var, dates, data_by_day)
    for v in vars_list:
        dates, data_by_day = _daily_arrays_and_dates(ds, v)
        if len(data_by_day) > 0:
            series.append((v, dates, data_by_day))

    if not series:
        print("No altitude data available for multi-violin plot.")
        return

    # ---- create stacked subplots ----
    n = len(series)
    fig, axes = plt.subplots(
        n, 1,
        figsize=(7, 10),
        sharex=True,
        sharey=True
    )
    if n == 1:
        axes = [axes]

    # ---- plot each variable ----
    all_dates_flat = []
    for ax, (v, dates, data_by_day) in zip(axes, series):
        positions = mdates.date2num(dates)  # REAL date axis
        all_dates_flat.extend(dates)

        vp = ax.violinplot(
            data_by_day,
            positions=positions,
            widths=widths,
            showmeans=True,
            showmedians=False,
            showextrema=True
        )

        # Style violins
        for body in vp["bodies"]:
            body.set_facecolor(color)
            body.set_alpha(0.6)
            body.set_edgecolor("black")
            body.set_linewidth(0.5)

        # Style lines (means/min/max/bars) if present
        for k in ("cmeans", "cmins", "cmaxes", "cbars"):
            if k in vp:
                vp[k].set_color("black")
                vp[k].set_linewidth(0.8)

        ax.set_title(alt_pretty.get(v, v), fontweight="bold")
        ax.set_ylabel("Altitude [m]")
        ax.set_ylim(0, ymax)

    # ---- x-axis formatting on bottom axis ----
    locator = mdates.DayLocator(interval=5)     # every 5 days
    formatter = mdates.DateFormatter("%d-%b")   # e.g. 10-May
    axes[-1].xaxis.set_major_locator(locator)
    axes[-1].xaxis.set_major_formatter(formatter)

    # Set x-limits tightly around available dates (prevents weird padding)
    if all_dates_flat:
        xmin = min(all_dates_flat)
        xmax = max(all_dates_flat)
        # add a small margin (half day) so first/last violins aren't clipped
        axes[-1].set_xlim(
            mdates.date2num(xmin) - 0.5,
            mdates.date2num(xmax) + 0.5
        )

    plt.setp(axes[-1].get_xticklabels(), rotation=0, ha="center")
    plt.tight_layout()

    outfile = savefig_to(
        "alt_timeseries",
        f"daily_violin_all_altitudes_{data_type}_{start_date}_{end_date}.png"
    )
    print(f"Saved multi-altitude violin figure: {outfile}")


if plot_daily_violin_all_altitudes:
    plot_all_altitude_violins(ds)





















"""------------------------------------ Daily altitude violin plots with BC -----------------------------------"""
def plot_daily_violin_with_bc(
    ds,
    alt_var,
    bc_var="bcfire",
    title=None,
    alt_ylabel="Altitude [m]",
    bc_ylabel=r"Black carbon flux [kg m$^{-2}$ s$^{-1}$]",
    dir_key="alt_timeseries",
    fname_prefix=None,
    color="#1f78ff",
):
    """
    Daily spatial-distribution violin plot for altitude, with a secondary y-axis
    showing daily area-weighted mean black carbon flux as a continuous line.

    - alt_var: altitude variable name (e.g. 'injh', 'apt', 'mami', 'apb')
    - bc_var:  black carbon variable name ('bcfire' chosen from metadata)
    """

    if alt_var not in ds:
        print(f"Altitude variable '{alt_var}' not in dataset, skipping.")
        return
    if bc_var not in ds:
        print(f"BC variable '{bc_var}' not in dataset, skipping.")
        return

    # ---------- ALTITUDE PART (same logic as your existing violin) ----------
    daily_cell_mean_alt = ds[alt_var].resample(time="1D").mean()

    dates = []
    data_by_day = []
    for t in daily_cell_mean_alt["time"].values:
        vals = (
            daily_cell_mean_alt.sel(time=t)
            .stack(cell=("latitude", "longitude"))
            .values
        )
        vals = vals[np.isfinite(vals)]
        if vals.size > 0:
            dates.append(pd.to_datetime(t))
            data_by_day.append(vals)

    if not data_by_day:
        print(f"No daily altitude data available for violin plot of '{alt_var}'.")
        return

    # Convert dates to Matplotlib date numbers for x positions (as before)
    positions = mdates.date2num(dates)

    fig, ax = plt.subplots(figsize=(10, 6))

    vp = ax.violinplot(
        data_by_day,
        positions=positions,
        widths=0.8,          # one day
        showmeans=True,
        showmedians=False,
        showextrema=True,
    )

    # Style violins exactly as before
    for body in vp["bodies"]:
        body.set_facecolor(color)
        body.set_alpha(0.6)
        body.set_edgecolor("black")
        body.set_linewidth(0.5)

    if "cmeans" in vp:
        vp["cmeans"].set_linewidth(1.0)
        vp["cmeans"].set_color("black")
    if "cmins" in vp:
        vp["cmins"].set_linewidth(0.8)
        vp["cmaxes"].set_linewidth(0.8)

    # X-axis formatting: dates as 10/20/30 of month (same as original)
    locator = mdates.DayLocator(bymonthday=[10, 20, 30])
    formatter = mdates.DateFormatter("%d-%b")
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    if title is None:
        title = alt_pretty.get(alt_var, alt_var)
    ax.set_title(title)
    ax.set_ylabel(alt_ylabel, fontsize=12)
    ax.margins(x=0.01)

    # ---------- BLACK CARBON PART (secondary axis) ----------
    # Area-weighted mean BC over the domain, then daily mean of that
    valid_w_bc = ds["cell_area"].where(ds[bc_var].notnull())
    inst_bc_mean = (
        (ds[bc_var] * valid_w_bc).sum(dim=["latitude", "longitude"])
        / valid_w_bc.sum(dim=["latitude", "longitude"])
    )
    bc_daily = inst_bc_mean.resample(time="1D").mean()  # same daily aggregation

    # Align BC series with the dates used in the violins
    date_index = pd.DatetimeIndex(dates)
    bc_daily_df = bc_daily.to_series()
    bc_for_plot = bc_daily_df.reindex(date_index)

    ax2 = ax.twinx()
    ax2.plot(
        dates,
        bc_for_plot.values,
        linestyle="-",
        linewidth=1.5,
        color="black",
        label="BC flux",
    )
    ax2.set_ylabel(bc_ylabel, color="black", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="black")

    # Optional combined legend (altitude violins + BC line)
    line_label = ax2.get_lines()[0]
    # Just show BC in legend; violins are self-explanatory
    ax2.legend([line_label], ["BC flux"], loc="upper left")

    plt.tight_layout()

    if fname_prefix is None:
        fname_prefix = f"daily_violin_altitude_{alt_short.get(alt_var, alt_var)}_with_BC"

    outname = f"{fname_prefix}_{data_type}_{start_date}_{end_date}.png"
    outfile = savefig_to(dir_key, outname)
    print(f"Saved altitude+BC violin plot for {alt_var} and {bc_var}: {outfile}")




if plot_daily_violin_with_bc:
    if altitude_variable in ds:
        plot_daily_violin_with_bc(
            ds,
            alt_var=altitude_variable,
            bc_var="bcfire",  # from metadata: Wildfire flux of Black Carbon
            title=alt_pretty.get(altitude_variable, altitude_variable),
            alt_ylabel="Altitude [m]",
            bc_ylabel=r"Black carbon flux [kg m$^{-2}$ s$^{-1}$]",
            dir_key="alt_timeseries",
            fname_prefix=f"daily_violin_altitude_{alt_short.get(altitude_variable, altitude_variable)}_with_BC",
            color="#1f78ff",
        )







""" """
def plot_daily_violin_with_bc_subpanel(
    ds,
    alt_var,
    bc_var="bcfire",
    title=None,
    alt_ylabel="Altitude \n [m]",
    bc_ylabel="BC flux \n" +  r"[kg m$^{-2}$ s$^{-1}$]",
    dir_key="alt_timeseries",
    fname_prefix=None,
    color="#1f78ff",
):
    """
    Daily spatial-distribution violin plot for altitude (top), with a
    narrower subpanel underneath showing daily area-weighted mean
    black carbon flux as a continuous line.

    - alt_var: altitude variable name (e.g. 'injh', 'apt', 'mami', 'apb')
    - bc_var:  black carbon variable name ('bcfire' from metadata)
    """

    if alt_var not in ds:
        print(f"Altitude variable '{alt_var}' not in dataset, skipping.")
        return
    if bc_var not in ds:
        print(f"BC variable '{bc_var}' not in dataset, skipping.")
        return

    # ---------- ALTITUDE PART (same logic as your existing violin) ----------
    daily_cell_mean_alt = ds[alt_var].resample(time="1D").mean()

    dates = []
    data_by_day = []
    for t in daily_cell_mean_alt["time"].values:
        vals = (
            daily_cell_mean_alt.sel(time=t)
            .stack(cell=("latitude", "longitude"))
            .values
        )
        vals = vals[np.isfinite(vals)]
        if vals.size > 0:
            dates.append(pd.to_datetime(t))
            data_by_day.append(vals)

    if not data_by_day:
        print(f"No daily altitude data available for violin plot of '{alt_var}'.")
        return

    # Convert dates to Matplotlib date numbers for x positions (as before)
    positions = mdates.date2num(dates)

    # --- Figure and subpanels: tall top (violin) + narrow bottom (BC) ---
    fig = plt.figure(figsize=(7, 4.5))
    gs = fig.add_gridspec(
        2, 1,
        height_ratios=[3, 1],  # top 3x higher than bottom
        hspace=0.11            # small vertical spacing
    )

    ax_violin = fig.add_subplot(gs[0])
    ax_bc = fig.add_subplot(gs[1], sharex=ax_violin)

    # ----- Violin plot on top -----
    vp = ax_violin.violinplot(
        data_by_day,
        positions=positions,
        widths=0.8,          # one day
        showmeans=True,
        showmedians=False,
        showextrema=True,
    )

    # Style violins as in your original
    for body in vp["bodies"]:
        body.set_facecolor(color)
        body.set_alpha(0.6)
        body.set_edgecolor("black")
        body.set_linewidth(0.5)

    if "cmeans" in vp:
        vp["cmeans"].set_linewidth(1.0)
        vp["cmeans"].set_color("black")
    if "cmins" in vp:
        vp["cmins"].set_linewidth(0.8)
        vp["cmaxes"].set_linewidth(0.8)

    # X-axis formatting: do it on the bottom axis (shared x)
    locator = mdates.DayLocator(bymonthday=[10, 20, 30])
    formatter = mdates.DateFormatter("%d-%b")
    ax_bc.xaxis.set_major_locator(locator)
    ax_bc.xaxis.set_major_formatter(formatter)

    if title is None:
        title = alt_pretty.get(alt_var, alt_var)
    ax_violin.set_title(title)
    ax_violin.set_ylabel(alt_ylabel) #, fontsize=12)
    ax_violin.margins(x=0.01)
    ax_violin.set_ylim(0, 9000)  

    # Hide x tick labels on top panel (since bottom will show them)
    plt.setp(ax_violin.get_xticklabels(), visible=False)

    # ---------- BLACK CARBON PART (bottom subpanel) ----------
    # Area-weighted mean BC over the domain, then daily mean
    valid_w_bc = ds["cell_area"].where(ds[bc_var].notnull())
    inst_bc_mean = (
        (ds[bc_var] * valid_w_bc).sum(dim=["latitude", "longitude"])
        / valid_w_bc.sum(dim=["latitude", "longitude"])
    )
    bc_daily = inst_bc_mean.resample(time="1D").mean()

    # Align BC series with the dates used in the violins
    date_index = pd.DatetimeIndex(dates)
    bc_daily_df = bc_daily.to_series()
    bc_for_plot = bc_daily_df.reindex(date_index)

    ax_bc.plot(
        dates,
        bc_for_plot.values,
        linestyle="-",
        linewidth=1.5,
        color="black",
    )
    ax_bc.set_ylabel(bc_ylabel) #, fontsize=12)
    ax_bc.tick_params(axis="x", labelrotation=0)
    ax_bc.grid(True, axis="y", linestyle="--", alpha=0.4)

    # Tight layout for the two panels
    # plt.tight_layout()

    if fname_prefix is None:
        fname_prefix = (
            f"daily_violin_altitude_{alt_short.get(alt_var, alt_var)}_with_BC_subpanel"
        )

    outname = f"{fname_prefix}_{data_type}_{start_date}_{end_date}.png"
    outfile = savefig_to(dir_key, outname)
    print(f"Saved altitude+BC subpanel figure for {alt_var} and {bc_var}: {outfile}")



if plot_daily_violin_with_bc_subpanel:
    if altitude_variable in ds:
        plot_daily_violin_with_bc_subpanel(
            ds,
            alt_var=altitude_variable,
            bc_var="bcfire",  # from metadata: Wildfire flux of Black Carbon
            title=alt_pretty.get(altitude_variable, altitude_variable),
            alt_ylabel="Altitude \n [m]",
            bc_ylabel="BC flux \n" +  r"[kg m$^{-2}$ s$^{-1}$]" + " \n",
            dir_key="alt_timeseries",
            fname_prefix=f"daily_violin_altitude_{alt_short.get(altitude_variable, altitude_variable)}_with_BC_subpanel",
            color="#1f78ff",
        )





def daily_altitude_stats_over_box(
    ds: xr.Dataset,
    var: str,
    box: tuple,
    start_date: str,
    end_date: str,
    lat_name: str = "latitude",
    lon_name: str = "longitude",
    time_name: str = "time",
    area_name: str = "cell_area",
) -> pd.DataFrame:
    """
    Compute daily stats for altitude variable `var` over lon/lat `box`:
      - daily area-weighted mean
      - RMSE and MAE relative to the daily area-weighted mean (spatial dispersion)
      - daily min/max (extremes) of the daily-mean gridcell field

    box = (lon_min, lon_max, lat_min, lat_max)
    Dates are inclusive.
    Returns a DataFrame indexed by date with columns:
      ['mean_m', 'rmse_m', 'mae_m', 'min_m', 'max_m']
    """

    if var not in ds:
        raise KeyError(f"Variable '{var}' not found. Available: {list(ds.data_vars)}")
    if area_name not in ds:
        raise KeyError(
            f"'{area_name}' not found in ds. Make sure you did ds=ds.assign(cell_area=...) before calling."
        )

    lon_min, lon_max, lat_min, lat_max = box

    # --- subset to box (handle latitude ascending or descending) ---
    lat_vals = ds[lat_name].values
    if lat_vals[0] > lat_vals[-1]:
        ds_box = ds.sel({lat_name: slice(lat_max, lat_min), lon_name: slice(lon_min, lon_max)})
    else:
        ds_box = ds.sel({lat_name: slice(lat_min, lat_max), lon_name: slice(lon_min, lon_max)})

    # --- subset time range (inclusive end date) ---
    # xarray slice is inclusive for exact timestamps; to be safe for sub-daily data,
    # extend end to end-of-day by adding 1 day and taking < next day.
    t0 = np.datetime64(pd.to_datetime(start_date))
    t1 = np.datetime64(pd.to_datetime(end_date) + pd.Timedelta(days=1))
    ds_box = ds_box.sel({time_name: slice(t0, t1)})

    # daily mean per grid cell (this is your requested "daily average")
    daily_cell = ds_box[var].resample({time_name: "1D"}).mean()

    # weights only where daily_cell is valid
    w = ds_box[area_name].where(daily_cell.notnull())

    # daily area-weighted mean over space
    mean_daily = (daily_cell * w).sum(dim=[lat_name, lon_name]) / w.sum(dim=[lat_name, lon_name])

    # RMSE relative to daily mean (spatial dispersion)
    se = (daily_cell - mean_daily) ** 2
    rmse_daily = np.sqrt((se * w).sum(dim=[lat_name, lon_name]) / w.sum(dim=[lat_name, lon_name]))

    # MAE relative to daily mean
    ae = np.abs(daily_cell - mean_daily)
    mae_daily = (ae * w).sum(dim=[lat_name, lon_name]) / w.sum(dim=[lat_name, lon_name])

    # Extremes (min/max) over space of the daily-mean gridcell field
    min_daily = daily_cell.min(dim=[lat_name, lon_name], skipna=True)
    max_daily = daily_cell.max(dim=[lat_name, lon_name], skipna=True)

    out = xr.Dataset(
        {
            "mean_m": mean_daily,
            "rmse_m": rmse_daily,
            "mae_m": mae_daily,
            "min_m": min_daily,
            "max_m": max_daily,
        }
    )

    df = out.to_dataframe().reset_index()

    # Keep only the requested inclusive date range (now that we're on daily timestamps)
    df = df[(df[time_name] >= pd.to_datetime(start_date)) & (df[time_name] <= pd.to_datetime(end_date))].copy()

    # nicer column name
    df.rename(columns={time_name: "date"}, inplace=True)
    df.sort_values("date", inplace=True)

    # set date as index if you prefer
    df.set_index("date", inplace=True)

    return df



if compute_altitude_stats:
    stats_df = daily_altitude_stats_over_box(
        ds=ds,
        var="mami",  # or altitude_variable
        box=CANADA_BOX,
        start_date=RMSE_MAE_START,
        end_date=RMSE_MAE_END,
    )

    print(stats_df)

    # Optional: overall extremes over the 15 days (of daily min/max)
    overall_min = stats_df["min_m"].min()
    overall_max = stats_df["max_m"].max()
    print(f"Overall min (daily-mean gridcells) in box: {overall_min:.2f} m")
    print(f"Overall max (daily-mean gridcells) in box: {overall_max:.2f} m")

    stats_df.to_csv(r"Z:\projects\NEVAR\Irene\region_stats\gfas_altitude_stats_CANADA_BOX_20250601_20250615.csv")

