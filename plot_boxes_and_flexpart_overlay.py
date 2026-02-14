from __future__ import annotations
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


CANADA_BOX = (-140.0, -55.0, 40.0, 65.0)
ATLANTIC_BOX = (-55.0, -10.0, 40.0, 65.0)
EUROPE_BOX = (-10.0, 40.0, 40.0, 65.0)
POLAR_BOX = (-140.0, 40.0, 65.0, 85.0)

REGION_BOXES = {
    "CANADA": CANADA_BOX,
    "ATLANTIC": ATLANTIC_BOX,
    "EUROPE": EUROPE_BOX,
    "POLAR": POLAR_BOX,
}

START_DATE = "2025-05-25"
END_DATE = "2025-06-15"

FLEXPART_NC = "/xnilu_wrk/users/ne/FORWARD_RUNS/BC_2025/OUT_BB_irene/grid_conc_20250101000000.nc"

OUTDIR = "/xnilu_wrk2/projects/NEVAR/Irene/region_stats"
OUTPNG = os.path.join(OUTDIR, f"NH_FLEXPART_meancol_{START_DATE}_to_{END_DATE}.png")


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def wrap_longitudes(lon: np.ndarray) -> np.ndarray:
    """Wrap longitudes to [-180, 180)."""
    lon = np.asarray(lon, dtype=float)
    return ((lon + 180.0) % 360.0) - 180.0


def _try_get_cartopy():
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        return ccrs, cfeature
    except Exception:
        return None, None


def _box_to_lonlat_poly(box):
    lon0, lon1, lat0, lat1 = box
    lons = [lon0, lon1, lon1, lon0, lon0]
    lats = [lat0, lat0, lat1, lat1, lat0]
    return lons, lats


def flexpart_time_mean_column(
    nc_path: str,
    start_date: str,
    end_date: str,
    *,
    varname: str = "spec001_mr",
) -> xr.DataArray:
    ds = xr.open_dataset(nc_path)

    if "longitude" not in ds.coords:
        raise KeyError("Dataset has no 'longitude' coordinate.")
    if "latitude" not in ds.coords:
        raise KeyError("Dataset has no 'latitude' coordinate.")

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    if "time" in ds.coords:
        ds_t = ds.sel(time=slice(start, end))
        if ds_t.sizes.get("time", 0) == 0:
            raise ValueError(f"No FLEXPART times found in range {start_date} → {end_date}")
    else:
        ds_t = ds

    if "height" not in ds_t.coords and "height" not in ds_t.dims:
        raise KeyError("Dataset has no 'height' coordinate/dimension (needed for column).")

    h = ds_t["height"]
    if h.size < 2:
        raise ValueError("Need at least two height levels to compute layer thickness.")

    dz_data = np.diff(np.concatenate(([0.0], h.values)))
    dz = xr.DataArray(dz_data, dims=h.dims, coords=h.coords)

    col = (ds_t[varname] * dz).sum("height") / 1000.0

    col.attrs["units"] = "μg m⁻²"

    if "time" in col.dims:
        col = col.mean("time", skipna=True)

    return col


def add_full_height_colorbar(fig, ax, mappable, *, width=0.025, pad=0.015, label=None):
    """
    Add a colorbar whose height exactly matches `ax`, even for Cartopy GeoAxes.

    width and pad are in figure-relative units.
    """
    fig.canvas.draw()

    pos = ax.get_position()
    cax = fig.add_axes([pos.x1 + pad, pos.y0, width, pos.height])
    cb = fig.colorbar(mappable, cax=cax)
    if label:
        cb.set_label(label)
    return cb


def plot_nh_mean_flexpart_and_boxes(
    out_png: str,
    col_mean: xr.DataArray,
    region_boxes: dict,
    *,
    title: str,
    lat_min: float = 35.0,
    lon_min: float = -160.0,
    lon_max: float = 60.0,
    norm: Normalize | None = None,
    cmap: str = "plasma_r",
):
    ccrs, cfeature = _try_get_cartopy()

    fig = plt.figure(figsize=(12, 9))

    if norm is None:
        norm = Normalize(vmin=200.0, vmax=8000.0)

    lats = col_mean["latitude"].values
    lons = wrap_longitudes(col_mean["longitude"].values)

    order = np.argsort(lons)
    lons = lons[order]
    Z = col_mean.values[:, order]

    if ccrs is not None:
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
        ax.set_extent([lon_min, lon_max, lat_min, 90], crs=ccrs.PlateCarree())

        ax.add_feature(cfeature.LAND, linewidth=0.2)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)

        pm = ax.pcolormesh(
            lons,
            lats,
            Z,
            transform=ccrs.PlateCarree(),
            shading="auto",
            cmap=cmap,
            norm=norm,
        )

        plot_kwargs = dict(transform=ccrs.PlateCarree())
        text_kwargs = dict(transform=ccrs.PlateCarree())
    else:
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, 90)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, linewidth=0.3, alpha=0.4)

        pm = ax.pcolormesh(
            lons,
            lats,
            Z,
            shading="auto",
            cmap=cmap,
            norm=norm,
        )

        plot_kwargs = {}
        text_kwargs = {}

    add_full_height_colorbar(
        fig,
        ax,
        pm,
        width=0.025,
        pad=0.015,
        label="FLEXPART mean column [μg m⁻²]",
    )

    for name, box in region_boxes.items():
        blons, blats = _box_to_lonlat_poly(box)
        ax.plot(blons, blats, linewidth=2.6, alpha=0.95, **plot_kwargs)

        lon0, lon1, la0, la1 = box
        x = 0.5 * (lon0 + lon1)
        y = 0.5 * (la0 + la1)

        ax.text(
            x,
            y,
            name,
            fontsize=10,
            ha="center",
            va="center",
            fontweight="bold",
            **text_kwargs,
        )

    ax.set_title(title, fontsize=14, fontweight="bold")

    plt.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main():
    ensure_dir(OUTDIR)

    col_mean = flexpart_time_mean_column(
        FLEXPART_NC,
        START_DATE,
        END_DATE,
        varname="spec001_mr",
    )

    plot_nh_mean_flexpart_and_boxes(
        OUTPNG,
        col_mean,
        REGION_BOXES,
        title=f"Chosen box regions and FLEXPART mean column for {START_DATE} to {END_DATE}",
        lat_min=35.0,
        lon_min=-160.0,
        lon_max=60.0,
        norm=Normalize(vmin=200.0, vmax=8000.0),
        cmap="plasma_r",
    )

    print(f"Saved: {OUTPNG}")


if __name__ == "__main__":
    main()
