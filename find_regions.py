import os
import re
import glob
import h5py
import pandas as pd
import numpy as np

# -------------------------
# CONFIG
# -------------------------

CANADA_BOX   = (-140.0, -55.0, 40.0, 65.0)
ATLANTIC_BOX = ( -55.0, -10.0, 40.0, 65.0)
EUROPE_BOX   = ( -10.0,  40.0, 40.0, 65.0)
POLAR_BOX    = (-140.0,  40.0, 65.0, 85.0)

START_DATE = "20250510"  
END_DATE   = "20250620"
ATL_TC_folder  = "/xnilu_wrk2/projects/NEVAR/Irene/data/ATL_TC_20250510_20250620" # os.path.join("..", "data", "ATL_TC_20250510_20250620")
OUTDIR = "/xnilu_wrk2/projects/NEVAR/Irene/data/region_lists_20250510_20250620"
os.makedirs(OUTDIR, exist_ok=True)

# EarthCARE ATL filename example:
# ECA_EXAF_ATL_TC__2A_20250601T002356Z_20250601T010712Z_05727B.h5
PAT = re.compile(r"_(\d{8})T(\d{6})Z_(\d{8})T(\d{6})Z_")

# -------------------------
# HELPERS
# -------------------------
def wrap_lon(lon):
    """Wrap lon to [-180, 180)."""
    lon = np.asarray(lon, dtype=float)
    return ((lon + 180.0) % 360.0) - 180.0

def in_box(lon, lat, box):
    """Return True if any (lon,lat) points fall inside box."""
    lon_min, lon_max, lat_min, lat_max = box
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    m = np.isfinite(lon) & np.isfinite(lat)
    if not np.any(m):
        return False
    lon = lon[m]
    lat = lat[m]
    return np.any((lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max))

def file_start_date_yyyymmdd(fname):
    """Extract start date YYYYMMDD from filename. Return None if not matching."""
    m = PAT.search(fname)
    if not m:
        return None
    return m.group(1)  # start YYYYMMDD

def read_track_latlon(h5_path):
    """Read only latitude/longitude from the EarthCARE ATL_TC HDF5."""
    with h5py.File(h5_path, "r") as f:
        ds = f["ScienceData"]
        lat = np.array(ds["latitude"])
        lon = np.array(ds["longitude"])
    return lat, wrap_lon(lon)

def extract_track_id(filename):
    """
    Returns track ID like '05727B' from an EarthCARE filename.
    """
    return os.path.splitext(filename)[0].split("_")[-1]


# -------------------------
# MAIN SCAN
# -------------------------
canada_files   = []
atlantic_files = []
europe_files   = []
polar_files    = []

all_files = sorted(glob.glob(os.path.join(ATL_TC_folder, "*.h5")))

for f in all_files:
    base = os.path.basename(f)
    d = file_start_date_yyyymmdd(base)
    if d is None:
        continue
    if not (START_DATE <= d <= END_DATE):
        continue

    try:
        lat, lon = read_track_latlon(f)
    except Exception as e:
        print(f"[WARN] Could not read lat/lon from {base}: {e}")
        continue

    track_id = extract_track_id(base)
    if in_box(lon, lat, CANADA_BOX):
        canada_files.append(track_id)
    if in_box(lon, lat, ATLANTIC_BOX):
        atlantic_files.append(track_id)
    if in_box(lon, lat, EUROPE_BOX):
        europe_files.append(track_id)
    if in_box(lon, lat, POLAR_BOX):
        polar_files.append(track_id)


# -------------------------
# SAVE OUTPUTS
# -------------------------
def save_list(name, items):
    # CSV
    df = pd.DataFrame({"filename": items})
    csv_path = os.path.join(OUTDIR, f"{name}_{START_DATE}_{END_DATE}.csv")
    df.to_csv(csv_path, index=False)

    # Optional .txt
    txt_path = os.path.join(OUTDIR, f"{name}_{START_DATE}_{END_DATE}.txt")
    with open(txt_path, "w", encoding="utf-8") as w:
        for x in items:
            w.write(x + "\n")

    print(f"{name}: {len(items)} files -> {csv_path}")

save_list("CANADA",   canada_files)
save_list("ATLANTIC", atlantic_files)
save_list("EUROPE",   europe_files)
save_list("POLAR",    polar_files)