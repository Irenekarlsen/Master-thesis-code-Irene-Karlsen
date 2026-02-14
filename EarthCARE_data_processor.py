import h5py, warnings, re, json
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")


class DataProcessor:
    def __init__(self, base_dir, plot_dir, data_dir, cache_file="cache_matched.json"):
        self.base_dir = Path(base_dir)
        self.plot_dir = Path(plot_dir)
        self.data_dir = Path(data_dir)
        self.cache_file = Path(cache_file)

    def _scan_files(self, directory, start_date, end_date, regions, extension):
        if isinstance(start_date, datetime):
            start_date = start_date.date()
        if isinstance(end_date, datetime):
            end_date = end_date.date()

        pattern = re.compile(
            r"_(?P<start>\d{8}T\d{6}Z)_(?P<end>\d{8}T\d{6}Z)_[0-9]+(?P<region>[A-Za-z])$",
            flags=re.IGNORECASE,
        )

        matched = []
        ext = extension.lstrip(".") if extension else ""

        cur = start_date
        while cur <= end_date:
            date_str = cur.strftime("%Y%m%d")
            glob_pat = f"*{date_str}T*.{ext}" if ext else f"*{date_str}T*"

            for p in Path(directory).rglob(glob_pat):
                if "EXBA" not in p.name.upper():
                    continue
                m = pattern.search(p.stem)
                if not m:
                    continue
                try:
                    dt_start = datetime.strptime(m.group("start"), "%Y%m%dT%H%M%SZ").date()
                except Exception:
                    continue
                region = m.group("region").upper()
                if start_date <= dt_start <= end_date and region in [r.upper() for r in regions]:
                    matched.append(str(p))

            cur += timedelta(days=1)

        return list(dict.fromkeys(matched))

    def get_files(
        self,
        directory,
        start_date,
        end_date,
        regions,
        extension=".h5",
        force_rescan=False,
    ):
        print(f"Getting files from {directory} between {start_date} and {end_date} ")
        # If cache exists and force_rescan is False, load from cache. Otherwise do a fresh scan.

        cache_parent = self.cache_file.parent
        if not cache_parent.exists():
            cache_parent.mkdir(parents=True, exist_ok=True)
            print(f"Created cache directory: {cache_parent}")

        if self.cache_file.exists() and not force_rescan:
            print(f"Loading cached file list from {self.cache_file}")
            return json.loads(self.cache_file.read_text())

        print("Cache not used—scanning directory tree")
        matched = self._scan_files(directory, start_date, end_date, regions, extension)

        with open(self.cache_file, "w") as f:
            json.dump(matched, f, indent=2)

        print(f"Scanned & cached {len(matched)} files to {self.cache_file}")
        return matched

    def filter_by_lat_lon(
        self,
        file_list,
        lat_name="latitude",
        lon_name="longitude",
        lat_bounds=(41.0, 83.0),
        lon_bounds=(-141.0, -52.0),
    ):
        """
        Given a list of HDF5 file paths, return only those that contain at least one
        (lat,lon) point within the provided bounds.

        Parameters:
          file_list : list of file-path strings
          lat_name  : name (within the HDF5) of the latitude dataset
          lon_name  : name of the longitude dataset
          lat_bounds: tuple (min_lat, max_lat)
          lon_bounds: tuple (min_lon, max_lon)
        """
        in_canada = []
        min_lat, max_lat = lat_bounds
        min_lon, max_lon = lon_bounds

        for fp in file_list:
            fp = Path(fp)
            try:
                with h5py.File(fp, "r") as hf:
                    ds = hf["ScienceData"]
                    lats = np.array(ds[lat_name])
                    lons = np.array(ds[lon_name])
            except KeyError:
                print(f" → Skipping {fp.name}: datasets '{lat_name}'/'{lon_name}' not found")
                continue

            mask = (
                (lats >= min_lat)
                & (lats <= max_lat)
                & (lons >= min_lon)
                & (lons <= max_lon)
            )
            if np.any(mask):
                in_canada.append(str(fp))

        print(f"{len(in_canada)} / {len(file_list)} files have data in the Canadian box")
        return in_canada

    def save_files(self, file_list, output_dir):
        """Copy each file in file_list into output_dir (creating it if needed)."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        for src in file_list:
            dst = out_path / Path(src).name
            shutil.copy2(src, dst)

        print(f"Copied {len(file_list)} files into {out_path}")


processor = DataProcessor(
    base_dir="xnilu_wrk2/projects/NEVAR",
    plot_dir="C:/…/plots",

    data_dir="../../data/CalVal/L2a/ATL_TC__2A/2025",
    cache_file="../data/ATL_TC_20250601_20250620/ATL_TC_20250510_20250620.json"
    # data_dir="../../data/CalVal/L2a/ATL_ALD_2A/2025",
    # cache_file="../data/ATL_ALD_20250601_20250620/ATL_ALD_20250510_20250620.json"
    # data_dir="../../data/CalVal/L2a/ATL_EBD_2A/2025",
    # cache_file="../data/ATL_EBD_20250510_20250620/ATL_EBD_20250510_20250620.json"
    # data_dir="../../data/CalVal/L2b/AC__TC__2B/2025",
    # cache_file="../data/AC__TC__20250601_20250620/AC__TC__20250510_20250620.json"
) 


matched = processor.get_files(
    directory=processor.data_dir,
    start_date=datetime(2025, 5, 10),
    end_date=datetime(2025, 6, 20),
    regions=["B", "C", "D"],
    force_rescan=True,
)



processor.save_files(
    matched,
    output_dir= Path(r"../data/ATL_TC_20250510_20250620")
    # output_dir= Path(r"../data/ATL_ALD_20250510_20250620")
    # output_dir= Path(r"../data/ATL_EBD_20250510_20250620")
    # output_dir= Path(r"../data/AC__TC__20250510_20250620")
)
