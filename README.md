# EarthCARE – FLEXPART – GFAS Plume Analysis

This repository contains all code used in my Master's thesis for the analysis and comparison of:

- **EarthCARE ATL products (ATL_TC, ATL_EBD, ATL_ALD)**
- **FLEXPART black carbon simulations**
- **GFAS fire radiative power (FRP) and injection heights**

The repository enables:

- Detection of aerosol plume layers using a threshold-based SPH method
- Cross-comparison between modeled (FLEXPART), satellite-retrieved (EarthCARE), and fire emission (GFAS) data
- Regional statistics over predefined geographic domains
- Generation of publication-quality figures
- Time-series, scatter, and map-based diagnostics



---

## Main Components

### 1. Data Handling

**EarthCARE_data_processor.py**  
- Scans directories for ATL files
- Filters by time range and region
- Handles caching of matched files
- Extracts latitude/longitude subsets

**find_regions.py**  
- Identifies which EarthCARE tracks intersect predefined regions:
  - Canada
  - Atlantic
  - Europe
  - Polar

---

### 2. FLEXPART Processing

- Reads NetCDF output
- Computes vertical column concentrations
- Interpolates concentrations onto EarthCARE track
- Generates cross-sections

---

### 3. SPH Layer Detection

**sph_method_implementation_threshold.py**

Implements threshold-based plume detection:

- Identifies contiguous layers above concentration threshold
- Supports multiple layer ranking strategies
- Returns:
  - Layer top
  - Layer bottom
  - Extinction-weighted mean height
  - Thickness
- Optional Gaussian smoothing along-track

This method is applied to:
- FLEXPART concentration fields
- EarthCARE extinction fields

---

### 4. GFAS Fire Data

**plot_GFAS.py**

- Loads daily GFAS FRP and injection height
- Computes grid-cell areas
- Produces:
  - Daily FRP totals
  - FRP maps
  - Altitude statistics
  - Violin plots
  - Fire–plume comparison figures

---

### 5. Combined EarthCARE–FLEXPART–GFAS Plots

**plot_EarthCARE_FLEXPART_GFAS_combined_clean.py**

Generates multi-panel figures including:

- FLEXPART column map
- EarthCARE track overlay
- GFAS FRP fire dots
- FLEXPART vertical cross-section
- EarthCARE extinction
- EarthCARE classification

---

### 6. Regional Statistics

**statistics_on_regions.py**
**plot_region_stats.py**

- Aggregates plume heights by region
- Computes daily and period averages
- Generates:
  - Scatter plots
  - Paired FLEXPART–EarthCARE comparisons
  - Regional summaries


---

## Data Requirements

This repository assumes access to:

- EarthCARE ATL_TC, ATL_EBD, ATL_ALD HDF5 files
- FLEXPART NetCDF output
- GFAS NetCDF files

Due to size restrictions, raw datasets are not included in this repository.

Paths to data directories are defined inside each script and may need modification.




