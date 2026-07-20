<p align="center">
  <img src="images/logo_IUEM.png" alt="IUEM Logo" width="500"/>
</p>

# SOMLIT RBR toolbox      
*![Last Commit](https://img.shields.io/github/last-commit/epoirierIRD/Somlit_rbr_toolbox)*
## Description

This GitHub repository holds python functions to process RBR Maestro CTD data collected at SOMLIT observatory in Plouzané, France. It is meant to be adaptable to other SOMLIT sites and
to be used by the whole SOMLIT community.

The aim is to provide [SOMLIT](https://www.somlit.fr/) community a list of functions to help process RBR CTD data according to SOMLIT standards. It could useful if your SBE CTD is in calibration and is temporary,
perhaps definitely replaced by a RBR Maestro or Concerto. CTD is deployed in shallow water (< 10m depth) off a quay.

The functions are based on [pyRSKtools](https://docs-static.rbr-global.com/pyrsktools/index.html) free library developped by RBR.
**Use pyRSKtools version 1.3.0 or later.** Earlier versions (including 1.1.2, previously recommended here) contain a bug in `RSK.RSK2RSK()`
(`AssertionError` in `create_rsk_table_map`) that can prevent some valid `.rsk` files from being split into per-profile files. Upgrade with:
```bash
pip install --upgrade pyrsktools
```

It has been developped at the European Institute for Marine Studies (IUEM), Plouzané, France. Specifications have been drawn up by Emilie Grossteffan and Peggy Rimmelin-Maury, active members of the SOMLIT community.
Code has been written by Etienne Poirier, IRD instrumentalist engineer, non SOMLIT member. Be aware that it is a code written by a non-IT engineer with all the cons that this implies.

It is open-source to encourage colleagues from oceanographic research laboratories use it and contribute.
The author would like to thank Mathieu Dever, RBR chief scientist for sharing his code and tutorials especially on RBR CODA data processing.

## Contents

- **RSKsomlit_plt.py**: contains plotting functions only
- **RSKsomlit_proc.py**: contains processing functions only (core function: `procRSK`)
- **channel_config.py**: aliases mapping a logical channel name (e.g. `"temperature1"`) to the various real channel names it can appear under across probes/firmwares (e.g. `"odo_temperature"`)
- **channel_utils.py**: helper functions to auto-detect which channels are actually present on a given probe, and resolve a logical channel name to its real one - this is what makes the toolbox work transparently with Maestro, Concerto, or any other RBR probe without editing code per probe type
- **diff_RBR_SBE.py**: functions to read Somlit-format CSVs (both our own RBR output and third-party SBE files) and automatically compare them by matching date
- **sensor_uncertainties.py** is a ressource file containing the sensor uncertainties for each channel used to show the error bars in the plots
- **sites.py** is the list of SOMLIT sites with corresponding Lat, Lon used in the processing, plus a `get_site(site_id)` lookup function
- **main_process.py** is the script to run to process every `.rsk` file found in the `raw_data` folder (not limited to a single hardcoded file name), with an interactive prompt to choose the SOMLIT site ID
- **main_SBE_rbr_compar.py** is the script that automatically compares the Somlit CSVs produced by `main_process.py` (under `proc_data/outputs/`) against SBE Somlit CSVs deposited in `SBE_completed/`, pairing files up by matching date
- **raw_data** is a folder containing the raw `.rsk` file(s) to process.
    `rbr_maestro.rsk` contains a profile from Lanveoc (14/10/2025) and 2 profiles from Ste Anne du Porzic (16/10/2025)
- **SBE_completed** is a folder (not searched recursively) where you deposit SBE profiles already converted to the Somlit CSV layout (`ID_SITE;DATE;HEURE;TEMPERATURE;FLUORESCENCE;[TURBIDITE];PAR;SALINITE;PROFONDEUR`), used by `main_SBE_rbr_compar.py`
- **environment.yml** is the file containing all packages and dependencies of the conda environment used to develop the code. It hase been created with command:
    ```bash
    conda env export > environment.yml
    ```

## Code features
**main_process.py**
- Finds every `.rsk` file directly inside `raw_data/` (any file name, any number of files - not limited to one hardcoded `rbr_maestro.rsk`) and lists them
- Asks interactively for the SOMLIT site ID before processing: prints every valid `id`/station/latitude from `sites.py` and re-prompts until a valid one is entered
- Reads each raw `.rsk` file:
    - creates one `.rsk` file per profile (in `/proc_data/`)
    - handles multiples days in a `.rsk` file
    - a corrupted/unreadable `.rsk` file is skipped with a clear warning instead of aborting the whole batch
- Process each daily .rsk file:
    - auto-detects which channels the probe actually has (Maestro, Concerto, or any other RBR logger) and silently skips processing steps / channels that don't apply (e.g. no dissolved oxygen compensation on a Concerto without a DO sensor)
    - computes down(d) and up(u) casts with 0.25 m binning
    - outputs RBR .csv and SOMLIT .csv files for d and u casts (UTF-8 with BOM, Excel-friendly, correct `°C`/`µ`/`²` characters)
    - outputs graphics with d and u cast for each parameter (salinity, temperature, etc...) - saved to disk without opening a blocking window, so a batch run of many profiles never waits on a figure being manually closed
    - outputs in `/proc_data/outputs`
- Retries automatically on transient Windows file-lock errors (`PermissionError`, e.g. from OneDrive sync or antivirus scanning) when writing/deleting files
- Shows the full error traceback and pauses before closing the console window if something goes wrong - including when the script is launched by double-clicking the `.py` file rather than from an IDE

**main_SBE_rbr_compar.py**
- Recursively scans `proc_data/outputs/` for every RBR Somlit CSV (`*_4somlit_*.csv`) and scans `SBE_completed/` for SBE Somlit CSVs
- Automatically pairs up RBR and SBE files that share the same calendar date (warns, without blocking, if their `ID_SITE` differs) - no more hardcoded single-file paths
- Compares only the physical parameters common to both files (e.g. skips `TURBIDITE` if the RBR probe doesn't have a turbidity sensor)
- Outputs profile-overlay and difference plots per matched pair, per parameter, under `figures_comparaison_SBE_RBR/<date>/<pair>/`

<p align="center">
  <img src="images/temperature.png" alt="Temperature graph" width="500"/>
</p>

## Functionality

- The code is based on **pyRSKtools v1.3.0+** open source RBR python library to process the data
- Custom functions have been developped to avoid opening Ruskin GUI and reduce clicks when handling the .rsk files and detecting the SOMLIT days and profiles
- Channel auto-detection means the same code path handles RBR Maestro (with DO, pH, fluorometer, etc.) and RBR Concerto (typically CTD + PAR only) transparently, and any mix of probe types in the same processing run
- Plots are to help the user choose between the down or the upcast which to save in the SOMLIT DB.

## Changelog / Recent updates (2026)

The following changes were made to generalize the toolbox beyond the RBR Maestro, and to make it more robust for day-to-day SOMLIT use (contributions from A. Domeau with AI-assisted development):

**Multi-probe support (Maestro, Concerto, and beyond)**
- New `channel_config.py` / `channel_utils.py`: every processing step now auto-detects which channels are actually present on the probe that produced a given `.rsk` file, instead of assuming a fixed Maestro channel set.
- The dissolved-oxygen compensation block in `procRSK()` (subsampling, pressure/salinity compensation, optode lag realignment) now runs only if a DO sensor (`dissolved_o2_concentration` + `temperature1`) is actually detected, and is skipped cleanly (with a log message) on probes without one, such as the Concerto.
- The `param` list passed to `process_rsk_folder()` is now treated as a wishlist: channels not present on a given probe (e.g. `ph`, `chlorophyll-a`, `fdom`, `turbidity`, DO on a Concerto) are filtered out automatically for both the CSV export and the plots, instead of raising an error.
- `toSomlitDB()` fills missing sensor columns (e.g. `FLUORESCENCE` on a probe without a fluorometer) with `NaN` instead of crashing.
- Per-probe tuning parameters (`ct_lag`, `cond_scale_factor`, `cond_offset`, `do_sensor_offset`) are now optional keyword arguments of `procRSK()` (defaulting to the historical Maestro values), instead of hardcoded constants.

**main_process.py**
- Processes every `.rsk` file found in `raw_data/` instead of a single hardcoded `rbr_maestro.rsk`.
- Interactive prompt for the SOMLIT site ID at startup (lists valid IDs from `sites.py`, re-asks on invalid input) instead of a hardcoded `site_id`.
- No more hardcoded absolute paths: `sys.path` is extended with `Functions/`/`Parameters/` subfolders (if present) computed relative to the script's own location, so the repository works unchanged after being moved or cloned to a different machine/account.
- Wrapped in a `try/except/finally` that prints the full error traceback and pauses before closing - previously, an unhandled error would close the console window instantly when the script was launched by double-clicking the `.py` file, hiding the cause.
- A single corrupted/unreadable `.rsk` file no longer aborts the whole batch; it's skipped with a warning and the other files are still processed.

**main_SBE_rbr_compar.py**
- Completely rewritten: instead of comparing one hardcoded pair of RBR/SBE files, it now recursively scans `proc_data/outputs/` for every RBR Somlit CSV and `SBE_completed/` for SBE Somlit CSVs, and automatically pairs them up by matching calendar date.
- New generic `read_somlit_csv()` reader in `diff_RBR_SBE.py` handles both our own RBR output and third-party SBE files sharing the Somlit CSV layout, including auto-detecting whether dates are written `YYYY-MM-DD` or `DD-MM-YYYY`.
- Same portable-path and error-handling improvements as `main_process.py`.

**Bug fixes**
- `sites.py`: added a proper `get_site(site_id)` lookup by the `"id"` field. Previously, `sites.sites[site_id]` treated `site_id` as a **list position**, which silently returned the wrong station for some IDs (e.g. `site_id=5` returned Banyuls instead of Brest) and raised `IndexError` for others (e.g. `site_id=22` for Sète, since the list only has 12 entries).
- `procRSK()`: `removeloops()` is now restricted to `profiles=profile_nb` (like `trim()` already was), fixing a crash (`IndexError` deep inside pyRSKtools) that occurred when `computeprofiles()` detected extra spurious micro-profiles - e.g. a probe dipped in and out of the water a couple of times for calibration right before the real cast.
- Fixed downcast/upcast output folder creation: both are now created independently (`exist_ok=True`) instead of only being created if *neither* already existed, which previously left one folder missing on a re-run.
- Fixed reading of RBR's own `RSK2CSV` export: it's written in Windows-1252 (not UTF-8), so unit symbols like `kg/m³` or `µmol/L` raised `UnicodeDecodeError` when read with the platform's default encoding. A robust reader now tries UTF-8, then cp1252, then latin-1.
- Fixed the final SOMLIT CSV output: added a UTF-8 BOM (`utf-8-sig`) so Excel correctly displays `°C`/`µg/l`/`µMol/m²/s`, and `newline=""` to stop Windows' text-mode newline translation from doubling up line endings (blank line between every row).
- Fixed a `pandas`-version compatibility issue where assigning a parsed datetime column via `.iloc[:, 0] = ...` raised a dtype error on newer pandas releases; now assigned by column label instead.
- Fixed a similar `pandas`-version issue in the SBE/RBR comparison plots, where `np.issubdtype(..., np.number)` raised on newer pandas extension dtypes instead of returning `False`; replaced with `pandas.api.types.is_numeric_dtype`.
- Transient Windows file-lock errors (`PermissionError`, typically from OneDrive syncing a just-written/deleted file, or antivirus scanning) are now retried automatically (a few attempts with a short delay) when exporting `.rsk` files and writing CSVs, instead of aborting the run.
- Batch plotting (`plot_up_down2`) no longer calls a blocking `plt.show()` when a `save_path` is given: figures are saved to disk and closed immediately, so processing many profiles never silently "hangs" waiting for a plot window to be closed by hand. Intermediate figures are also closed to avoid accumulating dozens of open matplotlib figures.
- **Known requirement**: `pyRSKtools` must be **1.3.0 or later** - version 1.1.2 (previously recommended in this README) has a bug in `RSK.RSK2RSK()` that raises `AssertionError` on some otherwise-valid `.rsk` files.

## Using the code

### Installation steps

1. Load the conda environement : [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/)

2. Clone the Repository on your machine:
   ```bash
   git clone https://github.com/epoirierIRD/Somlit_rbr_toolbox.git
   ```
3. Create the conda environment from the provided environment.yml file provided. This file contains all packages and dependencies you need to run the code.
**Be aware the pyrsktools pacakge is not in the .yml and must be installed via pip install, and must be version 1.3.0 or later (see note above).**
For Linux geeks:
    ```bash
    conda env create -f environment.yml
    ```
For Windows users:
use this environment file (Thanks A. Domeau): environment_win.yml

4. Activate the conda environement
    ```bash
    conda activate myenv
    ```
You can rename the environement at this setp.

5. Make sure your `.rsk` file(s) are in the `raw_data` folder (any file name works, you can drop several files in there at once)

6. Run the program main_process.py
    ```bash
    python main_process.py
    ```
    It will list the `.rsk` files found, then ask you to enter the SOMLIT site ID for this run (the list of valid IDs is printed first) before starting the processing.

7. A folder is created under your_path/Somlit_rbr_toolbox/proc_data
contaiins the outputs for all the .rsk files processed. They are stored in one
folder for each daily file.

8. To compare against SBE reference profiles, deposit the SBE Somlit-format CSVs in a `SBE_completed` folder next to `main_process.py`, then run:
    ```bash
    python main_SBE_rbr_compar.py
    ```
   Comparison plots for every date found in both sources are written to `figures_comparaison_SBE_RBR/`.

## Raise an issue

- When using the code any issue you will see or any idea that comes to your mind, please submit it via the github toolbox
Issue/ Create new Issue. That will help developpers improving the code for your needs.

## Avenues for Improvement

- Migrate the repository for github to https://versio.iuem.eu GitLab
- Remove useless lines of code / excessive comments
- Develop new functions
- Develop a friendly user interface for non python users, perhaps with a Windows .exe program
- Rebuilt the code with an object object-oriented programming (OOP) strategy

Contributions are welcomed to improve these points.

## To contribute

First step to contribute is to raise issues! 
Otherwise:

Follow this procedure to contribute:

1. **Fork the repository** and create your branch (my-new-feature):

   ```bash
   git checkout -b my-new-feature
   ```

2. **Make your changes** and test them thoroughly.


3. **Submit a Pull Request** with a detailed description of your changes.

## Contributors

IUEM:
- Etienne Poirier
- Emilie Grossteffan
- Peggy Rimmelin-Maury

MEDIMEER:
- Aurélien Domeau

## License

This project is licensed under [CC BY-SA 4.0]. 

