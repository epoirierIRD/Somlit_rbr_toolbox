#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 16:41:22 2025


@author: epoirier
Modified: adomeau
"""

import sys
import traceback
from pathlib import Path


def ask_site_id(sites):
    '''
    Affiche la liste des sites Somlit disponibles (id + nom de station,
    depuis sites.py) et demande à l'utilisateur de saisir l'ID du site
    correspondant à ce traitement. Redemande tant que la saisie n'est pas un
    entier valide correspondant à un site existant.

    Parameters
    ----------
    sites : module
        le module sites.py 

    Returns
    -------
    (int, str)
        l'ID du site choisi (garanti valide, présent dans sites.sites) et
        son nom de station
    '''
    print("\nSites Somlit disponibles :")
    for s in sites.sites:
        print(f"  id={s['id']:<3} {s['station']} (latitude {s['latitude']})")

    while True:
        raw_input_value = input(
            "\nEntre l'ID du site Somlit pour ce traitement : ").strip()
        try:
            candidate_id = int(raw_input_value)
        except ValueError:
            print(f"'{raw_input_value}' n'est pas un nombre entier valide, réessaie.")
            continue

        try:
            site = sites.get_site(candidate_id)
        except ValueError as e:
            print(e)
            continue

        return site["id"], site["station"]


def main():
    MAIN_DIR = Path(__file__).resolve().parent  # Folder where main_process.py lives

    # If the .py modules (RSKsomlit_proc.py, sites.py, etc.) live in
    # subfolders next to this script rather than right beside it - e.g. a
    # "Functions" folder for the processing modules and a "Parameters"
    # folder for sites.py - add them to sys.path so the imports below can
    # find them. Built from MAIN_DIR (relative to wherever this script
    # actually sits) rather than hardcoded to one machine's username/path,
    # so this works unchanged on any computer/user account. If you keep
    # every .py file directly alongside main_process.py, these two folders
    # simply won't exist and nothing is added - no harm either way.
    for subfolder in ("Functions", "Parameters"):
        candidate = MAIN_DIR / subfolder
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.append(str(candidate))

    import RSKsomlit_proc as rsksproc
    import sites

    # Dossier contenant les fichiers .rsk bruts à traiter. Tous les fichiers
    # ".rsk" trouvés directement dedans sont traités (pas seulement un
    # fichier nommé en dur) - chacun peut contenir un ou plusieurs profils.
    RAW_DATA_DIR = MAIN_DIR / "raw_data"

    # output folder proc_data for the single profile rsk files, one file per profile
    PROC_DATA_DIR = MAIN_DIR / "proc_data"

    if not RAW_DATA_DIR.exists():  # checks if raw_data folder exists
        raise FileNotFoundError(
            f"{RAW_DATA_DIR} does not exist, create it and put your .rsk files in it")

    if not PROC_DATA_DIR.exists():  # checks if proc_data folder exists
        raise FileNotFoundError(
            f"{PROC_DATA_DIR} does not exist, create it before running the script")

    # Recherche de tous les fichiers .rsk directement dans raw_data (non
    # récursif : si tu ranges tes fichiers dans des sous-dossiers, dis-le
    # moi et je passerai en recherche récursive avec .rglob à la place de
    # .glob).
    raw_rsk_files = sorted(RAW_DATA_DIR.glob("*.rsk"))

    if not raw_rsk_files:
        raise FileNotFoundError(
            f"Aucun fichier .rsk trouvé dans {RAW_DATA_DIR}. "
            f"Vérifie que tes fichiers y sont bien déposés."
        )

    print(f"📂 {len(raw_rsk_files)} fichier(s) .rsk trouvé(s) dans {RAW_DATA_DIR} :")
    for f in raw_rsk_files:
        print(f"  - {f.name}")

    # --- Sélection interactive du site Somlit ---
    site_id, station_name = ask_site_id(sites)
    print(f"\n✅ Site sélectionné : {station_name} (id={site_id})\n")

    '''
    processing of every .rsk file found above
    Function below splits each raw RSK file containing various profiles 
    in one RSK file per profile. All resulting single-profile files (across
    every raw .rsk processed) are accumulated to be processed together next.
    '''
    files_to_process = []
    for raw_file in raw_rsk_files:
        print(f"\n🔀 Découpage en profils : {raw_file.name}")
        files_to_process.extend(
            rsksproc.export_profiles2rsk(str(raw_file), PROC_DATA_DIR)
        )

    '''
    Processing of each rsk file (daily files, one per profile) created above
    and store the data in dedicated folders
    that are named after the file name
    all processed data, plots,... are to be found 
    under "your_path_to_local_repository/Somlit_rbr_toolbox/proc_data/outputs"

    NB: 'param' below is now a "wishlist" of channels, not a fixed
    requirement. The processing pipeline auto-detects, per rsk file, which
    channels are actually present on the probe that produced it (Maestro,
    Concerto, or any other RBR logger) and silently skips whatever isn't
    there (e.g. no dissolved_o2_concentration/ph/chlorophyll-a/fdom/
    turbidity on a Concerto). This means you can list every channel you
    might ever want across all your probes, and mix Maestro and Concerto
    rsk files (and now several raw .rsk files at once) in the same
    folder/run.
    '''
    rsksproc.process_rsk_folder(
        path_in=PROC_DATA_DIR,
        list_of_rsk=files_to_process,
        site_id=site_id,
        patm=10.1325,
        p_tresh=0.4,  # 0.4 for multiple rsk // 0.05 for simple profile
        c_tresh=5,  # 5 for multiple rsk // 0.5 for simple profile
        param=['conductivity',
               'temperature',
               # 'pressure',
               'temperature1',
               'dissolved_o2_concentration',
               'par',
               'ph',
               'chlorophyll-a',
               'fdom',
               'turbidity',
               # 'sea_pressure',
               'depth',
               'salinity',
               # 'speed_of_sound',
               # 'specific_conductivity',
               # 'dissolved_o2_saturation',
               # 'velocity',
               'density_anomaly'
               ],
        # Optional per-probe tuning (all default to the historical Maestro
        # values, so this line can simply be omitted for Maestro-only
        # processing). For a Concerto (no DO sensor), do_sensor_offset is
        # simply unused since the whole DO-compensation block is skipped
        # automatically. Adjust ct_lag here if the Concerto's C/T sensor
        # pair has a different response-time mismatch than the Maestro's.
        # ct_lag=-0.045,
        # cond_scale_factor=1.0,
        # cond_offset=0,
        # do_sensor_offset=0.472,
        )

    print("Programme terminé")


if __name__ == "__main__":
    # IMPORTANT: this try/except/finally is what keeps the console window
    # open and shows the actual error when the script is launched by
    # double-clicking the .py file. Without it, an unhandled exception
    # closes the window instantly (standard Windows behaviour when running
    # a .py file directly), hiding whatever went wrong. When run from
    # Spyder/an IDE this makes no difference - the traceback is shown
    # either way - but it's essential for double-click / plain "python
    # main_process.py" usage.
    try:
        main()
    except Exception:
        print("\n❌ Une erreur est survenue pendant le traitement :\n")
        traceback.print_exc()
    finally:
        input("\nAppuyez sur Entrée pour fermer...")