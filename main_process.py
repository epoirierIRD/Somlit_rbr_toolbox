#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 16:41:22 2025

@author: epoirier
"""

import RSKsomlit_proc as rsksproc
from pathlib import Path


MAIN_DIR = Path(__file__).resolve().parent # Folder where main_process.py lives

# define the path to the raw .rsk file to process that contains several profiles
file = MAIN_DIR / "raw_data" / "rbr_maestro.rsk" #for relative path
file = str(file)

# output folder proc_data for the single profile rsk files, one file per profile
PROC_DATA_DIR = MAIN_DIR / "proc_data"

if not PROC_DATA_DIR.exists(): # checks if proc_data folder exists
    raise FileNotFoundError(
        f"{PROC_DATA_DIR} does not exist, create it before running the script")
'''
processing of one single file here
info: can be done on several files too
Function below splits one single RSK files containing various profiles 
in one RSK file per profile
'''
files_to_process = rsksproc.export_profiles2rsk(file, PROC_DATA_DIR)

'''
Processing of each rsk file (daily files, one per profile) created above
and store the data in dedicated folders
that are named after the file name
all processed data, plots,... are to be found 
under "your_path_to_local_repository/Somlit_rbr_toolbox/proc_data/outputs"
'''
rsksproc.process_rsk_folder(
    path_in=PROC_DATA_DIR,
    list_of_rsk=files_to_process,
    site_id=5,
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
           ])
