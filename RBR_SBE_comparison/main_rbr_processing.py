#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 11:59:32 2025

@author: epoirier
"""

import RSKsomlit_proc as rsksproc


# define the path to the raw .rsk file to process that contains several profiles
file = "raw_data/231853_20251017_1321.rsk"
# output folder for the single profile rsk files
path = "proc_data"

files_to_process = rsksproc.export_profiles2rsk(
    file,
    path
)


# Then we will process each rsk file one after the other and store the data in dedicated folders
# that are named after the file name
# all processed data, plots,... are to be found under "proc_data/procdata"

rsksproc.process_rsk_folder(
    path_in=path,  # proc_data folder with the list of files created by the function above
    list_of_rsk=files_to_process,  # output from above
    site_id=5,
    patm=10.1325,
    p_tresh=0.4,  # 0.4 for multiple rsk // 0.05 for simple profile, best results with 0.4 for 15m depth
    c_tresh=5,  # 5 for multiple rsk // 0.5 for simple profile, best results with 5 for 15m depth
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
           # 'velocity',+
           'density_anomaly',
           'dissolved_o2_compensated',
           'temperature1_compensated'
           ])
