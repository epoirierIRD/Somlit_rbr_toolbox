#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 16:41:22 2025

@author: epoirier
"""

import pyrsktools as pyrsk
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import RSKsomlit_plt as rsksplt
import RSKsomlit_proc as rsksproc
import sensor_uncertainties as sun

# define the path to the raw .rsk files to process
# they are in the folder raw_rsk_files on your machine under the repository
# Somlit_rbr_toolbox that you have cloned
path = "your_path_to_local_repository/Somlit_rbr_toolbox/raw_rsk_files"


# First we gonna check if there is multiple rsk files in path folder
# Then check if there is duplicate in dates meaning rsk files containing the same data for the same day
# we remove them to keep only one file per Somlit day
# for each day, we keep the latest
# then we have a clear list of rsk files, one per day to process 
# This is to avoid reading original rsk files with no _YYYYMMDD.rsk

files_to_process = rsksproc.scan_rsk(path)

# Then we will process each rsk file (daily files) one after the other and store the data in dedicated folders
# that are named after the file name
# all processed data, plots,... are to be found under "your_path_to_local_repository/Somlit_rbr_toolbox/procdata"

rsksproc.process_rsk_folder(
    path_in = path,
    list_of_rsk = files_to_process,
    site_id =5,
    patm = 10.1325,
    p_tresh = 0.4, #0.4 for multiple rsk // 0.05 for simple profile
    c_tresh = 5, #5 for multiple rsk // 0.5 for simple profile
    param = ['conductivity',
          'temperature',
          #'pressure',
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
          ] )






