#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 15:31:49 2025

@author: epoirier
"""
import pyrsktools as pyrsk
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import pandas as pd
from datetime import datetime
import glob
from pathlib import Path
from collections import defaultdict
import re
import sys

# custom lib
import sites
import RSKsomlit_proc as rsksproc
import RSKsomlit_plt as rsksplt

#*****************************************************************************
# Processing functions
# ****************************************************************************
# The function find_profile finds the profiles number for one rsk file with one profile
# based on the highest sea-pressure difference in downcast. 
# This to filter the fake profiles due to swell or sensor 
# acclimatation in the beginning of somlit

def find_profile(rsk):
    # args:
    # - takes the rsk file before binning but after computation of the profiles
    # with the proper parameters (this functions depends on the parameters choosen
    # when doing the computation) Default for somlit are rsk.computeprofiles(0.05,5)
    # outputs:
    # the profile number with highest depth difference
    
    # empty list to be filled with [profile number, max pressure difference] by the program
    liste =[]
    # function below gives list with lists inside with all the indices for each down cast 
    downcastIndices = rsk.getprofilesindices(direction="down")
    # d number of down cast in rsk file
    for d in range(len(downcastIndices)):
        # print('d:' + str(d))       
        # i is the list of indices in the array for down cast d          
        i = downcastIndices[d]
        # for pofile d, compare max depth (last indice) - min depth(first indice)
        # we must check for nan values in sea-pressure first
        # we run on indices in list i to find the first one that is not 'nan'
        for n in i:
            if np.isnan(rsk.data[n]['sea_pressure']):
                #print(n)
                continue
            first_indice = n
            #print ('frist_indice'+str(n))
            break
        # we run again on indices but starting from the end to go faster and find the last indice of the profile
        # that is not 'nana
        for k in list(reversed(i)):
            if np.isnan(rsk.data[k]['sea_pressure']):
                #print(k)
                continue
            last_indice = k
            #print ('last_indice'+str(k))
            break
        # we calculate the pressure difference between the end and the beginning of the downcast
        deltap = rsk.data[k]['sea_pressure']-rsk.data[n]['sea_pressure']
        # a problem we have is that we get a 'nan' in the pressure difference with the line below
        #print('pressure difference is'+ str(deltap))
        liste.append([d,deltap])
        #print(liste)
    # we look for the biggest downcast in terms of pressure difference
    # Filter out entries with NaN, to be remove in new version of code
    filtered = [pair for pair in liste if not math.isnan(pair[1])]

    # Find the pair with the max b value
    max_pair = max(filtered, key=lambda x: x[1])

    #print(max_pair)

    profile = max_pair[0]
    #print(profile)
        
    return profile

#********************************************************************************
# check if one single rsk_file has multiple dates or not 
# outputs a boolean and the list of the dates even if one date only
# works ok
def has_multiple_days_and_dates(rsk_file):
    with pyrsk.RSK(rsk_file) as rsk:
          
        rsk.readdata()  # Load the data
    
    if rsk.data is None or len(rsk.data) == 0:
        return False  # No data means no multiple days
    
    # Extract unique dates from the datetime column
    timestamps = rsk.data['timestamp']  # numpy.datetime64 array
   
    dates = np.array([ts.astype('datetime64[D]') for ts in timestamps])
    unique_dates = np.unique(dates)
    
    #unique_dates = {ts.date() for ts in rsk.data['timestamp']}
    
    return len(unique_dates) > 1, unique_dates


# ******************************************************************************
# function to scan the rsk files in my folder 
# split by date if I have a multiple rsk 
# output the list of files rsk kept for next processing step 
def scan_rsk(path_in):
    
    remove_rsk_date_files(path_in) # to clear the folder with the previous _YYYYMMDD rsk files
    
    rsk_files = glob.glob(os.path.join(path_in, "*.rsk")) # creates the list of rsk files originals with path
    file_names = [os.path.basename(path) for path in rsk_files] # list of file names only
    print('Scanning RSK files, checking for multiple dates in files:')
    # Print each file name on a separate line
    for name in file_names:
        print(name)
    
    final_files = []
    final_dates = []
    
    for i, input_file in enumerate(rsk_files): # loop on my list of files, input file is a rsk file
        is_multiple, dates = rsksproc.has_multiple_days_and_dates(input_file)
        # to create alist of final_dates
        print('found these dates in the files:',input_file)
        print(dates)
        
        # loop to create a list of dates, string format and avoid duplicates
        # Use a set for fast lookup
        seen = set(final_dates)
        for d in dates:
            d_str = str(d)  # ensure type matches (e.g., in case it's np.datetime64 or other)
            if d_str not in seen:
                final_dates.append(d_str)
                seen.add(d_str)
        
        if is_multiple: # when having multiple dates in one rsk
            print ('found multiple dates in file:') # print the file containing multiples
            print(input_file)
            # split the rsk if it is multiple, unique_days is a list of dates
            created_files = rsksproc.split_rsk_by_day(input_file) 
          
            # print(f"✅ Created RSK files: {created_files}")
            # final_files.extend(created_files) # add the created files in the list
            
        else: # when no muyliple date
        # we have to rename _YYYYmmdd when our file is not duplicate
        # this is for the routine find duplicate
            
            # Construct filename
            day = np.datetime64(dates[0])
            day_str = str(day).replace('-', '')
            #filename = f"split_{day_str}.rsk"
        
            # Save the new RSK file
            with pyrsk.RSK(input_file) as rsk:
                 rsk.readdata()
                 output_file = rsk.RSK2RSK(suffix=day_str)  # Writes the new rsk file to keep the original one
                 created_files.append(output_file)
            
            # dt = np.datetime64(dates[0])
            # date_str = str(dt).replace('-', '')
            # rename_rsk_with_date(input_file, date_str)
            # final_files.append(input_file) 
            
    final_files_names = [os.path.basename(path) for path in final_files]
    
    final_dates.sort(key=lambda d: datetime.strptime(d, "%Y-%m-%d")) # sort dates chronological
    
    print('Identified SOMLIT dates in the group of files to process:')
    for date in final_dates:
        print(date)
    
    # remove duplicate files per day, some rsk files have the same day in it
    result = remove_duplicates(path_in)
    
    # sort files by date
    sorted_kept = sort_files_by_yymmdd(result['kept'])
    sorted_deleted = sort_files_by_yymmdd(result['deleted'])
    
    print('New list of RSK files (one per day) to process:')
    for f in sorted_kept:
        print(f" - {os.path.basename(f)}")
    print('Deleted files because duplicated')
    for f in sorted_deleted:
        print(f" - {os.path.basename(f)}")
    
    return sorted_kept


#unused function below, to rename a rsk file with a date _ at the end
# def rename_rsk_with_date(file_path, date_str):
#     """
#     Rename an .rsk file by appending _YYYYMMDD before the extension.

#     Parameters:
#         file_path (str): Full path to the original .rsk file.
#         date_str (str): Date string in format 'YYYYMMDD'.

#     Returns:
#         str: New full path after renaming.
#     """
#     if not file_path.endswith('.rsk'):
#         raise ValueError("File must have a .rsk extension")

#     dir_path = os.path.dirname(file_path)
#     base_name = os.path.basename(file_path)
#     name_without_ext = os.path.splitext(base_name)[0]

#     new_name = f"{name_without_ext}_{date_str}.rsk"
#     new_path = os.path.join(dir_path, new_name)

#     os.rename(file_path, new_path)
#     print(f"✅ Renamed to: {new_name}")
#     return new_path


# ***************************************************************************      
# function to remove the duplicates for one date
# output the files deleted and the files kept
def remove_duplicates(path_in):
    pattern = "*.rsk"
    rsk_files = glob.glob(os.path.join(path_in, pattern))
    rsk_files = [f for f in rsk_files if os.path.isfile(f)]

    # Match filenames ending in _YYYYMMDD.rsk
    date_pattern = re.compile(r"_(\d{8})\.rsk$")

    files_by_date = defaultdict(list)

    for f in rsk_files:
        match = date_pattern.search(os.path.basename(f))
        if match:
            date_str = match.group(1)
            files_by_date[date_str].append(f)

    kept = []
    deleted = []

    for date_str, files in files_by_date.items():
        if len(files) == 1:
            kept.append(files[0])
            print(f"✅ Only one file for date {date_str}: {os.path.basename(files[0])}")
            continue

        # Sort by modification time, keep the most recent
        files.sort(key=os.path.getmtime, reverse=True)
        kept_file = files[0]
        to_delete = files[1:]

        kept.append(kept_file)
        print(f"⚠️ Multiple files for date {date_str}. Keeping:")
        print(f"   ➤ {os.path.basename(kept_file)}")

        for f in to_delete:
            try:
                os.remove(f)
                deleted.append(f)
                print(f"   🗑️ Deleted: {os.path.basename(f)}")
            except Exception as e:
                print(f"   ❗Error deleting {os.path.basename(f)}: {e}")

    return {"kept": kept, "deleted": deleted}


# *****************************************************************************
# function to sort the files _YYYYMMDD chronologically
def sort_files_by_yymmdd(files):
    def extract_date(f):
        fname = os.path.basename(f)
        try:
            date_str = fname.split('_')[-1].replace('.rsk', '')
            return datetime.strptime(date_str, "%Y%m%d")
        except:
            return datetime.min  # fallback if pattern doesn't match

    return sorted(files, key=extract_date)



# ********************************************************************************
# Fonction to process a correctly rebuilt raw rsk file with all the channels inside, tridente included
# 8,9,10, chloro, fdom, turbidity, order not checked
# function can only output the profile_nb for one somlit experiment and not several ones on 
# different days that are stored in the same rsk file
def procRSK (path_in, patm, site_id, p_tresh, c_tresh, param, path_out):
    # warning it calls home made find_profile function above
    # args: - path_in: rsk file name with unidentified channels 8,9,10, marked unknown
    #       - patm: atmospheric pressure dBar
    #       - site_id of the point
    #       - pressure treshold for compute profile function
    #       - conductivity treshold for compute profile function
    #       - param: list of parameters you want in destination file ["temperature","chlorophyll-a","par","conductivity"]
    #       meaning they are the parameters in your source file + calculated ones
    #       - path_out: is the location to store the csv files outputted
    # outputs:
    #       - raw: raw rsk file object that only had the recomputeprofile step done
    #       - rsk_d: processed down cast
    #       - rsk_u: processed upcast
    #       - profile_nb: profile number identified for our somlit    

# using the method below is the right way to read the data
# with pyrsk.RSK("/home/epoirier1/Documents/PROJETS/2025/Proc_RBR_Somlit/rawdata/sample.rsk") as rsk:
    
    with pyrsk.RSK(path_in) as rsk:
       
        # read the data first
        rsk.readdata()
        
        

        
        # Use atmopsheric pressure patm to calculate sea pressure
        # Enter sea pressure of the somlit day here
        # In an ideal way the barometric pressure must be measured at each somlit and entered here
        # remind: -1hPa (air pressure) = +1cm sealevel
        # -100hPa = -1dbar = +1m sealevel
        rsk.deriveseapressure(patm)
        
        
        
        # # computing profiles
        # Keep a copy of the raw data to compare with the processed ones
        # the parameters below are adapted for very shallow profile like somlit
        # with acclimatation time
        #-----
        # # args pressure treshold and conductivity treshold
        # # works fine to detect 2 profiles, 2downcast, 2upcast and 2 profiles in rsk.regions
        # decreasing the treshold depth detects more profiles
        # up to 45 profiles 
        
        raw = rsk.computeprofiles(p_tresh,c_tresh)
       
        
        
        # Correct for A2D (analog to digital) zero-holder, find the missing samples and interpolate
        rsk.correcthold(action = "interp")
        
        
        #identify proper profile number
        profile_nb = find_profile(rsk)
        print('procrsk profile nb is' + str(profile_nb))
        
        
        # # get the indices for up a- profile_nb you want to choose, good because in a profile there is the up and down.nd down profiles
        # upcastIndices = rsk.getprofilesindices(direction="up")
        # downcastIndices = rsk.getprofilesindices(direction="down")
        
       
        
        # Low-pass filtering, windowlength is the number of values to use to calculate an average
        # We run at 2Hz, it is slower than the RBR (4Hz) so we won't apply any filter
        
        # rsk.smooth(channels = ["temperature"], windowLength = 5)
        
        # realignement CT
        # time lag of the temperature sensor
        # regarding the profiling speed very slow at somlit, and the red family of conductimeter
        # this lag must be slow << 10 ms (from processing specs, pyrsktools)
        # choosen arbitrary 5 ms shift of the temperature data earlier
        # lag = -0.005
        
        #there is an issue here
        
        
        # removing loops due to swell and probe measuring its wake
        # this might important in shallow coastal waters just as somlit location
          
        # first derivedepth to calculate depth from corrected sea pressure
        # latitude of somlit point at Plouzané written below, comes from a dictionnary
        rsk.derivedepth(sites.site_latitudes[site_id], seawaterLibrary="TEOS-10")
        
        # derive velocity , calculate velocity from depth and time
        # possible to add an argument here to do a window average of the salinity
        # not needed here as we go slow
        rsk.derivevelocity()
        
        # then remove loops
        # speed treshold 0.1m/s mini profiling speed to consider
        # this values is important as all the data below this speed value are removed
        rsk.removeloops(direction= "down", threshold= 0.05)
        
        # Derived variables
        # Salinity
        rsk.deriveseapressure()
        rsk.derivedepth()
        rsk.derivevelocity()
    
        rsk.derivesalinity()
        rsk.derivesigma()
        
        # trim the data to remove unwanted values out of range
        rsk.trim(reference='salinity',range=(0,1), profiles=profile_nb, direction='both', action='remove')
        #-------------------------------------------------------------------------
        # plot depth profile on full rsk file dataset, cannot show either down or up
        # show full profile by default
        # show cast = true does not work, colouring grey/white inefficient
        '''
        rsk.plotdata(channels="depth", profile=profile_nb)
        '''
        
        
        # -----------------------------------------------------------------------
        # create a copy to run independant processes for binaveraging and export on chosen up and down casts
        rsk_u = rsk.copy()
        rsk_d = rsk.copy()
        
        # possible de faire une boucle peut-être ci-dessous
       
        # choose bin Size here and depth limits for the binning process
        bin_size = 0.25
        start_depth = 0.75
        start_d = start_depth-(bin_size/2)
        end_d = round(max(rsk_d.data['depth'])/bin_size)*bin_size-bin_size # this parameter must be checked for not loosing data on downcast
         
        # bin average on depth 0.25dbar or 25 cm for DOWN cast
        rsk_d.binaverage(
            profiles = profile_nb, # we apply the binning only on our profile of interest
            binBy = "depth",
            binSize = bin_size,
            boundary = [start_d,end_d], # parameter to start at 0.75m depth
            # be carefull, choose start depth - binsize/2 to have the value starting the boundary
            direction = "down"
            )
        '''
        # Plot a few profiles of temperature, conductivity, and chlorophyll 
        # for the down cast
        fig, axes = rsk_d.plotprofiles(
            channels=["temperature", "salinity"],
            profiles=profile_nb,
            direction="both",
            reference='depth'
            )
        plt.show()
        '''
        
        rsk_u.binaverage(
            profiles = profile_nb, # we apply the binning only on our profile of interest
            binBy = "depth",
            binSize = bin_size,
            boundary = [start_depth,max(rsk_u.data['depth']).round(0)+start_d],  # parameter to start at 0.75m depth
            # be carefull, choose start depth - binsize/2 to have the value starting the boundary
            direction = "up"
            )
        '''
        # Plot a few profiles of temperature, conductivity, and chlorophyll
        # for the upcast
        fig, axes = rsk_u.plotprofiles(
            channels=["temperature", "salinity"],
            profiles=profile_nb,
            direction="both",
            reference='depth'
            )
        plt.show()
        '''        
        
        
        # Print a list of channels in the rsk file
        # rsk.printchannels()
        '''
        # Plots
        # Plot de timeseries of processed data, choose parameters on each plot
        # when swaping up or down it does not work
        # rsk.readprocesseddata()
        rsk.plotdata(channels=["depth","temperature","salinity"], profile = profile_nb)
        rsk.plotdata(channels=["depth","chlorophyll-a","turbidity"], profile = profile_nb)
        rsk.plotdata(channels=["depth","dissolved_o2_concentration","par"], profile = profile_nb)
        plt.show() 
        '''
        
        # # quality of the temp graph is poor
        # fig, axes = rsk.plotprofiles(
        # channels=["temperature", "salinity"],
        # # we choose profile 1 as it is the good one in our case
        # profiles=profile_nb,
        # direction="both",
        # )for ax in axes:
        #    line = ax.get_lines()[-1]
        # plt.show()
        
        # create a subfolder for this specific rsk file
        # Extract the base filename without extension
        base = os.path.splitext(os.path.basename(path_in))[0]
        file_output_folder = os.path.join(path_out, base)
        os.makedirs(file_output_folder, exist_ok=True)
        print('file output folder' + file_output_folder)
        
        
        #create new folder for destinations csv files
        # we want to go a directory up from raw data folder
        newpath_u = file_output_folder+'/upcast' 
        newpath_d = file_output_folder+'/downcast'
        
        # print('newpath_u'+newpath_u)
        
        if not os.path.exists(newpath_u) and not os.path.exists(newpath_d):
            os.makedirs(newpath_u)
            os.makedirs(newpath_d)
        
        # save required variables in a csv with the correct format
        # export down cast
        rsk_d.RSK2CSV(channels = 
            param, # list of parameters in argument
            profiles=profile_nb,
            comment= "down CAST",
            outputDir=newpath_d)
        # save export file name down cast because rsk2csv does not output it        
        csv_d = rsk_to_profile_csv(newpath_d,0)
        
        # export upcast
        rsk_u.RSK2CSV(channels = 
            param, 
            profiles=profile_nb,
            comment= "up CAST",
            outputDir=newpath_u)
        # save export file name down cast because rsk2csv does not output it        
        csv_u = rsk_to_profile_csv(newpath_u,0)
  
        # #output
        return raw,rsk,rsk_d,rsk_u, profile_nb, file_output_folder, csv_d, csv_u
    

        
        
# ****************************************************************************************************************
# function to procees a list of files in a chosen folder
# list of correct rsk files to process is given in argument
# the loop does not properly work certainly because of the variable of the profle_nb that does not update in the loop.
def process_rsk_folder(path_in, list_of_rsk, site_id, p_tresh, c_tresh, patm, param):
    
    
    # assuming the rsk files are in a rawdatafolder, we want to store the processes_data in a proc_data dir
    # get the dir a step up
    parent_dir = os.path.dirname(path_in)
    path_out = os.path.join(parent_dir, "procdata")
    # creates the path_out directory woth proc_data if it don't already exists
    os.makedirs(path_out, exist_ok=True)
    
    # Extract just the filenames from the list
    rsk_filenames_from_list = {os.path.basename(path) for path in list_of_rsk}
    # List of all rsk files in target directory
    all_files_in_dir = os.listdir(path_in)
    # finding the matching rsk files between two lists
    valid_files_to_process = [
        os.path.join(path_in, f)
        for f in all_files_in_dir
        if f in rsk_filenames_from_list
        ]

    # rsk_files = glob.glob(os.path.join(path_in, "*.rsk"))
    print ('🔄 List of files to be processed:')
    valid_sorted=sort_files_by_yymmdd(valid_files_to_process)# sort by increasinf date
    for f in valid_sorted:
        print(f" - {os.path.basename(f)}")# show the list of files to process
    
    for i, input_file in enumerate(valid_sorted):
        print(f"\n--- Processing file {i+1}/{len(valid_files_to_process)}: {input_file} ---")
        rsksproc.process_rsk_file(input_file, path_out, site_id, p_tresh, c_tresh, patm, param)
    
       
# *****************************************************************************************************************           
# function to do the processing on a single rsk file only.
# it is the same as process_rsk_folder but applied for one file only
# it creates the figures and csv in a folder nammed after the file name
def process_rsk_file(input_file, path_out, site_id, p_tresh, c_tresh, patm, param):
    
    name = os.path.basename(input_file)
    
    # # ask somlit number to user
    # print(f"\n🔄 Processing file: {name}")
    # sys.stdout.flush()  # <-- force flush the print buffer
    # somlit_number = input(f"Enter SOMLIT number for '{name}': ")

    
    # Extract the base filename without extension
    base = os.path.splitext(os.path.basename(input_file))[0]
    
    

    # Create a subfolder for this file
    file_output_folder = os.path.join(path_out, base)
    os.makedirs(file_output_folder, exist_ok=True)
    

    # Define paths for intermediate and final output
    #intermediate_csv = os.path.join(file_output_folder, f"{base}_processed.csv")
    final_csv_d = os.path.join(file_output_folder+'/downcast', f"{base}_4somlit_d.csv")
    final_csv_u = os.path.join(file_output_folder+'/upcast', f"{base}_4somlit_u.csv")
    try:
        print(f"🔄 Processing: {input_file}")

        # Step 1: Convert .rsk to .csv
        raw,rsk,rsk_d,rsk_u, profile_nb, file_output_folder, csv_d, csv_u = rsksproc.procRSK (input_file,
                                                                                              patm,
                                                                                              site_id,
                                                                                              p_tresh,
                                                                                              c_tresh,
                                                                                              param,
                                                                                              path_out)
        
        print(f"Output folder: {file_output_folder}")
        print(f"CSV D path: {csv_d}, CSV U path: {csv_u}")
        print(f"Profile number: {profile_nb}")

       
        #Step 2: Plot
        exclude = ['pressure','sea_pressure','depth']
        for param in [ x for x in param if x not in exclude] :
            rsksplt.plot_up_down2(rsk_d, rsk_u, param, profile_nb, file_output_folder)

        # Step 3: Convert to SOMLIT format
        rsksproc.toSomlitDB(csv_d, site_id, final_csv_d)
        rsksproc.toSomlitDB(csv_u, site_id, final_csv_u)

        print(f"✅ Done: Output in {file_output_folder}")
    except Exception as e:
        print(f"❌ Failed for {input_file}: {e}")
        
    #return somlit_number
        


#***************************************************************************************
# function to find path of csv file form ProcRSK for somlit2db function
def rsk_to_profile_csv(dir_path, profile_nb=0):
    dir_path2 = Path(dir_path).parent
    base_name = dir_path2.name  # Get the last folder name
    filename = f"{base_name}_profile{profile_nb}.csv"
    return str(Path(dir_path) / filename)


# **************************************************************************************
# function to process mrsk file, it means multiple rsk file containing multiple somlit in it.
# The rbr has been set on pause only and several somlit have been saved in one rsk file only
# the idea is to split this one rsk file in one file per day to enable the next steps of our routines
# return the list of rsk files created one per day

def split_rsk_by_day (mrsk_file):
    # mrsk_file is a file containing several days, several somlits
    
    created_files = [] # list of created files
    with pyrsk.RSK(mrsk_file) as rsk:
        rsk.readdata()
    
        # Extract unique dates from the datetime column
        timestamps = rsk.data['timestamp']  # numpy.datetime64 array
       
        dates = np.array([ts.astype('datetime64[D]') for ts in timestamps])
        unique_days = np.unique(dates)
        
        # Loop through each unique day and save a new .rsk file
        for day in unique_days:
            # Get mask for rows matching this day
            # mask says true or false if the date is in unique_day
            mask = dates == day
            # completely new copy of the rsk file
            day_rsk = rsk.copy()
            day_rsk.data = rsk.data[mask]  # Filtered data
            
            # Construct filename
            day_str = str(day).replace('-', '')
            #filename = f"split_{day_str}.rsk"
        
            # Save the new RSK file
            output_file = day_rsk.RSK2RSK(suffix=day_str)  # Writes the file
            created_files.append(output_file)
            
            print(f"✅ Saved: {output_file}")
            
        return created_files
    
# ********************************************************************************
# Function to process a csv file outputted from procRSK custom function
# It takes in charge up or downward cast depending on user choice
# It modifies the file to fit to Somlit database file format
def toSomlitDB (file_path, site_id, output_file):
    # args: - file_path: str csv file coming from proCRSK function, either down or up cast
    #       - site_id : 5 for SOMLIT
    #       - 
    # output_file: output csv_file
    #       - 
    #       - 
    #       -      
    
    
    with open(file_path, "r") as f:
        lines = f.readlines()
    
    # Step 2: Find the last line starting with '//'
    header_line_idx = None
    for idx, line in enumerate(lines):
        if line.startswith("//"):
            header_line_idx = idx
    
    # Step 3: Read the file into DataFrame, skipping earlier lines
    # Beware of sep ',   ' 4 spaces after comma
    df = pd.read_csv(
        file_path,
        sep =',    ',
        skiprows=header_line_idx + 1,     # Skip all lines before actual data
        header=None ,                      # No header in data part
        engine = "python"
    )
    
    # Step 4: Set the header from the last '//' line
    column_names = lines[header_line_idx].lstrip("//").strip().split(",    ")
    df.columns = column_names
    
    # remove line with nan
    df=df.dropna()
    
    # Convert the first column to datetime using your format
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], format='%Y-%m-%dT%H:%M:%S.%f')
    
    # Set the first column as the index, and avoid warning caused
    # by aving diffrent object types in the index: str, numerical, etc...
    df[df.columns[0]] = df[df.columns[0]].infer_objects()
    df.set_index(df.columns[0], inplace=True)  
    
    data = pd.Series([1, '2', 3])  # mixed types
    index = pd.Index(data.infer_objects())  # ensures inference like before

    # Assume df already has datetime index
    
    # 1. Extract date and time from index
    
    # For date only (no time)
    df['DATE'] = df.index.strftime('%Y-%m-%d')
    
    # For time only, without decimals in seconds
    df['HEURE'] = df.index.strftime('%H:%M:%S')
    
    
    # 2. Add first column filled with 5
    df.insert(0, 'ID_SITE', site_id)
    
    #Renames columns 
    df.rename(columns={
        'temperature(°C)': 'TEMPERATURE',
        'chlorophyll-a(ug/l)': 'FLUORESCENCE',
        'par(µMol/m²/s)': 'PAR',
        'salinity(PSU)':'SALINITE',
        'depth(m)':'PROFONDEUR'
    }, inplace=True)
        
    
    # 3. Reorder columns so the first 3 columns are ID_SITE, DATE, HEURE
    # followed by the rest of the original columns (excluding old index)
    cols = ['ID_SITE', 'DATE', 'HEURE', 'TEMPERATURE' ,'FLUORESCENCE' ,'PAR', 'SALINITE', 'PROFONDEUR']
    df = df[cols]
    
    
    # Round to specific decimal numbers per channel for Somlit output file
    
    df['TEMPERATURE'] = df['TEMPERATURE'].round(4)
    df['FLUORESCENCE']=df['FLUORESCENCE'].round(0)
    df['PAR'] = df['PAR'].round(3)
    df['SALINITE'] = df['SALINITE'].round(4)
    df['PROFONDEUR']=df['PROFONDEUR'].round(2)
    
    
    
    # 4. Prepare your multi-line header as a string
    header_lines = [
        "// SOMLIT somlit.fr;;;;;;;",
        "// RBR processing IUEM;"+datetime.now().strftime('%Y-%m-%d;%H:%M:%S')+";;;;;",# current time of processing
        "ID_SITE;DATE;HEURE;TEMPERATURE;FLUORESCENCE;PAR;SALINITE;PROFONDEUR",
        "//;(yyyy-mm-dd);(hh:mm:ss);°C;µg/l;µMol/m²/s;PSU;m",
        ";;;;;;;",
    ]
    
    # 5. Write the file with the custom header
    
    with open(output_file, 'w') as f:
        # Write custom header lines
        for line in header_lines:
            f.write(line + '\n')
        # Write DataFrame to file with ; separator, no header (already written)
        df.to_csv(f, sep=';', index=False, header=False)



# ****************************************************************************      
# function to clear the _YYYYMMDD.rsk files
def remove_rsk_date_files(folder):
    
    # Regex pattern for _YYYYMMDD.rsk
    pattern = re.compile(r'^.*_\d{8}\.rsk$')
    
    # Loop over files in folder
    for filename in os.listdir(folder):
        if pattern.match(filename):
            filepath = os.path.join(folder, filename)
            try:
                os.remove(filepath)
                print(f"Deleted: {filepath}")
            except Exception as e:
                print(f"❌ Failed to delete {filepath}: {e}")





