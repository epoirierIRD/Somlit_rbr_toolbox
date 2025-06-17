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


#*****************************************************************************
# Processing functions

# The function below find the profiles number based on the highest sea-pressure 
# difference in downcast. This to filter the fake profiles due to swell or sensor 
# acclimatation

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
    

# ********************************************************************************
# Fonction to process a correctly rebuilt raw rsk file with all the channels inside, tridente included
# 8,9,10, chloro, fdom, turbidity, order not checked


def procRSK (path_in, patm, latitude, path_out):
    # warning it calls home made find_profile function above
    # args: - path_in: rsk file name with unidentified channels 8,9,10, marked unknown
    #       - patm: atmospheric pressure
    #       - latitude of the point
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
        print(rsk)
        
                
        # Use atmopsheric pressure patm to calculate sea pressure
        # Enter sea pressure of the somlit day here
        # In an ideal way the barometric pressure must be measured at each somlit and entered here
        # remind: -1hPa (air pressure) = +1cm sealevel
        # -100hPa = -1dbar = +1m sealevel
        rsk.deriveseapressure(patm)
        
        # Keep a copy of the raw data to compare with the processed ones
        raw = rsk.computeprofiles(0.05,5)
        
        # Correct for A2D (analog to digital) zero-holder, find the missing samples and interpolate
        rsk.correcthold(action = "interp")
        
        # # computing profiles
        # # args pressure treshold and conductivity treshold
        # # works fine to detect 2 profiles, 2downcast, 2upcast and 2 profiles in rsk.regions
        # decreasing the treshold depth detects more profiles
        # up to 45 profiles 
        rsk.computeprofiles(0.05,5)
        print(rsk)
        
        #identify proper profile number
        profile_nb = find_profile(rsk)
        
        
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
        # latitude of somlit point at Plouzané written below
        rsk.derivedepth(latitude, seawaterLibrary="TEOS-10")
        
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
         
        # bin average on depth 0.25dbar or 25 cm for DOWN cast
        rsk_d.binaverage(
            binBy = "depth",
            binSize = 0.25,
            boundary = [0.625,12], # parameter to start at 0.75m depth
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
        # erreur sur le binning du profil up
        rsk_u.binaverage(
            binBy = "depth",
            binSize = 0.25,
            boundary = [0.625,12],  # parameter to start at 0.75m depth
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
        rsk.printchannels()
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
        
        #create new folder for destinations csv files
        # we want to go a directory up from raw data folder
        newpath_u = path_out+'/upcast' 
        newpath_d = path_out+'/downcast'
        if not os.path.exists(newpath_u) and not os.path.exists(newpath_d):
            os.makedirs(newpath_u)
            os.makedirs(newpath_d)
        
        # save required variables in a csv with the correct format
        # export down cast
        rsk_d.RSK2CSV(channels = [
            "temperature","chlorophyll-a","par","conductivity",
            "dissolved_o2_concentration","turbidity","salinity",
            "depth","density_anomaly"], 
            profiles=profile_nb,
            comment= "down CAST",
            outputDir=newpath_d)
        # export upcast
        rsk_u.RSK2CSV(channels = [
            "temperature","chlorophyll-a","par","conductivity",
            "dissolved_o2_concentration","turbidity","salinity",
            "depth","density_anomaly"], 
            profiles=profile_nb,
            comment= "up CAST",
            outputDir=newpath_u)
  
        # #output
        return raw,rsk,rsk_d,rsk_u, profile_nb
    


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
        
    )
    
    # Step 4: Set the header from the last '//' line
    column_names = lines[header_line_idx].lstrip("//").strip().split(",    ")
    df.columns = column_names
    
    # remove line with nan
    df=df.dropna()
    
    # Convert the first column to datetime using your format
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], format='%Y-%m-%dT%H:%M:%S.%f')
    
    # Set the first column as the index
    df.set_index(df.columns[0], inplace=True)  
    
    
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








  
'''
#function that joins 4 plots, not sure if it works
def plot_combined(rsk, raw, rsk_d, rsk_u, param, cast, profile_nb):
    # Plot raw and processed data
    fig1, axes1 = raw.plotprofiles(channels=[param], profiles=profile_raw, direction=cast)
    fig2, axes2 = rsk.plotprofiles(channels=[param], profiles=profile_proc, direction=cast)
    
    # Plot upcast and downcast data
    fig3, axes3 = rsk_d.plotprofiles(channels=[param], profiles=profile_nb, direction='down')
    fig4, axes4 = rsk_u.plotprofiles(channels=[param], profiles=profile_nb, direction='up')
    
    # Merge all four plots
    fig, axes = rsk.mergeplots(
        [fig1, axes1],
        [fig2, axes2],
        [fig3, axes3],
        [fig4, axes4]
    )
    
    # Customize appearance
    for ax in axes:
        lines = ax.get_lines()
        
        # Style raw vs processed data
        plt.setp(lines[0], linewidth=0.5, marker="o", markerfacecolor="w", label="Original data")
        plt.setp(lines[1], linewidth=0.5, marker="o", markerfacecolor="w", label="Processed data")
        
        # Style upcast vs downcast
        if len(lines) > 2:
            plt.setp(lines[2], linewidth=0.5, linestyle='--', dashes=[10, 5], label="Down_cast")
            plt.setp(lines[3], linewidth=0.5, linestyle='--', dashes=[10, 5], label="Up_cast")
    
    plt.legend()
    plt.title(f"{cast}_{param}")
    plt.show() 
'''





