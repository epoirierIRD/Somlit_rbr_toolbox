#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script file name: RSKsomlit_proc.py
Author: Etienne Poirier, IRD, Plouzané
Modified: Aurélien DOMEAU, CNRS, Sète with Claude Sonnet 5
Date created: 2025-06-02
Last update: 2026-07-16

Description: various processing functions for RBR CTD profile data 
used in Somlit experiments.

Crucial function: procRSK that processes a RSK file and its channels
"""
import pyrsktools as pyrsk
import numpy as np
import os
import math
import time
import pandas as pd
from datetime import datetime
import glob
from pathlib import Path
from collections import defaultdict
import re

# custom lib
import sites
import RSKsomlit_proc as rsksproc
import RSKsomlit_plt as rsksplt
import channel_utils


# %% Processing functions below
# %% func export_profiles2rsk
def export_profiles2rsk(inp_file, output_dir="."):
    """
    Takes one single RSK file with several profiles inside
    Split the RSK file into one RSK file per PROFILE region, preserving metadata
    Automatically deletes existing files with the same name when running the script

    Parameters
    ----------
    inp_file : str
        Path to the input .rsk file
    output_dir : str, optional
        Directory where the profile RSK files will be saved.
        Defaults to the directory up ".".
    Returns:
    ----------
    outputs : List
        export the list of files created
    """

    # base name without extension of rsk file
    base_name = os.path.splitext(os.path.basename(inp_file))[0]

    with pyrsk.RSK(inp_file) as rsk:  # metadata loaded

        # Try to get RegionProfile class
        try:
            from pyrsktools.datatypes import RegionProfile

            profiles = rsk.getregionsbytypes(RegionProfile)
        except Exception:
            profiles = [
                reg
                for reg in rsk.regions
                if getattr(reg, "type", "").upper() == "PROFILE"
                or reg.__class__.__name__ == "RegionProfile"
            ]

        if not profiles:
            rsk.readdata()
            rsk.computeprofiles()
            profiles = [
                reg
                for reg in rsk.regions
                if getattr(reg, "type", "").upper() == "PROFILE"
            ]

        if not profiles:
            raise RuntimeError("No PROFILE regions found in this file.")

        outputs = []
        for i, prof in enumerate(profiles):
            t1, t2 = prof.tstamp1, prof.tstamp2

            new = rsk.copy()
            new.readdata(t1, t2)
            new.computeprofiles()

            # safe filename suffix
            t1s = np.datetime_as_string(t1, unit="s").replace(":", "-")
            t2s = np.datetime_as_string(t2, unit="s").replace(":", "-")

            # include original file base name
            suffix = f"{base_name}_profile_{i}_{t1s}_to_{t2s}"
            outname = os.path.join(output_dir, f"{suffix}.rsk")

            # print(outname)
            # delete if file already exists
            if os.path.exists(outname):
                os.remove(outname)
                print("Deleted existing file:", outname)

            # export
            new.RSK2RSK(
                outputDir=output_dir, suffix=f"profile_{i}_{t1s}_to_{t2s}"
            )
            outputs.append(outname)
            print("Wrote:", outname)

    return outputs


# %% find_profile


def find_profile(rsk):
    """
    The function find_profile finds the profile number for one rsk file containing one profile
    based on the highest sea-pressure difference in the downcast.
    This to filter the fake profiles due to swell or sensor acclimatation in the beginning of
    the SOMLIT profile

    Parameters
    ----------
    rsk: str
        rsk file path before binning but after computation of the profiles
    Returns
    ----------
    profile: int
        the profile number with highest depth difference
    """
    # empty list to be filled with [profile number, max pressure difference] by the program
    liste = []
    # function below gives list with lists inside with all the indices for each down cast
    downcastIndices = rsk.getprofilesindices(direction="down")
    # d number of down cast in rsk file
    for d in range(len(downcastIndices)):
        # i is the list of indices in the array for down cast d
        i = downcastIndices[d]
        # for pofile d, compare max depth (last indice) - min depth(first indice)
        # we must check for nan values in sea-pressure first
        # we run on indices in list i to find the first one that is not 'nan'
        for n in i:
            if np.isnan(rsk.data[n]["sea_pressure"]):
                continue
            break
        # we run again on indices but starting from the end to go faster and find the last indice of the profile
        # that is not 'nana
        for k in list(reversed(i)):
            if np.isnan(rsk.data[k]["sea_pressure"]):
                continue
            break
        # we calculate the pressure difference between the end and the beginning of the downcast
        deltap = rsk.data[k]["sea_pressure"] - rsk.data[n]["sea_pressure"]
        # a problem we have is that we get a 'nan' in the pressure difference with the line below
        liste.append([d, deltap])
    # we look for the biggest downcast in terms of pressure difference
    # Filter out entries with NaN, to be remove in new version of code
    filtered = [pair for pair in liste if not math.isnan(pair[1])]

    # Find the pair with the max b value
    max_pair = max(filtered, key=lambda x: x[1])

    profile = max_pair[0]

    return profile


# %% has_multiple_days_and_dates


def has_multiple_days_and_dates(rsk_file):
    """
    This function checks checks if one single rsk_file has multiple dates or not
    # outputs a boolean and the list of the dates even if one date only
    # works ok

    Parameters
    ----------
    rsk_file : TYPE
        DESCRIPTION.

    Returns
    -------
    BOOLEAN
        indicates if there is True or False multiple dates in the rsk file
    unique_dates: List
        List of the dates found in the rsk fle even if there is only one

    """
    with pyrsk.RSK(rsk_file) as rsk:

        rsk.readdata()  # Load the data

    if rsk.data is None or len(rsk.data) == 0:
        return False  # No data means no multiple days

    # Extract unique dates from the datetime column
    timestamps = rsk.data["timestamp"]  # numpy.datetime64 array

    dates = np.array([ts.astype("datetime64[D]") for ts in timestamps])
    unique_dates = np.unique(dates)

    # unique_dates = {ts.date() for ts in rsk.data['timestamp']}

    return len(unique_dates) > 1, unique_dates


# %% scan_rsk


def scan_rsk(path_in):
    """
    This function scans the raw rsk files in my folder path_in
    If I have a multiple dates in the rsk, it splits by date the rsk
    and creates new rsk _YYYYMMDD.rsk, one per day in /proc_data
    
    Removes _YYYYMMDD.rsk in proc_data files before running the code to work only on raw_rsk files
    Beware that if a raw rsk file name is _YYYYNNDD, it will be removed
    After the processing it removes the duplicated files. somettimes data of the day -1 
    are kept in day 1 rsk file, that will lead in two day-1 files identical

    Parameters
    ----------
    path_in : str
        Origin folder containing the rsk files to scan

    Returns
    -------
    sorted_kept : List
        List of rsk files names kept after the scanning

    """
     
    # Path of the new 'proc_data' folder
    proc_data_path = os.path.join(path_in, "proc_data")
    print('proc_data_path',proc_data_path)
    
    # Create the folder if it doesn't exist
    os.makedirs(proc_data_path, exist_ok=True)

    # to clear the folder with the previous _YYYYMMDD rsk files
    remove_rsk_date_files(proc_data_path)

    # creates the list of raw rsk files of interest found in path_in
    rsk_files = glob.glob(os.path.join(path_in, "*.rsk"))
    
    # list of file names only
    file_names = [os.path.basename(path) for path in rsk_files]
    print("Scanning RSK files, checking for multiple dates in files:")
    # Print each file name on a separate line
    for name in file_names:
        print(name)

    final_dates = []
    created_files = []

    # loop on my list of files, input file is a rsk file
    for i, input_file in enumerate(rsk_files):
        is_multiple, dates = rsksproc.has_multiple_days_and_dates(input_file)
        # to create alist of final_dates
        print("found these dates in the files:", input_file)
        print(dates)

        # loop to create a list of dates, string format and avoid duplicates
        # Use a set for fast lookup
        seen = set(final_dates)
        for d in dates:
            # ensure type matches (e.g., in case it's np.datetime64 or other)
            d_str = str(d)
            if d_str not in seen:
                final_dates.append(d_str)
                seen.add(d_str)

        if is_multiple:  # when having multiple dates in one rsk
            # print the file containing multiples
            print("found multiple dates in file:")
            print(input_file)
            # split the rsk if it is multiple, unique_days is a list of dates
            created_files = rsksproc.split_rsk_by_day(input_file, proc_data_path)

        else:  # when no multiple date
            # we have to rename _YYYYmmdd when our file is not duplicate
            # this is for the routine find duplicate

            # Construct filename
            day = np.datetime64(dates[0])
            day_str = str(day).replace("-", "")
            # filename = f"split_{day_str}.rsk"

            # Save the new RSK file
            with pyrsk.RSK(input_file) as rsk:
                rsk.readdata()
                # Writes the new rsk file in the same folder and keep the original one
                output_file = rsk.RSK2RSK(outputDir=proc_data_path,suffix=day_str)
                created_files.append(output_file)

    final_dates.sort(
        key=lambda d: datetime.strptime(d, "%Y-%m-%d")
    )  # sort dates chronological

    print("Identified SOMLIT dates in the group of files to process:")
    for date in final_dates:
        print(date)

    # remove duplicate files per day, some rsk files have the same day in it
    result = remove_duplicates(proc_data_path)

    # sort files by date
    sorted_kept = sort_files_by_yymmdd(result["kept"])
    sorted_deleted = sort_files_by_yymmdd(result["deleted"])

    print("New list of RSK files (one per day) to process:")
    for f in sorted_kept:
        print(f" - {os.path.basename(f)}")
    print("Deleted files because duplicated")
    for f in sorted_deleted:
        print(f" - {os.path.basename(f)}")

    return sorted_kept

# %% remove_duplicates


def remove_duplicates(path_in):
    '''
    This functions scans a list of rsk files in a given folder and removes
    the rsk files that are duplicated. It outputs the files deleted and the
    files kept.

    Parameters
    ----------
    path_in : str
        folder path containing the list of rsk files to check for duplicates

    Returns
    -------
    dict
        Dictionnay with two keys: 
            'kept': list of files kept
            'deleted': list of files deleted becuse identified as duplicates
    '''
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
            print(
                f"✅ Only one file for date {date_str}: {
                    os.path.basename(files[0])}"
            )
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
            date_str = fname.split("_")[-1].replace(".rsk", "")
            return datetime.strptime(date_str, "%Y%m%d")
        except:
            return datetime.min  # fallback if pattern doesn't match

    return sorted(files, key=extract_date)


# %% _rsk2csv_with_retry

def _rsk2csv_with_retry(rsk_obj, channels, profiles, comment, output_dir,
                         max_retries=3, delay=1.5):
    '''
    Thin wrapper around RSK.RSK2CSV() that retries on PermissionError.

    On Windows, a freshly created/deleted file inside a folder synced by
    OneDrive (or scanned by antivirus real-time protection) can be
    transiently locked for a fraction of a second right after it's written,
    causing a "[Errno 13] Permission denied" even though the destination
    folder itself is perfectly writable. A short retry/backoff is enough to
    ride out this kind of transient lock instead of aborting the whole file's
    processing (which was silently skipping the SOMLIT export downstream).

    Parameters
    ----------
    rsk_obj : RSK object
        rsk_d or rsk_u, the object whose RSK2CSV() method is being called
    channels, profiles, comment, output_dir :
        same arguments as RSK.RSK2CSV()
    max_retries : int, optional
        number of attempts before giving up and re-raising. Default 3.
    delay : float, optional
        seconds to wait between attempts. Default 1.5.

    Returns
    -------
    None.
    '''
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            rsk_obj.RSK2CSV(
                channels=channels,
                profiles=profiles,
                comment=comment,
                outputDir=output_dir,
            )
            return
        except PermissionError as e:
            last_err = e
            print(f"[WARNING] Accès refusé en écriture (probablement un verrou "
                  f"temporaire OneDrive/antivirus) — tentative {attempt}/{max_retries}, "
                  f"nouvel essai dans {delay}s. Détail: {e}")
            time.sleep(delay)
    # all retries exhausted, re-raise so the caller's try/except still logs the failure
    raise last_err


# %% *** procRSK ***
def procRSK(path_in, patm, site_id, p_tresh, c_tresh, param, path_out,
            ct_lag=-0.045, cond_scale_factor=1.000000, cond_offset=0,
            do_sensor_offset=0.472):
    """
    This function is to process a raw rsk file containing one single SOMLIT 
    experiment on one single day. It applies all the required processing on the
    RBR data as per Harlveston, 2017 and Mathieu Dever (RBR chief scientist)
    instructions.
    It works for RBR Maestro rsk file with Tridente channels 8,9,10 identified 
    and/or unidentified.
    NB: Does not work if the rsk file contains several SOMLIT days 
    => first split the rsk with the tools provided

    Parameters
    ----------
    path_in : str
        rsk file name coming from RBR Maestro (CTD, RBR CODA, RBR Tridente,
                                               pH, PAR,...)
    patm : Float
        atmospheric pressure (dBar)
    site_id : int
        Somlit unique site_id of the marine station that logged the rsk file
        linked to latitude values via sites.py
    p_tresh : float
        pressure teshold (dBar) parameter used in pyRSKtools.computeprofile()
    c_tresh : float
       conductivity treshold (mS/cm) parameter used in pyRSKtools.computeprofile() 
    param : List
        List of channel names that you want to keep in your processed RBR. csv
        destination file. This is now treated as a "wishlist": channels that
        are not physically present on this particular probe (e.g. no DO sensor
        on a Concerto) are automatically skipped instead of raising an error.
    path_out : str
        Location path to the output folder of your choice for the processed data
    ct_lag : float, optional
        time lag (s) applied to the temperature channel to realign it with
        conductivity (sensor response time mismatch). Default is the value
        historically used for the Maestro (-0.045 s). Adjust per probe if needed.
    cond_scale_factor : float, optional
        scale factor (SF) applied to conductivity to correct for sensor proximity
        effects. Default 1.0 (no correction).
    cond_offset : float, optional
        offset applied to conductivity for the same proximity correction. Default 0.
    do_sensor_offset : float, optional
        distance (m) between the CTD and the DO optode, used to compute the
        advective lag correction. Only used if a DO sensor is detected on this
        probe; ignored otherwise (e.g. Concerto without oxygen sensor).

    Returns
    -------
    raw : RSK object
        RSK object that has been through the recomputeprofile process only
    rsk : RSK object
        RSK object containing the fully processed data of the profile
    rsk_d : RSK object
        RSK object containing the downcast data only
    rsk_u : RSK object
        RSK object containing the upcast data only
    profile_nb : int
        Profile number identified in the rsk file for this Somlit
    file_output_folder : str
        Path to the output folder name for the specific profile
    csv_d : str
        Export filename of downcast
    csv_u : str
        Export filemane of upcast

    """
    with pyrsk.RSK(path_in) as rsk:

        # read the data first
        rsk.readdata()

        # Removes atmopsheric pressure patm to calculate hydrostatic sea pressure
        # In an ideal way the barometric pressure must be measured at each somlit and entered here
        # remind: -1hPa (air pressure) = +1cm sealevel
        # -100hPa = -1dbar = +1m sealevel
        rsk.deriveseapressure(patm)

        # Computing profiles

        # raw is a copy of the raw data to compare with the processed ones later
        # the parameters adapted for very shallow profile like somlit
        # with acclimatation time are (0.05,5)
        # decreasing the treshold p_treshold detects more profiles
        raw = rsk.computeprofiles(p_tresh, c_tresh)

        # Correct for A2D (analog to digital) zero-holder,
        # find the missing samples and interpolate
        # You must compute profile first before doing this otherwise error!!
        rsk.correcthold(action="interp")

        # identify proper profile number of interest
        profile_nb = find_profile(rsk)
        print("procrsk profile nb is" + str(profile_nb))

        # Detect logical channels really available on this probe (auto-detection,
        # works for any RBR probe: Maestro, Concerto, Duo...). This lets the rest
        # of the function skip processing steps for sensors that are physically
        # absent (e.g. no DO optode / no fluorometer on a Concerto) instead of
        # crashing on a hardcoded channel name.
        available_channels = channel_utils.get_available_channels(rsk)
        print(f"Channels detected in {os.path.basename(path_in)}: {available_channels}")

        has_do_sensor = (
            channel_utils.has_channel(rsk, "dissolved_o2_concentration")
            and channel_utils.has_channel(rsk, "temperature1")
        )
        if not has_do_sensor:
            print(f"[INFO] No DO sensor (dissolved_o2_concentration + temperature1) "
                  f"detected on {os.path.basename(path_in)} — DO compensation will be skipped.")

        # Low-pass filtering, windowlength is the number of values to use to calculate an average
        # We run at 2Hz, it is slower than the RBR (4Hz) so we won't apply any filter
        # rsk.smooth(channels = ["temperature"], windowLength = 5)

        # realignement CT (M. Dever)
        # time lag of the temperature sensor that is slower than the conductivity sensor
        # we talk about sensitivity speed with water changes
        deltat = ct_lag  # in second, per-probe tunable (see procRSK parameters)
        # Select the channel to be corrected
        var = "temperature"
        rsk.alignchannel(channel=var, lag=deltat, lagunits="seconds")

        # Sensor proximity effect on conductivity (M. Dever)
        # Apply potential known proximity effect correction to conductivity
        # Select the channel to be corrected
        var = "conductivity"

        # proximity impact results in a scalar correction to conductivity
        # SF=1 means no scale factor applied
        # offset = 0 means no offset applied
        # values may change with logger in cage or cond. sensor near metallic/plastic objects
        SF = cond_scale_factor
        offset = cond_offset

        if (SF != 1) | (offset != 0):

            rsk.data[var] = SF * rsk.data[var] + offset

            # inserted in metadata info
            logentry = (
                "in-situ adjustements to the "
                + var
                + " channel using a scaling factor of "
                + str(SF)
                + ", and an offset of "
                + str(offset)
                + "."
            )
            rsk.appendlog(rsk, logentry)

        # Compute CTD secondary variables
        # first derivedepth to calculate depth from corrected sea pressure
        # latitude of somlit point comes from the dictionnary sites.py,
        # looked up by its actual "id" field (see sites.get_site() docstring:
        # sites.sites[site_id] was indexing the list by POSITION, which
        # silently returned the wrong station for some ids and raised
        # IndexError for others, e.g. site_id=22 for Sète).
        rsk.derivedepth(
            sites.get_site(site_id)["latitude"], seawaterLibrary="TEOS-10"
        )

        # derive velocity , calculate velocity from depth and time
        # possible to add an argument here to do a window average of the salinity
        # not needed here as we go slow
        rsk.derivevelocity()
        rsk.derivesalinity()
        rsk.derivesigma()

        # Compensate DO values from salinity (jan.2026) from M.Dever ODO_EcoCTD script
        # This whole block only makes sense if the probe actually has a DO optode
        # (e.g. Maestro with Tridente/ODO). A Concerto (CTD + PAR only) or any
        # other probe without oxygen simply skips this step.
        if has_do_sensor:

            # SUB-SAMPLING DO and temperature data from 8Hz at 1Hz
            # we create newvariables DOXY but later on we will add them to new channels
            # names resolved via channel_utils in case the real channel is stored
            # under an alias (e.g. "odo_temperature" instead of "temperature1")
            DOXY = channel_utils.get_channel_array(rsk, "dissolved_o2_concentration")
            DOXY_raw = np.full_like(DOXY, np.nan)
            DOXY_raw[::8] = DOXY[::8]
            DOXY = DOXY_raw

            # observe that temperature1 is the temp logger from the DO sensor
            DOXY_TEMP = channel_utils.get_channel_array(rsk, "temperature1")
            DOXY_TEMP_raw = np.full_like(DOXY_TEMP, np.nan)
            DOXY_TEMP_raw[::8] = DOXY_TEMP[::8]
            DOXY_TEMP = DOXY_TEMP_raw

            # Pressure compensation of DOXY data (if needed)
            c0 = 3.2e-5
            Fcp = 1 + c0 * (rsk.data["sea_pressure"] - patm)
            DOXY = DOXY * Fcp

            # Salinity compensation (if needed)
            S = rsk.data["salinity"]
            Ts = np.log(
                (298.15 - rsk.data["temperature"])
                / (273.15 + rsk.data["temperature"])
            )
            Fcs = np.exp(
                S
                * (
                    -6.24097e-3
                    - 6.93498e-3 * Ts
                    - 6.90358e-3 * Ts**2
                    - 4.29155e-3 * Ts**3
                )
                - 3.11680e-7 * S**2
            )
            DOXY = DOXY * Fcs

            # Vertical alignment of DO measure
            time = rsk.data["timestamp"]
            time_sec = (time - time[0]) / np.timedelta64(1, "s")  # numeric seconds
            # dp_dt is the vertical profiling speed
            dp_dt = np.concatenate(
                (
                    [0],
                    # since time is already in seconds
                    np.diff(rsk.data["sea_pressure"]) / np.diff(time_sec),
                )
            )
            # set of a min speed 0.1
            dp_dt[np.abs(dp_dt < 0.1)] = 0.1

            # distance between CTD and optode [m], per-probe tunable
            optode_offset = do_sensor_offset

            # compute advective lag [in s]; that is the time it takes for a water parcel to
            # travel the "offset" distance, given the profiling speed "dpdt"
            lag_adv = optode_offset / dp_dt

            mask = ~np.isnan(DOXY)  # boolean showing true for nonNan values only

            # a shift of DOXY time is done to align with CTD measurements
            DOXY[mask] = np.interp(
                time_sec[mask] + lag_adv[mask], time_sec[mask], DOXY[mask]
            )
            # a shift of temperature1 time is done to align with CTD measurements
            DOXY_TEMP[mask] = np.interp(
                time_sec[mask] + lag_adv[mask], time_sec[mask], DOXY_TEMP[mask]
            )

            # create a new channel to attribute the DO corrected data calculated above
            # and the corresponding temperaure1 serie
            rsk.addchannel(
                DOXY, "dissolved_o2_compensated", units="µmol/L", isMeasured=0, isDerived=1
            )
            rsk.addchannel(
                DOXY_TEMP,
                "temperature1_compensated",
                units="°C",
                isMeasured=1,
                isDerived=0,
            )

        # Remove loops, this functions removes data or put it "nan". They must be handled later on
        # removing loops due to swell and probe measuring its wake
        # this might important in shallow coastal waters just as somlit location
        # speed treshold 0.05m/s mini profiling speed to consider
        # this values is important as all the data below this speed value are removed
        # waveheight and period may affect the treshold value
        # probably a parameter to be chosen by somlit user
        # sometimes 50cm amplitude swell at ste anne porzic
        # the key is the distance between the treshold and the real speed
        # the slower you go the more affected you are by the swell
        # if you are slow, put a treshold near you speed value
        # if you are fast, treshold near 0, to be confirmed
        #
        # BUGFIX: must be restricted to profiles=profile_nb, exactly like the
        # trim() call just below already does. Without it, removeloops()
        # processes EVERY profile region that computeprofiles() detected in
        # the whole file - including tiny spurious ones (e.g. the probe
        # dipped in/out of the water a couple of times right before the real
        # cast, each dip briefly toggling conductivity and getting counted as
        # its own 2-3 sample "profile"). pyrsktools's own removeloops()
        # implementation indexes into these profiles assuming at least 2
        # points, and raises "IndexError: index 1 is out of bounds for axis 0
        # with size 1" on any profile that only has 1 sample once trimmed -
        # observed on an RBR Concerto cast preceded by such calibration dips.
        rsk.removeloops(profiles=profile_nb, direction="down", threshold=0.05)

        # trim the data to remove unwanted values out of range
        rsk.trim(
            reference="salinity",
            range=(0, 1),
            profiles=profile_nb,
            direction="both",
            action="remove",
        )
        # create a copy to run independant processes for binaveraging and export
        # on chosen up and down casts
        rsk_u = rsk.copy()
        rsk_d = rsk.copy()

        # perhaps write a loop for the part below

        # Binning parameters to achieve one value per 0.25 m
        # choose bin Size here and depth limits for the binning process
        bin_size = 0.25
        start_depth = 0.75
        start_d = start_depth - (bin_size / 2)
        # this parameter must be checked for not loosing data on downcast
        # on somlit the profile is short (shallow depth) and we don't want to loose data
        end_d = int(np.nanmax(rsk_d.data["depth"]) // bin_size) * bin_size

        # contact RBR to do the binning on both up and down
        # Binning down cast
        # bin average on depth 0.25dbar or 25 cm for DOWN cast
        rsk_d.binaverage(
            profiles=profile_nb,  # we apply the binning only on our profile of interest
            binBy="depth",
            binSize=bin_size,
            boundary=[start_d, end_d],  # parameter to start at 0.75m depth
            # be carefull, choose start depth - binsize/2 to have the value starting the boundary
            direction="down",
        )

        # Binning up cast
        rsk_u.binaverage(
            profiles=profile_nb,  # we apply the binning only on our profile of interest
            binBy="depth",
            binSize=bin_size,
            # parameter to start at 0.75m depth, handking "nan' values
            boundary=[
                start_depth,
                round(np.nanmax(rsk_u.data["depth"])) + start_d,
            ],
            # be carefull, choose start depth - binsize/2 to have the value starting the boundary
            direction="up",
        )

        # create a subfolder for this specific rsk file
        # Extract the base filename without extension
        base = os.path.splitext(os.path.basename(path_in))[0]
        file_output_folder = os.path.join(path_out, base)
        os.makedirs(file_output_folder, exist_ok=True)
        print("file output folder" + file_output_folder)

        # create new folder for destinations csv files
        # we want to go a directory up from raw data folder
        newpath_u = file_output_folder + "/upcast"
        newpath_d = file_output_folder + "/downcast"

        # BUGFIX: previously both folders were only created if NEITHER existed,
        # so on a re-run where only one of the two was present (e.g. partial
        # previous run, or one manually deleted), the missing one was never
        # created and the corresponding RSK2CSV() call below would fail.
        # Each folder is now created independently and idempotently.
        os.makedirs(newpath_u, exist_ok=True)
        os.makedirs(newpath_d, exist_ok=True)

        # Filter the requested channel "wishlist" down to what actually exists on
        # this probe (auto-detection, works for Maestro, Concerto, or any other
        # RBR probe). Channels not present (e.g. dissolved_o2_concentration, ph,
        # chlorophyll-a, fdom, turbidity on a Concerto) are skipped with a log
        # message instead of making RSK2CSV fail.
        exported_param = channel_utils.filter_existing_channels(rsk_d, param)
        skipped_param = channel_utils.missing_channels(rsk_d, param)
        if skipped_param:
            print(f"[INFO] Channel(s) requested but not present on this probe, "
                  f"skipped for {os.path.basename(path_in)}: {skipped_param}")

        # save required variables in a csv with the correct format
        # export down cast (retry-wrapped: see _rsk2csv_with_retry() for why)
        _rsk2csv_with_retry(
            rsk_d,
            channels=exported_param,  # filtered to what this probe actually has
            profiles=profile_nb,
            comment="down CAST",
            output_dir=newpath_d,
        )
        # save export file name down cast because rsk2csv does not output it
        csv_d = rsk_to_profile_csv(newpath_d, 0)

        # export upcast
        _rsk2csv_with_retry(
            rsk_u,
            channels=exported_param,
            profiles=profile_nb,
            comment="up CAST",
            output_dir=newpath_u,
        )
        # save export file name down cast because rsk2csv does not output it
        csv_u = rsk_to_profile_csv(newpath_u, 0)

        # output
        return (
            raw,
            rsk,
            rsk_d,
            rsk_u,
            profile_nb,
            file_output_folder,
            csv_d,
            csv_u,
            exported_param,
        )


# %% process_rsk_folder
def process_rsk_folder(path_in, list_of_rsk, site_id, p_tresh, c_tresh, patm, param, **rsk_kwargs):
    '''
    This function is to procees a list of files in a chosen folder and apply
    the function process_rsk_file on each file
    # list of correct rsk files to process is given in argument
    # the loop does not properly work certainly because of the variable of the profle_nb that does not update in the loop.

    Parameters
    ----------
    path_in : str
        Location of the folder with the rsk files to process
    list_of_rsk : list
        List of the names of the rsk files to process
    site_id : int
        Site id of the Somlit point, cf. sites.py
    p_tresh : float
        pressure teshold (dBar) parameter used in pyRSKtools.computeprofile()
    c_tresh : float
       conductivity treshold (mS/cm) parameter used in pyRSKtools.computeprofile() 
    patm : Float
        atmospheric pressure (dBar)
    param : List
        List of channel names that you want to keep in your processed RBR. csv
        destination file. Treated as a wishlist: channels absent on a given
        probe (e.g. Concerto without DO/fluorometer) are auto-skipped.
    **rsk_kwargs :
        optional per-probe tuning forwarded to procRSK(), e.g. ct_lag=,
        cond_scale_factor=, cond_offset=, do_sensor_offset=. See procRSK()
        docstring for details. Leave empty to keep the historical Maestro defaults.

    Returns
    -------
    None.

    '''

    # assuming the rsk files are in a raw data folder given, we want to store
    # the processes_data in a proc_data dir
    # get the dir a step up
    # parent_dir = os.path.dirname(path_in)
    
    path_out = os.path.join(path_in, "outputs")
    # creates the path_out directory woth proc_data if it don't already exists
    os.makedirs(path_out, exist_ok=True)
    
    # Extract just the filenames from the list
    # rsk_filenames_from_list = {os.path.basename(path) for path in list_of_rsk}
    
    # List of all rsk files in target directory
    # all_files_in_dir = os.listdir(path_in)
    
    
    '''
    # finding the matching rsk files between two lists    
    valid_files_to_process = [
        os.path.join(path_in, f)
        for f in all_files_in_dir
        if f in rsk_filenames_from_list
    ]
    
    
    print("list_of_rsk:", list_of_rsk)
    print("rsk_filenames_from_list:", rsk_filenames_from_list)
    print("all_files_in_dir:", all_files_in_dir)
    print("valid_files_to_process:", valid_files_to_process)
    '''
    # no need to filter as done previously
    valid_files_to_process = list_of_rsk
    
    print("🔄 List of files to be processed:")
    valid_sorted = sort_files_by_yymmdd(
        valid_files_to_process
    )  # sort by increasinf date
    for f in valid_sorted:
        print(f" - {os.path.basename(f)}")  # show the list of files to process

    for i, input_file in enumerate(valid_sorted):
        print(
            f"\n--- Processing file {i +
                                     1}/{len(valid_files_to_process)}: {input_file} ---"
        )
        rsksproc.process_rsk_file(
            input_file, path_out, site_id, p_tresh, c_tresh, patm, param,
            **rsk_kwargs
        )


# %% process_rsk_file
#
def process_rsk_file(input_file, path_out, site_id, p_tresh, c_tresh, patm, param, **rsk_kwargs):
    '''
    This function does the processing on a single rsk file only. The rsk file
    is supposed to contain only one profile
    It is the same as process_rsk_folder but applied for one file only
    It creates the figures and csv in a folder nammed after the file name

    Parameters
    ----------
    input_file : str
        RSK file nme to process
    path_out : str
        Folder path to store the processing results  
    site_id : int
        Site id of the Somlit point, cf. sites.py
    p_tresh : float
        pressure teshold (dBar) parameter used in pyRSKtools.computeprofile()
    c_tresh : float
       conductivity treshold (mS/cm) parameter used in pyRSKtools.computeprofile() 
    patm : Float
        atmospheric pressure (dBar)
    param : List
        List of channel names that you want to keep in your processed RBR. csv
        destination file. Treated as a wishlist: channels absent on this probe
        (e.g. Concerto without DO/fluorometer) are auto-skipped for both the
        CSV export and the plots.
    **rsk_kwargs :
        optional per-probe tuning forwarded to procRSK() (ct_lag, cond_scale_factor,
        cond_offset, do_sensor_offset). Leave empty for the historical Maestro defaults.

    Returns
    -------
    None.

    '''
    # Extract the base filename without extension
    base = os.path.splitext(os.path.basename(input_file))[0]

    # Create a subfolder for this file
    file_output_folder = os.path.join(path_out, base)
    os.makedirs(file_output_folder, exist_ok=True)

    # Define paths for intermediate and final output
    final_csv_d = os.path.join(
        file_output_folder + "/downcast", f"{base}_4somlit_d.csv"
    )
    final_csv_u = os.path.join(
        file_output_folder + "/upcast", f"{base}_4somlit_u.csv"
    )
    try:
        print(f"🔄 Processing: {input_file}")

        # Step 1: Convert .rsk to .csv
        (
            raw,
            rsk,
            rsk_d,
            rsk_u,
            profile_nb,
            file_output_folder,
            csv_d,
            csv_u,
            exported_param,
        ) = rsksproc.procRSK(
            input_file, patm, site_id, p_tresh, c_tresh, param, path_out,
            **rsk_kwargs
        )

        print(f"Output folder: {file_output_folder}")
        print(f"CSV D path: {csv_d}, CSV U path: {csv_u}")
        print(f"Profile number: {profile_nb}")

        # Step 2: Plot
        # Use exported_param (already filtered to what this specific probe has,
        # e.g. no ph/chlorophyll-a/fdom/turbidity/DO on a Concerto) instead of
        # the raw wishlist, so we never try to plot a channel that doesn't exist.
        exclude = ["pressure", "sea_pressure", "depth"]
        for p in [x for x in exported_param if x not in exclude]:
            rsksplt.plot_up_down2(
                rsk_d, rsk_u, p, profile_nb, file_output_folder
            )

        # Step 3: Convert to SOMLIT format
        rsksproc.toSomlitDB(csv_d, site_id, final_csv_d)
        rsksproc.toSomlitDB(csv_u, site_id, final_csv_u)

        print(f"✅ Done: Output in {file_output_folder}")
    except Exception as e:
        print(f"❌ Failed for {input_file}: {e}")


# %% rsk_to_profile_csv

def rsk_to_profile_csv(dir_path, profile_nb=0):
    '''
    Function to find the path of a csv file outputted from procRSK
    and to be used in somlit2db function

    Parameters
    ----------
    dir_path : str
        Directory path where is stored our rsk file
    profile_nb : int, optional
        profile number processed in this rsk file

    Returns
    -------
    str
        path of the RBR csv file containing the processed data from procRSK

    '''
    dir_path2 = Path(dir_path).parent
    base_name = dir_path2.name  # Get the last folder name
    filename = f"{base_name}_profile{profile_nb}.csv"
    return str(Path(dir_path) / filename)


# %% split_rsk_by_day

def split_rsk_by_day(mrsk_file, output_dir):
    '''
    Function to process mrsk file. A mrsk file is a RSK file containing multiple somlit days in it.
    It comes because the RBR probe has been set on pause between the Somlits. Therefore several somlit days
    have been saved in one rsk file.
    This function splits this mrsk file into one rsk file per day to enable the next steps of our routines

    Parameters
    ----------
    mrsk_file : str
        Path to the multiple RSK file to split
    output_dir : str
        Directory adress to save the newly created rsk

    Returns
    -------
    created_files : List
        List of the rsk files created, one per day

    '''
    # mrsk_file is a file containing several days, several somlits

    created_files = []  # list of created files
    with pyrsk.RSK(mrsk_file) as rsk:
        rsk.readdata()

        # Extract unique dates from the datetime column
        timestamps = rsk.data["timestamp"]  # numpy.datetime64 array

        dates = np.array([ts.astype("datetime64[D]") for ts in timestamps])
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
            day_str = str(day).replace("-", "")

            # Save the new RSK file
            output_file = day_rsk.RSK2RSK(outputDir=output_dir, suffix=day_str)  # Writes the file
            created_files.append(output_file)

            print(f"✅ Saved: {output_file}")

        return created_files


# %% toSomlitDB
#


# %% _read_text_lines_robust

def _read_text_lines_robust(file_path):
    '''
    Read all lines of a text file, tolerating the fact that RBR's own CSV
    export (RSK.RSK2CSV) writes files in Windows-1252, not UTF-8. Unit
    symbols in the header/comment lines such as "kg/m³", "µmol/L", "°C" are
    encoded as single bytes (e.g. 0xB3 for "³") that are not valid on their
    own in UTF-8, causing:
        UnicodeDecodeError: 'utf-8' codec can't decode byte 0xb3 ...
    when the file is opened with Python's default/UTF-8 text encoding.

    This tries utf-8 first (harmless if the file happens to be plain ASCII),
    then falls back to cp1252 (RBR's actual encoding), and finally latin-1
    as a last resort (it can decode any byte, so it never raises).

    Parameters
    ----------
    file_path : str
        path to the text/csv file to read

    Returns
    -------
    list of str
        the lines of the file, as returned by f.readlines()
    '''
    for encoding in ("utf-8", "cp1252"):
        try:
            with open(file_path, "r", encoding=encoding) as f:
                return f.readlines()
        except UnicodeDecodeError:
            continue
    with open(file_path, "r", encoding="latin-1") as f:
        return f.readlines()


# %% toSomlitDB
#


def toSomlitDB(file_path, site_id, output_file):
    '''
    Function to process a csv file with RBR processed data outputted from procRSK 
    It takes in charge up or downward cast depending on user choice
    It modifies the file to fit to Somlit database file format
    as per the specifications given by SOMLIT community

    Parameters
    ----------
    file_path : str
        CSV file generated by procRSK with either downcast or upcast data
    site_id : int
        Somlit site id where RBR data have been collected: cf sites.py
    output_file : str
        file name for the somlit file to output

    Returns
    -------
    None.

    '''
    # NB: uses a robust reader because RBR's RSK2CSV export is written in
    # cp1252, not UTF-8 (see _read_text_lines_robust docstring above).
    lines = _read_text_lines_robust(file_path)

    # Step 2: Find the last line starting with '//'
    header_line_idx = None
    for idx, line in enumerate(lines):
        if line.startswith("//"):
            header_line_idx = idx

    # Step 3: Read the file into DataFrame, skipping earlier lines
    # Beware of sep ',   ' 4 spaces after comma
    df = pd.read_csv(
        file_path,
        sep=",    ",
        skiprows=header_line_idx + 1,  # Skip all lines before actual data
        header=None,  # No header in data part
        engine="python",
        encoding="cp1252",
    )

    # Step 4: Set the header from the last '//' line
    column_names = lines[header_line_idx].lstrip("//").strip().split(",    ")
    df.columns = column_names

    # remove line with nan
    df = df.dropna()

    # Convert the first column to datetime using your format
    # NB: assign via the column label (not .iloc) — recent pandas versions can
    # infer this column as a strict StringDtype from read_csv, and in-place
    # assignment through .iloc then raises "Invalid value for dtype 'str'"
    # when handed datetime64 values. Assigning by label replaces the column
    # (and its dtype) outright instead of mutating it in place.
    first_col = df.columns[0]
    df[first_col] = pd.to_datetime(df[first_col], format="%Y-%m-%dT%H:%M:%S.%f")

    # Set the first column as the index, and avoid warning caused
    # by aving diffrent object types in the index: str, numerical, etc...
    df[df.columns[0]] = df[df.columns[0]].infer_objects()
    df.set_index(df.columns[0], inplace=True)

    # Assume df already has datetime index

    # 1. Extract date and time from index

    # For date only (no time)
    df["DATE"] = df.index.strftime("%Y-%m-%d")

    # For time only, without decimals in seconds
    df["HEURE"] = df.index.strftime("%H:%M:%S")

    # 2. Add first column filled with 5
    df.insert(0, "ID_SITE", site_id)

    # mapping to handle, changed in units, columns names etc with Ruskin versions
    mapping = {
        "temperature(°C)".lower(): "TEMPERATURE",
        "chlorophyll-a(ug/l)".lower(): "FLUORESCENCE",
        # handles µg/L instead of ug/L
        "chlorophyll-a(µg/L)".lower(): "FLUORESCENCE",
        "par(µmol/m²/s)".lower(): "PAR",
        "par(µMol/m²/s)".lower(): "PAR",
        "salinity(PSU)".lower(): "SALINITE",
        "depth(m)".lower(): "PROFONDEUR",
    }
    # Renames columns
    df.rename(columns=lambda x: mapping.get(x.lower(), x), inplace=True)

    # 3. Reorder columns so the first 3 columns are ID_SITE, DATE, HEURE
    # followed by the rest of the original columns (excluding old index)
    cols = [
        "ID_SITE",
        "DATE",
        "HEURE",
        "TEMPERATURE",
        "FLUORESCENCE",
        "PAR",
        "SALINITE",
        "PROFONDEUR",
    ]

    # The SOMLIT file format expects this fixed set of columns, but some probes
    # don't carry every sensor (e.g. a Concerto has no fluorometer, so there is
    # no FLUORESCENCE data). Rather than crash on a missing column, fill it with
    # NaN so the export still runs and simply leaves that field empty.
    missing_cols = [c for c in cols if c not in df.columns]
    if missing_cols:
        print(f"[INFO] Colonne(s) absente(s) (capteur non présent sur cette sonde), "
              f"remplie(s) avec NaN dans l'export SOMLIT: {missing_cols}")
        for c in missing_cols:
            df[c] = np.nan

    df = df[cols]

    # Round to specific decimal numbers per channel for Somlit output file

    df["TEMPERATURE"] = df["TEMPERATURE"].round(4)
    df["FLUORESCENCE"] = df["FLUORESCENCE"].round(0)
    df["PAR"] = df["PAR"].round(3)
    df["SALINITE"] = df["SALINITE"].round(4)
    df["PROFONDEUR"] = df["PROFONDEUR"].round(2)

    # 4. Prepare your multi-line header as a string
    header_lines = [
        "// SOMLIT somlit.fr;;;;;;;",
        "// RBR processing IUEM/MEDIMEER;"
        + datetime.now().strftime("%Y-%m-%d;%H:%M:%S")
        + ";;;;;",  # current time of processing
        "ID_SITE;DATE;HEURE;TEMPERATURE;FLUORESCENCE;PAR;SALINITE;PROFONDEUR",
        "//;(yyyy-mm-dd);(hh:mm:ss);°C;µg/l;µMol/m²/s;PSU;m",
        ";;;;;;;",
    ]

    # 5. Write the file with the custom header
    # - encoding="utf-8-sig": adds a UTF-8 BOM so Excel (Windows/French locale)
    #   auto-detects UTF-8 instead of falling back to cp1252 and mangling
    #   special characters like °, µ, ² in the units line.
    # - newline="": prevents Python's universal-newline translation from
    #   doubling up the line endings that pandas.to_csv() already writes
    #   (the classic "blank line between every row" bug on Windows).
    with open(output_file, "w", encoding="utf-8-sig", newline="") as f:
        # Write custom header lines. Terminated with "\r\n" to match what
        # pandas.to_csv() writes below for the data rows (its csv writer
        # defaults to "\r\n" regardless of OS) — keeps the whole file on one
        # consistent line-ending convention.
        for line in header_lines:
            f.write(line + "\r\n")
        # Write DataFrame to file with ; separator, no header (already written)
        df.to_csv(f, sep=";", index=False, header=False)


# %% remove_rsk_date_files

def remove_rsk_date_files(folder):
    '''
    Function to clear the _YYYYMMDD.rsk string at the end of the file names

    Parameters
    ----------
    folder : str
        Path to the folder where the rsk files to process are

    Returns
    -------
    None.

    '''

    # Regex pattern for _YYYYMMDD.rsk
    pattern = re.compile(r"^.*_\d{8}\.rsk$")

    # Loop over files in folder
    for filename in os.listdir(folder):
        if pattern.match(filename):
            filepath = os.path.join(folder, filename)
            try:
                os.remove(filepath)
                print(f"Deleted: {filepath}")
            except Exception as e:
                print(f"❌ Failed to delete {filepath}: {e}")