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

# custom lib
from sensor_uncertainties import get_uncertainty

# Plotting functions

# ****************************************************************************
# Basic plot of a few parameters from a rsk file

def basic_rsk_plot(rsk_path, parameters):
    # args:
    # rsk_path is a string with the path of the source rsk file
    # parameters is a list of parameters to plot
    # ex: ["conductivity", "temperature"]
    # outputs: nothing
    
    #create an object with the rsk file
    with pyrsk.RS(rsk_path) as rsk:    
        # Print a list of all the channels in the RSK file
        rsk.printchannels()
        # Read data
        rsk.readdata()
        # Derive sea pressure from total pressure
        rsk.deriveseapressure()
        # Plot a few profiles of temperature, conductivity, and chlorophyll
        fig, axes = rsk.plotprofiles(
            channels=parameters,
            profiles=0,
            direction="both",
            )
        plt.show()
    
# test code
'''
rsk_path = '/home/epoirier1/Documents/METROLOGIE/2024/intercomp_steanne_12092024/Maestro_P2I_231853/231853_20240912_1229.rsk'
# rsk_path = '/home/epoirier1/Documents/METROLOGIE/2024/intercomp_steanne_12092024/Maestro_P2I_236135/236135_20240912_1238.rsk'
parameters = ["conductivity", "temperature"]

basic_rsk_plot(rsk_path, parameters)
  
'''
# ************************************************************************
# Plot raw and processed data for one parameter from a RSK file

def plot_raw_proc(rsk_cast, rsk, param, cast, profile, uncertainty=None):
    # args:
    # rsk_cast is the rsk object of interest to plot that has been bin averaged (Processed data)
    # rsk is the rsk object non bin averaged data (Original data)
    # param is the channel to plot
    # cast is up or down
    # profile is the profil number
    # outputs:
    # tuple fig, axes
    
    # Plot original and processed profiles
    fig1, axes1 = rsk.plotprofiles(channels=[param],profiles=profile,direction=cast)
    fig2, axes2 = rsk_cast.plotprofiles(channels=[param],profiles=profile,direction=cast)
    
    # Merge both figures into one
    fig, axes = rsk.mergeplots(
             [fig1,axes1],
             [fig2,axes2],
         )
    # Customize and add error bars if uncertainty is provided
    for ax in axes:
        lines = ax.get_lines()
        if len(lines) < 2:
            continue
        # Original line (assumed first) and processed line (assumed second)
        orig_line = lines[-2]
        proc_line = lines[-1]
        
        # Style processed data
        plt.setp(proc_line, linewidth=0.5, marker = "o", markerfacecolor = "w")
        
        # Add horizontal error bars on processed data if uncertainty is specified
        if uncertainty is not None:
            xdata = proc_line.get_xdata()
            ydata = proc_line.get_ydata()
            ax.errorbar(
                x=xdata,
                y=ydata,
                xerr=uncertainty,
                fmt='none',
                ecolor='gray',
                elinewidth=0.5,
                capsize=2,
                alpha=0.7
            )
    plt.legend(labels=["Original data","Processed data"])
    plt.title(f"{cast}_{param}")
    plt.show()
    return fig, axes

# ************************************************************************
# Plot down cast and upcast on the same graph, the option is not working
# in plotprofile when setting direction=both, RBR has been contacted for that
# function below is sober and does not include uncertainties

def plot_up_down(rsk_d, rsk_u, param, profile_nb):
    # args:
    # rsk_d rsk data object of down cast
    # rsk_u rsk data object of up cast
    # param: variable to plot 
    # profile_nb: profile number where up and down cast of interest are stored
    # outputs:
    # tuple fig, axes
    
    fig1, axes1 = rsk_d.plotprofiles(channels=[param],profiles=profile_nb,direction='down')
    fig2, axes2 = rsk_u.plotprofiles(channels=[param],profiles=profile_nb,direction='up')

    fig, axes = rsk_d.mergeplots(
             [fig1,axes1],
             [fig2,axes2],
         )
    print(axes)
    for ax in axes:
        line = ax.get_lines()[-1]
        plt.setp(line, linewidth=0.5, linestyle = '--')
        line.set_dashes([10, 5])
    plt.legend(labels=["Down_cast","Up_cast"])
    plt.title (param)
    plt.show()
    return fig, axes

# Better function showing uncertainties, help with chatgpt
def plot_up_down2(rsk_d, rsk_u, param, profile_nb, save_path=None):
    """
    Plot upcast and downcast profiles with uncertainty bars and styled circular markers.

    Args:
        rsk_d: RSK object for downcast data.
        rsk_u: RSK object for upcast data.
        param (str): Parameter/channel name to plot.
        profile_nb (int): Profile number to use.
        save_path: place to save the plots.

    Returns:
        (fig, axes): Matplotlib figure and axes.
    """

    # Get uncertainty
    try:
        uncertainty = get_uncertainty(param)
    except KeyError:
        print(f"[WARNING] No uncertainty defined for '{param}' — skipping error bars.")
        uncertainty = None

    # Generate individual plots
    fig1, axes1 = rsk_d.plotprofiles(channels=[param], profiles=profile_nb, direction='down')
    fig2, axes2 = rsk_u.plotprofiles(channels=[param], profiles=profile_nb, direction='up')

    # Merge into a single figure
    fig, axes = rsk_d.mergeplots([fig1, axes1], [fig2, axes2])

    for ax in axes:
        lines = ax.get_lines()
        if len(lines) < 2:
            continue

        down_line = lines[-2]
        up_line = lines[-1]

        # Extract data
        x_down, y_down = down_line.get_xdata(), down_line.get_ydata()
        x_up, y_up = up_line.get_xdata(), up_line.get_ydata()

        # Style downcast line
        down_line.set_color('blue')
        down_line.set_linestyle('-')
        down_line.set_linewidth(1.0)
        down_line.set_marker('o')
        down_line.set_markerfacecolor('blue')
        down_line.set_markeredgecolor('black')
        down_line.set_markersize(5)

        # Style upcast line
        up_line.set_color('red')
        up_line.set_linestyle('--')
        up_line.set_linewidth(1.0)
        up_line.set_marker('o')
        up_line.set_markerfacecolor('red')
        up_line.set_markeredgecolor('black')
        up_line.set_markersize(5)

        # Add uncertainty bars if available
        if uncertainty:
            ax.errorbar(
                x_down, y_down,
                xerr=uncertainty,
                fmt='none',
                ecolor='blue',
                alpha=0.4,
                capsize=2,
                elinewidth=0.7,
                label='_nolegend_'
            )
            ax.errorbar(
                x_up, y_up,
                xerr=uncertainty,
                fmt='none',
                ecolor='red',
                alpha=0.4,
                capsize=2,
                elinewidth=0.7,
                label='_nolegend_'
            )

    # Finalize plot
    plt.legend(labels=["Downcast", "Upcast"])
    plt.title(param)
    plt.xlabel(param)
    plt.ylabel("Depth")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("my_plot.png", dpi=300)
    
    # save the plots in files in a choosen save_path location
    if save_path:        
        
        # Create 'figures' folder if it doesn't exist
        figures_dir = os.path.join(save_path, "figures")
        os.makedirs(figures_dir, exist_ok=True)  # won't raise error if it already exists
        
        # Create a filename based on the parameter name
        save_filename = os.path.join(figures_dir, f"{param}.png")
        
        plt.savefig(save_filename, dpi=150, bbox_inches='tight')


    plt.show()

    return fig, axes


'''
def plot_all(rsk, rsk_d, rsk_u, param, profile_nb):
    
    fig1, ax1 = plt.subplots()
    for cast in ['down','up']:
        if cast == 'down':
            fig1, ax1 = rsk.plotprofiles(channels=[param],profiles=profile_nb,direction=cast)  
            fig1, ax2 = rsk_d.plotprofiles(channels=[param],profiles=profile_nb,direction='down')
        else:
            fig1, ax3 = rsk.plotprofiles(channels=[param],profiles=profile_nb,direction=cast)
            fig1, ax4 = rsk_d.plotprofiles(channels=[param],profiles=profile_nb,direction='up')

    # for ax in (ax1,ax2,ax3,ax4):
    #     line = ax.get_lines()
    #     plt.setp(line[0], linewidth=0.5, label="Original data downcast")
    #     plt.setp(line[1], linewidth=0.5, marker="o", markerfacecolor="w", label="Processed data")
    #     plt.setp(line[2], linewidth=0.8, linestyle="--",label="Original data dupcast")  # dashed line
    #     line.set_dashes([10, 5])
    #     plt.setp(line[3], linewidth=0.5, marker="o", markerfacecolor="w", label="Processed data")
    plt.legend(labels=["Original data","Processed data"])
    plt.title (cast+'_'+param)
    plt.show()
'''
    
'''    
# bout de code pour visualiser un seul profil et 3 variables
with pyrsk.RSK("/home/epoirier1/Documents/PROJETS/2025/Proc_RBR_Somlit/rawdata/maestroP2I_231853_20240130.rsk") as rsk:
    rsk.readdata()
    rsk.deriveseapressure()
    rsk.derivesalinity()
    fig, axes = rsk.plotprofiles(

       channels=["conductivity", "temperature", "salinity"],

       profiles=range(0, 1),

       direction="down",

   )
   

plt.show()
'''








