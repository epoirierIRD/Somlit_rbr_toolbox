#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script file name: RSKsomlit_plt.py
Author: Etienne Poirier, IRD, Plouzané
Date created: 2025-06-02
Last update: 2026-01-08
Description: various plotting functions for RBR Maestro profile data 
used in Somlit experiments
"""

import pyrsktools as pyrsk
import matplotlib.pyplot as plt
import os

# custom lib
from sensor_uncertainties import get_uncertainty

# Plotting functions

# %%


def basic_rsk_plot(rsk_path, parameters):
    '''
    Function to plot a few parameters from a rsk file

    Parameters
    ----------
    rsk_path : str
        the path to the source rsk file
    parameters : list of strings
        parameters to plot 
        eg. ['conductivity', 'temperature']

    Returns
    -------
    None.

    '''
    # create an object with the rsk file
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

# %% plot_raw_proc


def plot_raw_proc(rsk_cast, rsk, param, cast, profile, uncertainty=None):
    '''
    Function to plot raw and processed RBR data for one channel from a RSK file

    Parameters
    ----------
    rsk_cast : RSK object
        RSK object of interest to plot. Has been bin averaged and processed
    rsk : RSK object
        RSK object raw, not processed, not bin averaged, => original data
    param : str
        channel of interest to plot
    cast : str
        is the direction, must be 'up' or 'down'
    profile : int
        is the profile number identified in procRSK
    uncertainty : float, optional
        is the uncertainty attached to the channel of interest. The default is None.
        uncertainties are found in sensor_uncertainties.py

    Returns
    -------
    fig : TYPE
        DESCRIPTION.
    axes : TYPE
        DESCRIPTION.

    '''

    # check for proper parameter cast
    # must be 'up' or 'down' nothing else
    if cast not in ("up", "down"):
        raise ValueError("cast must be 'up' or 'down'")

    # Plot original and processed profiles
    fig1, axes1 = rsk.plotprofiles(
        channels=[param], profiles=profile, direction=cast)
    fig2, axes2 = rsk_cast.plotprofiles(
        channels=[param], profiles=profile, direction=cast)

    # Merge both figures into one
    fig, axes = rsk.mergeplots(
        [fig1, axes1],
        [fig2, axes2],
    )
    # Customize and add error bars if uncertainty is provided
    for ax in axes:
        lines = ax.get_lines()
        if len(lines) < 2:
            continue
        # Original line (assumed first) and processed line (assumed second)
        # orig_line = lines[-2]
        proc_line = lines[-1]

        # Style processed data
        plt.setp(proc_line, linewidth=0.5, marker="o", markerfacecolor="w")

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
    plt.legend(labels=["Original data", "Processed data"])
    plt.title(f"{cast}_{param}")
    plt.show()
    return fig, axes

# %% plot_up_down


def plot_up_down(rsk_d, rsk_u, param, profile_nb):
    '''
    Function to plot downcast and upcast on the same graph.
    Developped because pyRSKtools.plotprofile option direction=both
    does not work. RBR knows. Function below is sober and does not include 
    uncertainties. version 2 is more complete plot_up_down2

    Parameters
    ----------
    rsk_d : RSK object
        RSK object with downcast data only
    rsk_u : RSK object
        RSK object with upcast data only
    param : str
        channel name to plot
    profile_nb : int
        profile number containing downcast and upcast of interest

    Returns
    -------
    fig : TYPE
        DESCRIPTION.
    axes : TYPE
        DESCRIPTION.

    '''

    fig1, axes1 = rsk_d.plotprofiles(
        channels=[param], profiles=profile_nb, direction='down')
    fig2, axes2 = rsk_u.plotprofiles(
        channels=[param], profiles=profile_nb, direction='up')

    fig, axes = rsk_d.mergeplots(
        [fig1, axes1],
        [fig2, axes2],
    )
    print(axes)
    for ax in axes:
        line = ax.get_lines()[-1]
        plt.setp(line, linewidth=0.5, linestyle='--')
        line.set_dashes([10, 5])
    plt.legend(labels=["Downcast", "Upcast"])
    plt.title(param)
    plt.show()
    return fig, axes

# %% plot_up_down2


def plot_up_down2(rsk_d, rsk_u, param, profile_nb, save_path=None):
    """
    Plot upcast and downcast profiles with uncertainty bars and styled circular markers.
    An improved version of plot_up_down

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
        print(f"[WARNING] No uncertainty defined for '{
              param}' — skipping error bars.")
        uncertainty = None

    # Generate individual plots
    fig1, axes1 = rsk_d.plotprofiles(
        channels=[param], profiles=profile_nb, direction='down')
    fig2, axes2 = rsk_u.plotprofiles(
        channels=[param], profiles=profile_nb, direction='up')

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

    # save the plots in files in a choosen save_path location
    if save_path:

        # Create 'figures' folder if it doesn't exist
        figures_dir = os.path.join(save_path, "figures")
        # won't raise error if it already exists
        os.makedirs(figures_dir, exist_ok=True)

        # Create a filename based on the parameter name
        save_filename = os.path.join(figures_dir, f"{param}.png")

        plt.savefig(save_filename, dpi=150, bbox_inches='tight')

    plt.show()

    return fig, axes
