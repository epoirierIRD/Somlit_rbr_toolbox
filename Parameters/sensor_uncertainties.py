#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 11:11:33 2025

@author: epoirier, with ChatGPT

Uncertainties for each sensor channel (values in physical units).
Applies to RBR Maestro units 231853 and 201759 from IUEM, Plouzané.
"""
# values written below will be plotted + and - on either side of the data on the graph
CHANNEL_UNCERTAINTIES = {
    'conductivity': 0.003,                   # mS/cm
    'temperature': 0.002,                    # °C
    'pressure': 0.375,                       # dbar
    'temperature1': 0.002,                   # °C
    'temperature1_compensated': 0.002,       # °C
    'dissolved_o2_concentration': 2.0,       # µmol/L
    'dissolved_o2_compensated': 2.0,                   # µmol/L
    'par': 1.4,                              # µMol/m²/s
    'ph': 0.1,                               # pH units (estimated)
    'chlorophyll-a': 0.1,                    # µg/L
    'fdom': 0.5,                              # ppb, +/- 5%
    'turbidity': 0.5,                         # FTU, +/-5%
    'sea_pressure': 0.375,                   # dbar (same as pressure)
    'depth': 0.375,                          # m
    'salinity': 0.01,                        # PSU (estimated)
    'speed_of_sound': 0,                     # m/s (unknown)
    'specific_conductivity': 0,              # µS/cm (unknown)
    'dissolved_o2_saturation': 0,            # % (unknown)
    'velocity': 0,                           # m/s  (unknown)
    'density_anomaly': 0                     # kg/m³ (unknown)
}


def get_uncertainty(channel_name):
    """
    Return the uncertainty for a given channel name.

    Parameters:
        channel_name (str): Name of the sensor channel.

    Returns:
        float: Uncertainty value for the specified channel.

    Raises:
        KeyError: If the channel is not found in the dictionary.
    """
    return CHANNEL_UNCERTAINTIES[channel_name]
