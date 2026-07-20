# -*- coding: utf-8 -*-
"""
Created on Wed May 13 10:51:03 2026

@author: Domeau
"""
CHANNEL_ALIASES = {
    "temperature": ["temperature"],
    "conductivity": ["conductivity"],
    "pressure": ["pressure"],
    "sea_pressure": ["sea_pressure"],
    "depth": ["depth", "depth(m)"],
    "salinity": ["salinity"],
    "density_anomaly": ["density_anomaly", "specificGravity"],

    "temperature1": [
        "temperature1",
        "odo_temperature"
    ],

    "dissolved_o2_concentration": [
        "dissolved_o2_concentration",
        "oxygen_concentration"
    ],

    "par": ["par"],
    "ph": ["ph"],

    "chlorophyll-a": [
        "chlorophyll-a",
        "chlorophyll"
    ],

    "fdom": ["fdom"],

    "turbidity": [
        "turbidity",
        "ntu"
    ],

    # channels supplémentaires vus sur RBR Concerto (et potentiellement d'autres sondes)
    "speed_of_sound": ["speed_of_sound"],
    "specific_conductivity": ["specific_conductivity"],
    "dissolved_o2_saturation": ["dissolved_o2_saturation"],
    "velocity": ["velocity"],
}

# Channels dérivés/ajoutés par procRSK() une fois le traitement DO effectué.
# Ils ne sont présents sur AUCUNE sonde brute : ils n'existent que si le
# capteur d'oxygène (dissolved_o2_concentration + temperature1) est physiquement
# présent et que procRSK() a pu tourner la compensation DO dessus.
CHANNEL_ALIASES.update({
    "dissolved_o2_compensated": ["dissolved_o2_compensated"],
    "temperature1_compensated": ["temperature1_compensated"],
})
