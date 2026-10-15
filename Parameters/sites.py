#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 11:21:52 2025

@author: epoirier
modified: adomeau
"""

# Site IDs and correponding latitudes, names, list of dictionnaries

'''

#
site_latitudes = {
    1: 43.523, # Site 1: Marseille
    2: 45.762, # Site 2: Arcachon
    3: 47.674, # Site 3: Loire
    4: 48.745, # Site 4: Roscoff
    5: 48.350  # Site 5: Plouzane
}
#

# how to use
# latitude = site_latitudes[5]
# print(latitude)  # Output: 48.383

'''
sites = [
    {"id": 1, "station": "Wimereux", "latitude": 50.6875}, # Point C
    {"id": 3, "station": "Roscoff", "latitude": 48.7778}, # Astan
    {"id": 5, "station": "Brest", "latitude": 48.3589}, # Portzic
    {"id": 6, "station": "Arcachon", "latitude": 44.6641}, # Eyrac
    {"id": 7, "station": "Gironde", "latitude": 45.5167},# pk 86
    {"id": 10, "station": "Banyuls", "latitude": 42.4883}, # Sola
    {"id": 11, "station": "Marseille", "latitude": 43.2417}, # Frioul
    {"id": 12, "station": "Villefranche", "latitude": 43.6833}, # Point B
    {"id": 17, "station": "Luc-sur-Mer", "latitude": 49.3188}, # Luc-sur-Mer
    {"id": 18 , "station": "La Rochelle", "latitude": 46.0842}, # Antioche
    {"id": 19, "station": "Dinard", "latitude": 48.6333}, # Bizeux
    {"id": 22 , "station": "Sete", "latitude": 43.3267}, # Sete
    ]


# %% get_site

def get_site(site_id):
    '''
    Retourne le dictionnaire du site correspondant à site_id, en recherchant
    par la clé "id" (le vrai identifiant Somlit du site), PAS par position
    dans la liste `sites`.

    BUGFIX: le code appelait auparavant directement `sites.sites[site_id]`,
    ce qui utilisait site_id comme un INDICE DE LISTE. Comme la liste n'est
    pas triée/dense par id, ça retournait silencieusement le mauvais site
    (site_id=5 renvoyait "Banyuls" au lieu de "Brest", par coïncidence sans
    erreur) et plantait carrément avec IndexError dès que site_id dépassait
    la longueur de la liste (ex: site_id=22 -> "list index out of range",
    alors que Sète est bien dans la liste, juste pas à la position 22).

    Parameters
    ----------
    site_id : int
        identifiant Somlit du site, cf. la clé "id" dans la liste `sites`

    Returns
    -------
    dict
        le dictionnaire {"id", "station", "latitude"} du site demandé

    Raises
    ------
    ValueError
        si site_id ne correspond à aucune entrée de `sites`
    '''
    for site in sites:
        if site["id"] == site_id:
            return site

    valid_ids = [site["id"] for site in sites]
    raise ValueError(
        f"site_id={site_id} introuvable dans sites.py. "
        f"IDs valides: {valid_ids}"
    )