# -*- coding: utf-8 -*-
"""
Created on Wed May 13 10:55:28 2026

@author: adomeau
"""

from channel_config import CHANNEL_ALIASES


def get_available_channels(rsk):
    """Retourne la liste des noms de channels présents dans le fichier."""
    return [c.longName for c in rsk.channels]


def resolve_channel(rsk, logical_name):
    """
    Retourne le nom réel du channel correspondant à un nom logique.
    Exemple :
        resolve_channel(rsk, "chlorophyll-a")
        -> "chlorophyll-a"
        -> "chlorophyll"
        -> None
    """
    available = get_available_channels(rsk)

    # parcourir tous les alias possibles pour ce logical_name
    for candidate in CHANNEL_ALIASES.get(logical_name, [logical_name]):
        if candidate in available:
            return candidate

    return None


def has_channel(rsk, logical_name):
    """Retourne True si le channel existe."""
    return resolve_channel(rsk, logical_name) is not None


def filter_existing_channels(rsk, requested_channels):
    """
    Retourne uniquement les channels demandés qui existent réellement.
    """
    return [
        ch for ch in requested_channels
        if has_channel(rsk, ch)
    ]


def missing_channels(rsk, requested_channels):
    """
    Retourne les channels demandés qui n'existent PAS sur cette sonde.
    Utile pour logguer/avertir quand un channel est absent (ex: sonde Concerto
    sans oxygène ni fluorimètre) au lieu de planter plus loin dans le traitement.
    """
    return [
        ch for ch in requested_channels
        if not has_channel(rsk, ch)
    ]


def get_channel_array(rsk, logical_name):
    """
    Retourne le tableau de données rsk.data[...] correspondant à un nom logique
    de channel, en résolvant l'alias réel (ex: "temperature1" peut être stocké
    sous le nom "odo_temperature" selon la sonde/firmware, cf. channel_config.py).

    Parameters
    ----------
    rsk : RSK object
        objet RSK pyrsktools, données déjà chargées (rsk.readdata() effectué)
    logical_name : str
        nom logique du channel (cf. clés de CHANNEL_ALIASES)

    Returns
    -------
    array-like ou None
        les données du channel, ou None si le channel n'existe pas sur cette sonde
    """
    real_name = resolve_channel(rsk, logical_name)
    if real_name is None:
        return None
    return rsk.data[real_name]
