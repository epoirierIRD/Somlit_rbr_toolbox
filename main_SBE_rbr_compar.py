#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 13:47:16 2025

@author: epoirier
Modifier: adomeau
"""

import sys
import traceback
from pathlib import Path


def main():
    MAIN_DIR = Path(__file__).resolve().parent  # Folder where this script lives

    # If the .py modules (diff_RBR_SBE.py, etc.) live in subfolders next to
    # this script rather than right beside it - e.g. a "Functions" folder
    # for the processing modules and a "Parameters" folder for sites.py -
    # add them to sys.path so the import below can find them. Built from
    # MAIN_DIR (relative to wherever this script actually sits) rather than
    # hardcoded to one machine's username/path, so this works unchanged on
    # any computer/user account. If you keep every .py file directly
    # alongside this script, these two folders simply won't exist and
    # nothing is added - no harm either way.
    for subfolder in ("Functions", "Parameters"):
        candidate = MAIN_DIR / subfolder
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.append(str(candidate))

    import diff_RBR_SBE as diffSBErbr

    # Dossier contenant les CSV Somlit produits par le pipeline RBR
    # (RSKsomlit_proc.toSomlitDB, appelé depuis main_process.py). Recherche
    # récursive de tous les "*_4somlit_*.csv" en dessous de ce dossier.
    RBR_OUTPUTS_DIR = MAIN_DIR / "proc_data" / "outputs"

    # Dossier où sont déposés les CSV SBE déjà au format Somlit (un fichier
    # par profil/jour). Recherche non récursive : tous les .csv directement
    # dans ce dossier.
    SBE_COMPLETED_DIR = MAIN_DIR / "SBE_completed"

    # Dossier de sortie pour les graphiques de comparaison, un sous-dossier
    # par date puis par paire de fichiers comparée.
    COMPARAISON_FIGURES_DIR = MAIN_DIR / "figures_comparaison_SBE_RBR"

    if not RBR_OUTPUTS_DIR.exists():
        raise FileNotFoundError(
            f"{RBR_OUTPUTS_DIR} n'existe pas. Lance d'abord main_process.py "
            f"pour générer les CSV Somlit à partir des fichiers RBR."
        )
    if not SBE_COMPLETED_DIR.exists():
        raise FileNotFoundError(
            f"{SBE_COMPLETED_DIR} n'existe pas. Crée ce dossier et places-y "
            f"les CSV SBE déjà au format Somlit."
        )

    '''
    Recherche automatique de toutes les paires (fichier Somlit RBR, fichier
    Somlit SBE) partageant la même date, puis génération des graphiques de
    comparaison (profil superposé + différence RBR-SBE, pour chaque
    paramètre commun aux deux capteurs) pour chaque paire trouvée.

    Si une date a plusieurs fichiers RBR (descente ET remontée par exemple)
    et/ou plusieurs fichiers SBE, toutes les combinaisons RBR/SBE de cette
    date sont comparées entre elles.
    '''
    diffSBErbr.compare_all_matching_dates(
        rbr_root=str(RBR_OUTPUTS_DIR),
        sbe_root=str(SBE_COMPLETED_DIR),
        save_folder=str(COMPARAISON_FIGURES_DIR),
    )

    print("Programme terminé")


if __name__ == "__main__":
    # IMPORTANT: this try/except/finally is what keeps the console window
    # open and shows the actual error when the script is launched by
    # double-clicking the .py file. Without it, an unhandled exception
    # closes the window instantly (standard Windows behaviour when running
    # a .py file directly), hiding whatever went wrong. When run from
    # Spyder/an IDE this makes no difference - the traceback is shown
    # either way - but it's essential for double-click / plain "python
    # main_SBE_rbr_compar.py" usage.
    try:
        main()
    except Exception:
        print("\n❌ Une erreur est survenue pendant le traitement :\n")
        traceback.print_exc()
    finally:
        input("\nAppuyez sur Entrée pour fermer...")