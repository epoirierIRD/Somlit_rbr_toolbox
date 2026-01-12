#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 13:47:16 2025

@author: epoirier
"""

import diff_RBR_SBE as diffSBErbr


''' 
# upcast and downcast together
# Great that works
'''
# upcast
df_rbr_2_up = diffSBErbr.read_RBR_csv(
    '../proc_data/outputs/231853_20251017_1321_profile_0_2025-10-14T10-06-15_to_2025-10-14T10-07-31/upcast/231853_20251017_1321_profile_0_2025-10-14T10-06-15_to_2025-10-14T10-07-31_profile0.csv')
df_sbe_up = diffSBErbr.parse_custom_sbe(
    '../raw_data/SBE19plusV2cast32lanveoc141025_modif_upcast.csv')
df_sbe_up = diffSBErbr.rename_sbe_columns(df_sbe_up, df_rbr_2_up)

# downcast
df_rbr_2_down = diffSBErbr.read_RBR_csv(
    '../proc_data/outputs/231853_20251017_1321_profile_0_2025-10-14T10-06-15_to_2025-10-14T10-07-31/downcast/231853_20251017_1321_profile_0_2025-10-14T10-06-15_to_2025-10-14T10-07-31_profile0.csv')
df_sbe_down = diffSBErbr.parse_custom_sbe(
    '../raw_data/SBE19plusV2cast32lanveoc141025_modif_downcast.csv')
df_sbe_down = diffSBErbr.rename_sbe_columns(df_sbe_down, df_rbr_2_down)

# comparison
diffSBErbr.plot_comparisons_up_down(
    df_sbe_down, df_rbr_2_down, df_sbe_up, df_rbr_2_up)
