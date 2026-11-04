# -*- coding: utf-8 -*-
"""
Processing configuration for procRSK().

All tunable processing choices live here so that they can be changed
without editing the processing code, and so that each SOMLIT site / probe
can document its own settings.

Conductivity-temperature (C-T) lag
----------------------------------
The thermistor responds more slowly than the conductivity cell. If the two
channels are not realigned, salinity computed in a gradient (thermocline)
shows spikes: on a downcast into colder water the late temperature reads
too warm and salinity spikes low. With a 2 Hz RBR Concerto and 0.25 m bins
most bins hold a single sample, so these spikes go straight into the SOMLIT
output (diagnostic of 2026-08-24 and 2026-09-08, Sete: -0.61 PSU at 13.25 m).

The lag is a physical time (sensor response + flushing of the cell) and is
expressed in seconds. The sampling rate only sets its resolution: it can
only be resolved to one sample period (0.5 s at 2 Hz, 0.125 s at 8 Hz).

Sign convention: CT_LAG values are passed to pyRSKtools RSK.alignchannel()
on the temperature channel. A negative value shifts temperature earlier in
time, i.e. compensates a thermistor that lags the conductivity cell.

References
----------
- RBR, pyRSKtools documentation: RSK.alignchannel(), RSK.calculateCTlag().
- Halverson M., Jackson J., Richards C., Melling H., Brunsting R., Dempsey M.,
  Gatien G., Hamilton A., Hunt B., Jacob W., Zimmermann S. (2017).
  Guidelines for processing RBR CTD profiles. Canadian Technical Report of
  Hydrography and Ocean Sciences 314.
"""

# "auto"  : estimate the lag on each profile (downcast) by minimising
#           salinity spikiness, with CT_LAG_PROBE_DEFAULTS as fallback.
# a float : use this fixed lag (seconds) for every profile.
# None    : use the per-probe default from CT_LAG_PROBE_DEFAULTS.
CT_LAG_MODE = "auto"

# Search range for the automatic estimate, in seconds of thermistor delay
# (positive = temperature lags conductivity). The search is done in whole
# sample periods within this range.
CT_LAG_SEARCH_MIN_S = 0.0
CT_LAG_SEARCH_MAX_S = 2.0

# Only samples deeper than this sea pressure (dbar) are used to estimate the
# lag, to avoid surface noise, soak and air/water transitions.
CT_LAG_MIN_PRESSURE_DBAR = 2.0

# Window (samples) of the running median used as the "smooth" salinity
# reference when measuring spikiness.
CT_LAG_SPIKE_WINDOW = 7

# The automatic estimate is rejected (fallback to probe default) if the
# best lag does not reduce spikiness by at least this fraction compared to
# no correction. Avoids applying a random lag on profiles with no gradient.
CT_LAG_MIN_IMPROVEMENT = 0.10

# If salinity spikiness without any lag is already below this value (PSU),
# the profile has no usable gradient: the lag cannot be estimated and does
# not matter, so the probe default is used (e.g. homogeneous winter profile).
CT_LAG_MIN_SPIKINESS_PSU = 0.005

# Among lags whose spikiness is within this relative tolerance of the best
# one, the smallest lag is kept (avoids picking a large lag on a flat
# minimum).
CT_LAG_TOLERANCE = 0.05

# Fallback / fixed lag per instrument model, in seconds (alignchannel
# convention, negative = shift temperature earlier). Keys are matched as a
# case-insensitive prefix of rsk.instrument.model ("RBRconcerto³" ...).
# - RBRconcerto: -0.5 s, estimated on Sete 241494 @2 Hz (2026-08-24,
#   2026-09-08). To be confirmed on 8 Hz units.
# - RBRmaestro: -0.045 s, historical value used at Plouzane.
CT_LAG_PROBE_DEFAULTS = {
    "RBRconcerto": -0.5,
    "RBRmaestro": -0.045,
}
# Used when the probe model matches no key above.
CT_LAG_DEFAULT_S = -0.045
