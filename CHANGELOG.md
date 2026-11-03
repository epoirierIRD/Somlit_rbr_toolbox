# Changelog

All notable changes to this toolbox are listed here. Dates are ISO 8601.

## [Unreleased] - branch fix/binaverage-extremes

### Fixed
- Salinity spikes (up to -0.6 PSU vs SBE) in the binned RBR Concerto
  downcasts. Cause: the C-T lag was fixed at -0.045 s (Maestro value)
  while the 2 Hz Concerto thermistor lags the conductivity cell by about
  one sample (0.5 s). Most 0.25 m bins hold a single sample at 2 Hz, so
  binning did not smooth the spikes out.
- `alignchannel()` is now restricted to the profile of interest, which
  avoids "Time lag must be smaller than profile time length" on spurious
  short profiles.

### Added
- `Parameters/processing_config.py`: C-T lag settings (automatic estimate
  per profile, fixed value, or per-probe default).
- `estimate_ct_lag()` / `resolve_ct_lag()` in `RSKsomlit_proc.py`.
- `main_evaluate_ct_lag.py`: before/after RBR - SBE comparison on every
  date available.
- `.gitignore` for caches, generated outputs and raw data.
- This changelog.

### Changed
- `procRSK(ct_lag=None)` now defaults to the configuration file instead of
  -0.045 s.
- SOMLIT export keeps 3 decimals for FLUORESCENCE.
