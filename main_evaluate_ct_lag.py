# -*- coding: utf-8 -*-
"""
Evaluate the effect of the C-T lag setting on the RBR - SBE difference,
for every date where both an RBR .rsk (raw_data/) and an SBE SOMLIT csv
(SBE_completed/) exist. Downcasts only.

For each raw .rsk the script:
  1. splits it into one .rsk per profile (export_profiles2rsk) in a
     temporary folder, and drops profiles already seen in another raw file
     (same probe serial and same start time),
  2. runs procRSK() + toSomlitDB() twice: with the historical fixed lag
     (BASELINE_CT_LAG) and with the current configuration
     (Parameters/processing_config.py),
  3. compares each SOMLIT downcast with the SBE file(s) of the same date,
     bin by bin on PROFONDEUR.

Outputs (in figures_comparaison_SBE_RBR/evaluation_ct_lag/):
  - ct_lag_evaluation.csv : one row per (RBR profile, SBE file, setting)
  - ct_lag_evaluation.png : max |dS| and debiased RMS of dS per date
  - <date>_<profile>_vs_<sbe>.png : dS and dT profiles, before / after

Raw files are only read, never modified.

Author: Aurelien Domeau (CNRS), with Claude
"""
import os
import sys
import shutil
import tempfile
import contextlib
import io
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MAIN_DIR = Path(__file__).resolve().parent
for subfolder in ("Functions", "Parameters"):
    sys.path.append(str(MAIN_DIR / subfolder))

import pyrsktools as pyrsk  # noqa: E402
import RSKsomlit_proc as rsksproc  # noqa: E402
import diff_RBR_SBE as diffSBErbr  # noqa: E402
import processing_config as cfg  # noqa: E402

# Historical setting, kept here as the "before" reference
BASELINE_CT_LAG = -0.045
SETTINGS = {
    "baseline_-0.045s": BASELINE_CT_LAG,
    f"config_{cfg.CT_LAG_MODE}": None,  # None -> processing_config.CT_LAG_MODE
}

SITE_ID = 22  # Sete; change for another SOMLIT site
PATM = 10.1325
P_TRESH = 0.4
C_TRESH = 5
PARAM = ["conductivity", "temperature", "pressure", "sea_pressure", "depth",
         "salinity", "density_anomaly", "par", "chlorophyll-a", "turbidity"]

# A bin is counted as a spike when |dS - median(dS)| exceeds this (PSU)
SPIKE_THRESHOLD_PSU = 0.1
# Pairs with |median dS| above this (PSU) are flagged as invalid references
MAX_MEDIAN_OFFSET_PSU = 1.0

RAW_DIR = MAIN_DIR / "raw_data"
SBE_DIR = MAIN_DIR / "SBE_completed"
OUT_DIR = MAIN_DIR / "figures_comparaison_SBE_RBR" / "evaluation_ct_lag"


def split_unique_profiles(raw_files, work_dir):
    """Split every raw .rsk into single-profile .rsk, dropping duplicates."""
    seen = set()
    unique = []
    for raw in raw_files:
        with contextlib.redirect_stdout(io.StringIO()):
            parts = rsksproc.export_profiles2rsk(str(raw), str(work_dir))
        for part in parts:
            with pyrsk.RSK(part) as rsk:
                rsk.readdata()
                key = (rsk.instrument.serialID, str(rsk.data["timestamp"][0]))
            if key in seen:
                continue
            seen.add(key)
            unique.append(part)
    return unique


def process_one(rsk_file, ct_lag, out_dir):
    """Run procRSK + toSomlitDB on one profile, return the SOMLIT downcast csv
    and the C-T lag actually applied."""
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        result = rsksproc.procRSK(rsk_file, PATM, SITE_ID, P_TRESH, C_TRESH,
                                  PARAM, str(out_dir), ct_lag=ct_lag)
        csv_d = result[6]
        somlit_d = os.path.join(os.path.dirname(csv_d), "somlit_d.csv")
        rsksproc.toSomlitDB(csv_d, SITE_ID, somlit_d)
    applied = [line for line in log.getvalue().splitlines()
               if line.startswith("C-T lag applied")]
    return somlit_d, (applied[0] if applied else "")


def compare(rbr, sbe):
    """Bin-by-bin RBR - SBE difference on common depths."""
    a = rbr[["PROFONDEUR", "SALINITE", "TEMPERATURE"]].copy()
    b = sbe[["PROFONDEUR", "SALINITE", "TEMPERATURE"]].copy()
    for df in (a, b):
        df["PROFONDEUR"] = df["PROFONDEUR"].round(2)
    b = b.groupby("PROFONDEUR", as_index=False).mean()
    m = a.merge(b, on="PROFONDEUR", suffixes=("_rbr", "_sbe")).dropna()
    m["dS"] = m["SALINITE_rbr"] - m["SALINITE_sbe"]
    m["dT"] = m["TEMPERATURE_rbr"] - m["TEMPERATURE_sbe"]
    return m.sort_values("PROFONDEUR")


def stats(m, n_sbe_bins):
    ds = m["dS"] - m["dS"].median()
    return {
        "n_bins_common": len(m),
        "n_bins_sbe": n_sbe_bins,
        "dS_median": m["dS"].median(),
        "dS_max_abs": m["dS"].abs().max(),
        "dS_rms_debiased": float(np.sqrt(np.mean(ds ** 2))),
        "n_spikes": int((ds.abs() > SPIKE_THRESHOLD_PSU).sum()),
        "dT_median": m["dT"].median(),
        "dT_max_abs": m["dT"].abs().max(),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_files = sorted(RAW_DIR.glob("*.rsk"))
    sbe = {}
    for f in sorted(SBE_DIR.glob("*.csv")):
        df = diffSBErbr.read_somlit_csv(str(f))
        if not df.empty:
            sbe.setdefault(df["DATE"].iloc[0], []).append((f.stem, df))

    work = Path(tempfile.mkdtemp(prefix="ct_lag_eval_"))
    rows = []
    try:
        profiles = split_unique_profiles(raw_files, work)
        print(f"{len(profiles)} unique profile(s) from {len(raw_files)} raw file(s)")
        for prof in profiles:
            results = {}
            for name, lag in SETTINGS.items():
                try:
                    csv, applied = process_one(prof, lag, work / name)
                    df = diffSBErbr.read_somlit_csv(csv)
                except Exception as e:  # keep going on other profiles
                    print(f"[WARNING] {os.path.basename(prof)} ({name}) failed: {e}")
                    continue
                if df.empty:
                    continue
                results[name] = (df, applied)
            if not results:
                continue
            date = next(iter(results.values()))[0]["DATE"].iloc[0]
            if date not in sbe:
                continue
            label = Path(prof).stem
            for sbe_name, sbe_df in sbe[date]:
                fig, ax = plt.subplots(1, 2, figsize=(9, 7), sharey=True)
                for name, (df, applied) in results.items():
                    m = compare(df, sbe_df)
                    row = {"date": str(date), "rbr_profile": label,
                           "sbe_file": sbe_name, "setting": name,
                           "ct_lag_applied": applied.replace("C-T lag applied to temperature: ", "")}
                    row.update(stats(m, sbe_df["PROFONDEUR"].round(2).nunique()))
                    rows.append(row)
                    ax[0].plot(m["dS"], m["PROFONDEUR"], "o-", ms=3, lw=0.8, label=name)
                    ax[1].plot(m["dT"], m["PROFONDEUR"], "o-", ms=3, lw=0.8, label=name)
                for a, xl in zip(ax, ("dS RBR - SBE (PSU)", "dT RBR - SBE (degC)")):
                    a.axvline(0, color="k", lw=0.5)
                    a.set_xlabel(xl)
                    a.grid(alpha=0.3)
                    a.legend(fontsize=8)
                ax[0].invert_yaxis()
                ax[0].set_ylabel("Depth (m)")
                fig.suptitle(f"{date} downcast\n{label} vs {sbe_name}", fontsize=9)
                fig.tight_layout()
                fig.savefig(OUT_DIR / f"{date}_{label[:40]}_vs_{sbe_name}.png", dpi=110)
                plt.close(fig)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    if not rows:
        print("No RBR/SBE pair found.")
        return
    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "ct_lag_evaluation.csv", index=False, float_format="%.4f")

    # Pairs whose median offset is far beyond any sensor bias are not a
    # valid reference (wrong file, broken SBE salinity...): reported, but
    # left out of the summary figure and statistics.
    bad = table["dS_median"].abs() > MAX_MEDIAN_OFFSET_PSU
    for _, r in table[bad].drop_duplicates(["rbr_profile", "sbe_file"]).iterrows():
        print(f"[WARNING] {r['date']} {r['sbe_file']}: median dS = {r['dS_median']:.2f} PSU, "
              f"pair excluded from the summary (check the SBE file).")
    valid = table[~bad].sort_values(["date", "rbr_profile", "sbe_file"])

    # Summary figure: one point per (date, RBR profile, SBE file)
    valid = valid.assign(pair=valid["date"] + " p" +
                         valid["rbr_profile"].str.extract(r"_profile_(\d+)_", expand=False) +
                         " / " + valid["sbe_file"].str.replace("_sete", "").str.replace(r"^\d{4}_CTD_", "", regex=True))
    pairs = list(dict.fromkeys(valid["pair"]))
    fig, ax = plt.subplots(2, 1, figsize=(max(8, 0.6 * len(pairs)), 8), sharex=True)
    for name, g in valid.groupby("setting", sort=False):
        x = [pairs.index(p) for p in g["pair"]]
        ax[0].plot(x, g["dS_max_abs"], "o-", label=name)
        ax[1].plot(x, g["dS_rms_debiased"], "o-", label=name)
    ax[0].set_ylabel("max |dS| (PSU)")
    ax[1].set_ylabel("RMS(dS - median) (PSU)")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend()
        a.set_ylim(bottom=0)
    ax[1].set_xticks(range(len(pairs)))
    ax[1].set_xticklabels(pairs, rotation=90, fontsize=8)
    fig.suptitle("RBR - SBE salinity difference, downcast, per RBR profile / SBE file")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ct_lag_evaluation.png", dpi=110)
    plt.close(fig)

    summary = valid.groupby("setting", sort=False)[
        ["dS_max_abs", "dS_rms_debiased", "n_spikes", "dS_median"]].agg(["median", "max"])
    print(summary.round(3).to_string())
    print(f"Results written to {OUT_DIR}")


if __name__ == "__main__":
    main()
