import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib.colors import LogNorm
from scipy.stats import ks_2samp, cramervonmises_2samp
from scipy.optimize import minimize_scalar
from astropy.time import Time
import ROOT

# Add custom module paths
sys.path.append('/storage/gpfs_data/pauger/users/jdejesus/my_python_modules/')


sys.path.append('/storage/gpfs_data/pauger/users/jdejesus/my_python_modules/auger')
from augerADSTgeneral import extract_sd_info

def extract_energy_icrc2025(
    events_id,
    photodetector,
    adst_path="/storage/gpfs_data/pauger/users/jdejesus/observer_productions/icrc-2025-test7/SDInfill_Phase1/",
    adst_files=("SDInfill_PhaseI_adst_wc_mini.root", ),
):
    """
    Extract SD energy (log10(E/eV)) for events, both PMT and SiPM, using the ICRC 2025-test7 Observer production.

    This function maps event identifiers (SD+MD) to their corresponding SD
    event IDs and retrieves the SD energy estimate from the ICRC 2025 ADST files.
    This is to be ensure that the two datasets -PMT and SiPM- use the same energy scale
    before further anaylsis

    Parameters
    ----------
    events_id : array-like of str (either for the PMT or SiPM )
        event ids of the form
        'auger_162670441700__sd_39643837__md_228'.

    photodetector : str 
        "pmt" or "sipm". Only relevant to put cuts in the gps seconds
        to improve computation time

    adst_path : str, optional
        Path to the directory containing the ICRC 2025 Observer production.

    adst_files : tuple of str, optional
        Names of the ADST files to be scanned.

    Returns
    -------
    dict
        Dictionary with keys:
            - 'event_id': UMD event IDs (SD+MD)
            - 'lge_icrc_2025': SD log10(E/eV) from ICRC 2025 reconstruction

    Requirements
    ------------
    - Auger Offline environment configured
    - AUGEROFFLINEROOT environment variable set
    - extract_sd_info(event) available in the namespace
    """

    if photodetector not in {"pmt", "sipm"}:
        raise ValueError("photodetector must be either 'pmt' or 'sipm'")

    # ------------------------------------------------------------------
    # Build SD -> MD event ID mapping
    # ------------------------------------------------------------------
    sd_to_md = {}
    for evt in events_id:
        sd_id = evt.split("__md")[0]
        sd_to_md[sd_id] = evt

    sd_event_ids = set(sd_to_md.keys())

    # ------------------------------------------------------------------
    # Check environment and load Offline libraries
    # ------------------------------------------------------------------
    if "AUGEROFFLINEROOT" not in os.environ:
        raise RuntimeError("AUGEROFFLINEROOT environment variable is not set")

    auger_offline_root = os.environ["AUGEROFFLINEROOT"]
    ROOT.gSystem.Load(os.path.join(auger_offline_root, "lib/libRecEventKG.so"))

    # ------------------------------------------------------------------
    # Check input files
    # ------------------------------------------------------------------
    adst_file_paths = []
    for fname in adst_files:
        full_path = os.path.join(adst_path, fname)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"ADST file not found: {full_path}")
        adst_file_paths.append(full_path)

    # ------------------------------------------------------------------
    # Output container
    # ------------------------------------------------------------------
    outdict = {
        "event_id": [],
        "lge_icrc_2025": [],
        "gps": [],
        "yymmdd": [],
        "photodetector": []
    }

    if photodetector == "pmt":
        date_1 = "2015-10-01T00:00:01"
        date_2 = "2016-11-01T00:00:01"
    else:
        date_1 = "2018-01-01T00:00:01"
        date_2 = "2022-03-01T00:00:01"

    utc_time_min = Time(date_1, scale="utc")
    utc_time_max = Time(date_2, scale="utc")
    
    gps_min, gps_max = utc_time_min.gps, utc_time_max.gps

    # ------------------------------------------------------------------
    # Loop over ADST files and events
    # ------------------------------------------------------------------
    for fname in adst_file_paths:

        files = ROOT.std.vector("string")()
        files.push_back(fname)

        rec_file = ROOT.RecEventFile(files)
        event = ROOT.RecEvent()
        geo = ROOT.DetectorGeometry()

        rec_file.ReadDetectorGeometry(geo)
        rec_file.SetBuffers(event)

        nn = 0 
        while rec_file.ReadNextEvent() == ROOT.RecEventFile.eSuccess:

            sd_event_id = str(event.GetEventId())
            yymmdd = event.GetYYMMDD()

            sd_info = extract_sd_info(event)

            nn += 1
            if nn % 5000 == 0:
                print(f"{sd_event_id}  {yymmdd}")

            if sd_info["gps_sec"]  > gps_max + 100:
                break

            if not (gps_min < sd_info["gps_sec"] < gps_max):
                continue

            # Keep only requested SD events
            if sd_event_id not in sd_event_ids:
                continue

            # Avoid duplicates if event appears in multiple files
            md_event_id = sd_to_md[sd_event_id]
            if md_event_id in outdict["event_id"]:
                continue

            

            outdict["event_id"].append(md_event_id)
            outdict["lge_icrc_2025"].append(sd_info["lgE"])
            outdict["gps"].append(sd_info["gps_sec"])
            outdict["yymmdd"].append(yymmdd)
            outdict["photodetector"].append(photodetector)


    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"n_input_events_id           = {len(events_id)}")
    print(f"n_events_found_icrc_2025    = {len(outdict['event_id'])}")
    print(f"n_unique_events_icrc_2025  = {len(set(outdict['event_id']))}")

    return outdict


# ================================================
# Load the csv module-level measurements
# ================================================


pmt_path = ("/storage/gpfs_data/pauger/users/jdejesus/kit/datasets/pmt/reproduced_dataset_2025/module_level_info/"
            "module_level_stratGap_nOnes2_nGaps1_wSize7_rejSmall0_rejCorr1_effSmall1.04_effLarge0.95_rMin200_minCounters3_ldfSara1_fixBeta1.csv")

sipm_path1 = ("/storage/gpfs_data/pauger/users/jdejesus/kit/datasets/sipm/data/module_level_info/"
            "module_level_info_sipm_tmin0_tmax2048_gpsMin1230336019_gpsMax1261872018.csv")

sipm_path2 = ("/storage/gpfs_data/pauger/users/jdejesus/kit/datasets/sipm/data/module_level_info/"
            "module_level_info_sipm_tmin0_tmax2048_gpsMin1261872019_gpsMax1275004818.csv")

sipm_path3 = ("/storage/gpfs_data/pauger/users/jdejesus/kit/datasets/sipm/data/module_level_info/"
            "module_level_info_sipm_tmin0_tmax2048_gpsMin1198800019_gpsMax1230336018.csv")

sipm_path4 = ("/storage/gpfs_data/pauger/users/jdejesus/kit/datasets/sipm/data/module_level_info/"
            "module_level_info_sipm_tmin0_tmax2048_gpsMin1275004819_gpsMax1293494418.csv")

sipm_path5 = ("/storage/gpfs_data/pauger/users/jdejesus/kit/datasets/sipm/data/module_level_info/"
            "module_level_info_sipm_tmin0_tmax2048_gpsMin1293494419_gpsMax1306540818.csv")

df_pmt = pd.read_csv(pmt_path)
#print(df_pmt.columns)
#print(df_pmt.info())

df1 = pd.read_csv(sipm_path1)
df2 = pd.read_csv(sipm_path2)
df3 = pd.read_csv(sipm_path3)
df4 = pd.read_csv(sipm_path4)
df5 = pd.read_csv(sipm_path5)

df_sipm = pd.concat([df1, df2, df3, df4, df5])
assert len(df_sipm) == len(df1) + len(df2) + len(df3) + len(df4) + len(df5)

# ------------------------------------------------------------
# ICRC 2025 energy for PMT and SiPM
# ------------------------------------------------------------

# Coarse cuts to avoid too much computation time in fetching the ICRC2025
# events
df_pmt = df_pmt.query("theta < 45 and lge > 17.1 and lge < 19")
df_sipm = df_sipm.query("theta < 45 and lge > 17.1 and lge < 19")


print("Fetching ICRC2025 energy for PMT events...")
events_id_pmt = np.unique(df_pmt["event_id"].to_numpy())
energy_icrc_2025_pmt = extract_energy_icrc2025(events_id_pmt, photodetector="pmt")

df_pmt_2025 = pd.DataFrame.from_dict(energy_icrc_2025_pmt)
pmt_csv_name = "energy_pmt_icrc2025.csv"
df_pmt_2025.to_csv(pmt_csv_name, index=False)
print(f"{pmt_csv_name} generated")
print("")


print("Fetching ICRC2025 energy for SiPM events...")
events_id_sipm = np.unique(df_sipm["event_id"].to_numpy())
energy_icrc_2025_sipm = extract_energy_icrc2025(events_id_sipm, photodetector="sipm")

df_sipm_2025 = pd.DataFrame.from_dict(energy_icrc_2025_sipm)
sipm_csv_name = "energy_sipm_icrc2025.csv"
df_sipm_2025.to_csv(sipm_csv_name, index=False)
print(f"{sipm_csv_name} generated")
print("")


print(" merging all.. ")

df_merged = pd.concat([df_pmt_2025, df_sipm_2025])

outifle_csv_name = "energy_icrc2025.csv"
df_merged.to_csv(outifle_csv_name, index=False)
print(f"{outifle_csv_name} generated")