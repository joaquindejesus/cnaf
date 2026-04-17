# ======== Standard library imports ========
import os
import pickle
import sys
from pathlib import Path

# ======== Third-party imports ========
import numpy as np
from scipy.stats import chi2
import pandas as pd 
import matplotlib.pyplot as plt 

# =========== My imports ================
sys.path.append(os.path.expanduser("/storage/gpfs_data/pauger/users/jdejesus/my_python_modules"))
from myClasses import Profile
from myMplStyles import setStyle
from myUsefulFunctions import generate_colormap

from sequence_class import Sequences



noise_early = (0, 100)
noise_late = (2048-100, 2048)
trigger_scope = (550, 1500)
trace_length = 2048

gdir = "/storage/gpfs_data/pauger/users/jdejesus/condor/md_binary_sequences_extractor/merged_sequences_csv/individual_modules/"

counter_list = os.listdir(gdir)

for counter in counter_list:
    counter_dir = os.path.join(gdir, counter)

    #if counter not in ["0688", "1760", "1767", "0643"]:
        #continue
    
    #if counter not in ["0093"]:
        #continue

    if not os.path.isdir(counter_dir):
        continue

    fname_list = [f for f in os.listdir(counter_dir) if f.endswith(".csv")] # e.g., ["sequences_c1760_101.csv", ..]


    for fname in fname_list:
        module_str = fname.split("_")[-1].split(".")[0]
        module = int(module_str)

        print(f" -- Counter {counter} - mod {module}")

        fpath = os.path.join(counter_dir, fname)
        df = pd.read_csv(fpath)

        seq = Sequences(
            counter=int(counter),
            module=module,
            noise_early=noise_early,
            noise_late=noise_late,
            trigger_scope=trigger_scope,
            trace_length=trace_length,
            df=df
        )

        # Sanity checks
        seq.sanity_checks()

        # Exclusion of bad periods
        print(" Excluding bad periods..")
        outdir_bp = Path(
            f"/storage/gpfs_data/pauger/users/jdejesus/analysis/analyze_sequences/outputs/{str(counter).zfill(4)}/{module}/bad_periods"
        )
        outdir_bp.mkdir(parents=True, exist_ok=True)

        seq.exclude_bad_periods(outdir_bp)

        # Various exploratory plots 
        print("")
        print(" Producing exploratory plots...")
        outdir_exp = Path(
            f"/storage/gpfs_data/pauger/users/jdejesus/analysis/analyze_sequences/outputs/{str(counter).zfill(4)}/{module}/exploratory_plots"
        )
        outdir_exp.mkdir(parents=True, exist_ok=True)

        kmax = 5

        seq.exploratory_plots(kmax=kmax, outdir=outdir_exp)

        # Background rate estimation -> we estimate rate_sequences = # sequences / (observation_time)
        # and rate_muon_pattern = # muon_patterns / (observation_time)
        # where observation_time is n_events_unique * dt, where dt is the length of the time window (late or early)
        
        print("")
        print(" Producing background rate estimation...")
        outdir_bkg = Path(
            f"/storage/gpfs_data/pauger/users/jdejesus/analysis/analyze_sequences/outputs/{str(counter).zfill(4)}/{module}/background"
        )
        outdir_bkg.mkdir(parents=True, exist_ok=True)

        seq.estimate_background_rate(outdir=outdir_bkg, isT4=0, isT5=0, kmax=64)

        # Backround P(w) distribution
        print("")
        print("Producing P(w) distributions...")
        seq.estimate_background_prob_distribution(outdir=outdir_bkg)

        # Getin
        print("")
        print("Analyzing single sequences...")
        
        outdir_est =  Path(
            f"/storage/gpfs_data/pauger/users/jdejesus/analysis/analyze_sequences/outputs/{str(counter).zfill(4)}/{module}/estimate_ineff"
        )
        outdir_est.mkdir(parents=True, exist_ok=True)

        seq.analyze_single_bar_events(lge_min=17.3, outdir=outdir_est)

        