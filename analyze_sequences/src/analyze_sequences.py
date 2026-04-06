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

    #if counter != "0688":
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

        seq.sanity_checks()

        outdir_bp = Path(
            f"/storage/gpfs_data/pauger/users/jdejesus/analysis/analyze_sequences/outputs/{str(counter).zfill(4)}/{module}/bad_periods"
        )
        outdir_bp.mkdir(parents=True, exist_ok=True)

        seq.exclude_bad_periods(outdir_bp)

