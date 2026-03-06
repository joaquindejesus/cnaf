import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib.colors import LogNorm
from scipy.stats import ks_2samp, cramervonmises_2samp
from scipy.optimize import minimize_scalar
import ROOT

# Add custom module paths
sys.path.append('/storage/gpfs_data/pauger/users/jdejesus/my_python_modules/')
from myClasses import Profile
from myUsefulFunctions import generate_colormap

sys.path.append('/storage/gpfs_data/pauger/users/jdejesus/my_python_modules/auger')
from augerADSTgeneral import plot_module_traces


def save_figure(fig, fname):
    """Save a matplotlib figure and close it."""
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print(f"{fname} generated")

gdir = "../../outputs/mean_ldf/"

theta_min, theta_max = 0, 25 
module_size = "large"
core_in_uc = False 
susp_csv = (gdir + f"suspect_overcount_pmt_th{theta_min:.0f}-{theta_max:.0f}_"
               f"{module_size}_coreInUC{int(core_in_uc)}.csv")

df = pd.read_csv(susp_csv)

adst_path = ("/storage/gpfs_data/pauger/users/jdejesus/kit/datasets/pmt/reproduced_dataset_2025/adsts/"
        "ADST_cuts_stratGap_nOnes2_nGaps1_wSize7_rejSmall0_rejCorr1_effSmall1.04_effLarge0.95_"
        "rMin200_minCounters3_ldfSara1_fixBeta1.root")

n = 0 

for idx, row in df.iterrows():
    
     
    cid = row["counter_id"]
    mid = row["module_id"]
    evt_id = row["event_id"]

    fig = plot_module_traces(counter_id=cid, module_id=mid,
                        event_id=evt_id, fname=adst_path, sort_by_channel_id=False)

    figname = f"trace_{evt_id}_c{cid}m{mid}.png"
    save_figure(fig, figname)


    n +=1

    if n > 25:
        break 