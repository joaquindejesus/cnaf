"""
Script to read the merged ADST PMT data with the reco configuration as close as possible as we would do it today (PMT v2), extract event information, and dump results into a csv.

Also, it may read other configurations of reconstructinos that were produced in the KIT cluster during 2025 and are stored at
/storage/gpfs_data/pauger/users/jdejesus/kit/datasets/pmt/reproduced_dataset_2025/adsts


"""

import os
import sys
import logging

import pandas as pd

# Add custom module paths
sys.path.append('/storage/gpfs_data/pauger/users/jdejesus/my_python_modules/')

sys.path.append("/storage/gpfs_data/pauger/users/jdejesus/analysis/compare_pmt_sipm/src/")
from adst_reader_functions import readADST_data

sys.path.append('/storage/gpfs_data/pauger/users/jdejesus/my_python_modules/auger')
from augerADSTgeneral import get_module_level_info

# ============================================================
# Global configuration
# ============================================================

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


# Global directory containing the merged ADST with the PMT data (here I have the PMT v2 dataset)
gPath = "/storage/gpfs_data/pauger/users/jdejesus/analysis/compare_pmt_sipm/outputs/adst_cuts/"

# Here are the ADST generated in the kit cluster
gPath2 = "/storage/gpfs_data/pauger/users/jdejesus/kit/datasets/pmt/reproduced_dataset_2025/adsts"

## Parameters - see /storage/gpfs_data/pauger/users/jdejesus/analysis/compare_pmt_sipm/src/adst_cuts/run_cuts.sh
REJECTSMALL=1
REJECTCORRIENTES=1
STRATEGY="Consecutive"
NONES=3
NGAPS=0
WINDOWSIZE=1024

#EFFSMALL=1.04
EFFSMALL=0.94 # this is for 1x1 infinite window

#EFFLARGE=0.95
EFFLARGE=0.87 # this is for 1x1 infinite window

RMIN=200
MINCOUNTERS=3
## PARAMETERS FOR MdLDFFinder
# Wether to use the LDF parameterization of the PMT paper (LDFSARA=1) or the current LDF model (LDFSARA=0)
# Note that this will also change the CIC function
LDFSARA=0
FIXBETA=1

tag = (
    f"strat{STRATEGY}_"
    f"nOnes{NONES}_"
    f"nGaps{NGAPS}_"
    f"wSize{WINDOWSIZE}_"
    f"rejSmall{REJECTSMALL}_"
    f"rejCorr{REJECTCORRIENTES}_"
    f"effSmall{EFFSMALL}_"
    f"effLarge{EFFLARGE}_"
    f"rMin{RMIN}_"
    f"minCounters{MINCOUNTERS}_"
    f"ldfSara{LDFSARA}_"
    f"fixBeta{FIXBETA}"
)



# input filename
fname = os.path.join(gPath, f"ADST_cuts_{tag}.root")

# Check input file exists
if not os.path.exists(fname):
    fname = os.path.join(gPath2, f"ADST_cuts_{tag}.root")
if not os.path.exists(fname):
    raise FileNotFoundError(f"{fname} does not exist.")


## creating output directory where to dump the event-level csv file
out_dir = f"/storage/gpfs_data/pauger/users/jdejesus/analysis/compare_pmt_sipm/outputs/adst_to_csv"
os.makedirs(out_dir, exist_ok=True)
# output filename
outfile_name_events = os.path.join(out_dir, f"events_{tag}.csv")

#######################################
## Reading the event-level ADST
#######################################
logging.info(f"Analyzing {fname} for event-level info..")

events_info = readADST_data(fname, verbose=True)
n_events_info = len(events_info["event_id"])
logging.info(f"Events analyzed = {n_events_info}")

# dumping events_info dictionary into a csv
df = pd.DataFrame.from_dict(events_info)
df.to_csv(outfile_name_events, index=False)

print(f"{outfile_name_events} generated")
