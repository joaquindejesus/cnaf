
import os
import sys
import logging

sys.path.append("../")
from adst_reader_functions import  (find_1x1_patterns, validate_1x1_patterns,
                                   find_111_patterns, validate_111_patterns)

mock_trace = [0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0]

start_time_patterns_1x1 = find_1x1_patterns(trace=mock_trace, w=7)
start_time_patterns_111 = find_111_patterns(trace=mock_trace, w=7)

print(f"test. mock_trace = {mock_trace}")
print("start_times_1x1: ", start_time_patterns_1x1)
print("start_times_111: ", start_time_patterns_111)

# ============================================================
# Global configuration
# ============================================================

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Global directory containing ADST files
gPath = "/storage/gpfs_data/pauger/users/jdejesus/kit/datasets/pmt/reproduced_dataset_2025/adsts/"

## Parameters - see /cr/data01/joaquin/analysis/data_analysis_pmt_paper_repro/src/adst_cuts/run_cuts.sh
REJECTSMALL=0
REJECTCORRIENTES=1
STRATEGY="Gap"
NONES=2
NGAPS=1
WINDOWSIZE=1024
EFFSMALL=0.94
EFFLARGE=0.87
RMIN=200
MINCOUNTERS=3
## PARAMETERS FOR MdLDFFinder
# Wether to use the LDF parameterization of the PMT paper (LDFSARA=1) or the current LDF model (LDFSARA=0)
# Note that this will also change the CIC function
LDFSARA=1
FIXBETA=1

# input filename
fname = gPath + (f"ADST_cuts_strat{STRATEGY}_nOnes{NONES}_"
    f"nGaps{NGAPS}_"
    f"wSize{WINDOWSIZE}_"
    f"rejSmall{REJECTSMALL}_"
    f"rejCorr{REJECTCORRIENTES}_"
    f"effSmall{EFFSMALL}_"
    f"effLarge{EFFLARGE}_"
    f"rMin{RMIN}_"
    f"minCounters{MINCOUNTERS}_"
    f"ldfSara{LDFSARA}_"
    f"fixBeta{FIXBETA}.root")


# Check input file exists
if not os.path.exists(fname):
    raise FileNotFoundError(f"{fname} does not exist.")

## creating output directory
out_dir = "../../outputs/check_traces/"
os.makedirs(out_dir, exist_ok=True)

# output file name
outfile_name = out_dir + (f"check_traces_strat{STRATEGY}_nOnes{NONES}_"
    f"nGaps{NGAPS}_"
    f"wSize{WINDOWSIZE}_"
    f"rejSmall{REJECTSMALL}_"
    f"rejCorr{REJECTCORRIENTES}_"
    f"effSmall{EFFSMALL}_"
    f"effLarge{EFFLARGE}_"
    f"rMin{RMIN}_"
    f"minCounters{MINCOUNTERS}_"
    f"ldfSara{LDFSARA}_"
    f"fixBeta{FIXBETA}.log")

outfile = open(outfile_name, "w")

logging.info(f"Analyzing {fname}..")

if STRATEGY == "Gap" and NONES == 2 and NGAPS == 1:
    validate_1x1_patterns(fname, outfile=outfile, w=WINDOWSIZE, verbose=True)
elif STRATEGY == "Consecutive" and NONES == 3 and NGAPS == 0:
    validate_111_patterns(fname, outfile=outfile, w=WINDOWSIZE, verbose=True)
elif STRATEGY == "Gap" and NONES == 3 and NGAPS == 0:
    validate_111_patterns(fname, outfile=outfile, w=WINDOWSIZE, verbose=True)
else:
    raise ValueError("Parameters are not valid")

outfile.close()
print(f"{outfile_name} generated")
