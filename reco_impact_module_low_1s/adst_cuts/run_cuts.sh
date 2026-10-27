#!/bin/bash
#set -euo pipefail

# trunk
#eval $(/storage/gpfs_data/pauger/users/jdejesus/auger_software/offline/install/bin/this-auger-offline.sh)

# icrc2025-test7
eval $(/storage/gpfs_data/pauger/users/jdejesus/auger_software/offline/releases/icrc2025-test7/install/bin/this-auger-offline.sh)

OUT_DIR="."

PHASE="2"

selectADSTEvents \
    -c md_analysis_phase${PHASE}.cfg \
    -o ${OUT_DIR}/ADST_cuts_phase${PHASE}_01-2025_09-2026.root \
    -n off \
    /storage/gpfs_data/pauger/users/jdejesus/condor/md_data_reconstruction/outputs/phase${PHASE}/{2025,2026}/*/ADST_phase${PHASE}_xad_*_12h00.root \
    >> ${OUT_DIR}/adst_cuts_phase${PHASE}.log 2>&1
