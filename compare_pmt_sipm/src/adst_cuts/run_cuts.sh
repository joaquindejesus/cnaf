#!/bin/bash

eval $(/storage/gpfs_data/pauger/users/jdejesus/auger_software/offline/worktrees/UMD_PMT_rec_install/bin/this-auger-offline.sh)

# reconstruction config


## PARAMETERS FOR MdModuleRejector
# Wether to reject or not small modules
REJECTSMALL=1
# Wether to reject or not counter 93 (in Sarahs thesis it is stated that this counter was not
# in acquisition for the PMT data)
REJECTCORRIENTES=1

## PARAMETERS FOR MdMuonCounter
STRATEGY="Consecutive"
NONES=3
NGAPS=0
WINDOWSIZE=1024

## PARAMETERS FOR MdBiasCorrector
# Efficiency correction for small modules (5m2)
#EFFSMALL=1.04 # this is for 1x1, window of 7 bins
EFFSMALL=0.94 # this is for 1x1 infinite window

# Efficiency correction for large modules (10m2)
#EFFLARGE=0.95 # this is for 1x1, window of 7 bins
EFFLARGE=0.87 # this is for 1x1 infinite window

## PARAMETERS FOR MdEeventSelector
# Counters with r < RMIN will be rejected
RMIN=200
# Minimum number of counters to fit an LDF
MINCOUNTERS=3

## PARAMETERS FOR MdLDFFinder
# Wether to use the LDF parameterization of the PMT paper (LDFSARA=1) or the current LDF model (LDFSARA=0)
# Note that this will also change the CIC function
LDFSARA=0
# Wether to fix beta in every event or allow it to vary if lever arm conditions are met
FIXBETA=1

# phase 1 is UB only,
# phase 2 is UUB only
phase="1"

out_dir="/storage/gpfs_data/pauger/users/jdejesus/analysis/compare_pmt_sipm/outputs/adst_cuts"

mkdir -p "$out_dir"


selectADSTEvents -c md_analysis_phase${phase}.cfg -o ${out_dir}/ADST_cuts_strat${STRATEGY}_nOnes${NONES}_\
nGaps${NGAPS}_\
wSize${WINDOWSIZE}_\
rejSmall${REJECTSMALL}_\
rejCorr${REJECTCORRIENTES}_\
effSmall${EFFSMALL}_\
effLarge${EFFLARGE}_\
rMin${RMIN}_\
minCounters${MINCOUNTERS}_\
ldfSara${LDFSARA}_\
fixBeta${FIXBETA}.root -n off \
\
/storage/gpfs_data/pauger/users/jdejesus/condor/md_old_pmt_data_reco_paper_repro/outputs/20*/*/ADST_xad_*_strat${STRATEGY}_nOnes${NONES}_\
nGaps${NGAPS}_\
wSize${WINDOWSIZE}_\
rejSmall${REJECTSMALL}_\
rejCorr${REJECTCORRIENTES}_\
effSmall${EFFSMALL}_\
effLarge${EFFLARGE}_\
rMin${RMIN}_\
minCounters${MINCOUNTERS}_\
ldfSara${LDFSARA}_\
fixBeta${FIXBETA}.root 2>&1 | tee \
\
${out_dir}/adst_cuts_strat${STRATEGY}_nOnes${NONES}_\
nGaps${NGAPS}_\
wSize${WINDOWSIZE}_\
rejSmall${REJECTSMALL}_\
rejCorr${REJECTCORRIENTES}_\
effSmall${EFFSMALL}_\
effLarge${EFFLARGE}_\
rMin${RMIN}_\
minCounters${MINCOUNTERS}_\
ldfSara${LDFSARA}_\
fixBeta${FIXBETA}.log




