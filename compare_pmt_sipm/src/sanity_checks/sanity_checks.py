import os
import sys

import pandas as pd
import matplotlib.pyplot as plt

from functions import *
sys.path.append('/storage/gpfs_data/pauger/users/jdejesus/my_python_modules/')
from myMplStyles import setStyle

setStyle(False)

## Parameters - see /cr/data01/joaquin/analysis/data_analysis_pmt_paper_repro/src/adst_cuts/run_cuts.sh
REJECTSMALL=1
REJECTCORRIENTES=1
STRATEGY="Consecutive"
NONES=3
NGAPS=0
WINDOWSIZE=1024
#EFFSMALL=1.04
EFFSMALL = 0.94 # this is for infinte window 1x1
#EFFLARGE=0.95
EFFLARGE = 0.87 # this is for infinite window 1x1
RMIN=200
MINCOUNTERS=3
## PARAMETERS FOR MdLDFFinder
# Wether to use the LDF parameterization of the PMT paper (LDFSARA=1) or the current LDF model (LDFSARA=0)
# Note that this will also change the CIC function
LDFSARA=0
FIXBETA=1

parameters_str = (f"strat{STRATEGY}_nOnes{NONES}_nGaps{NGAPS}_wSize{WINDOWSIZE}_rejSmall{REJECTSMALL}_"
                f"rejCorr{REJECTCORRIENTES}_effSmall{EFFSMALL}_"
                f"effLarge{EFFLARGE}_rMin{RMIN}_minCounters{MINCOUNTERS}_"
                f"ldfSara{LDFSARA}_fixBeta{FIXBETA}")

# Input file name
basename = f"events_{parameters_str}.csv"
input_dir = "/storage/gpfs_data/pauger/users/jdejesus/kit/datasets/pmt/reproduced_dataset_2025/csv/"
fname = os.path.join(input_dir, basename)

if not os.path.isfile(fname):
    input_dir = "/storage/gpfs_data/pauger/users/jdejesus/analysis/compare_pmt_sipm/outputs/adst_to_csv/"
    fname = os.path.join(input_dir, basename)

if not os.path.isfile(fname):
    raise FileNotFoundError(f"{fname} was not found!")

print(f"input file = {fname}")

# defining cuts
lge_min = 17.3
lge_max = 18.3
theta_max = 45
hottest_sd_in_uc = True

# additional cuts to be put in the titles of the figures
cuts_title = (f"CUTS: {lge_min:.2f} < lge < {lge_max:.2f}. "
               f"theta < {theta_max}. hottest_sd_uc = {hottest_sd_in_uc}")
# additional cuts to be included in the figures name
cuts_str = (f"lgemin{lge_min:.2f}_lgemax{lge_max:.2f}_thetaMax{theta_max:.0f}_"
            f"hottestinUC{int(hottest_sd_in_uc)}")


df = pd.read_csv(fname)

print(f"number of events before cuts: {len(df)}")

df = make_cuts(df=df, lge_min=lge_min, lge_max=lge_max, theta_max=theta_max,
               hottest_sd_in_uc=hottest_sd_in_uc)

print(f"number of events after cuts: {len(df)}")

## creating output directory
out_dir = "../../outputs/sanity_checks"
os.makedirs(out_dir, exist_ok=True)

# ===============================
# Plot cores distribution

figures_core = plot_core_distributions(df)

# scatter plot
fig, ax = figures_core["scatter"]["fig"], figures_core["scatter"]["ax"]

ax.set_title(cuts_title)

figname = f"core_scatter_{parameters_str}_{cuts_str}.png"
fig.savefig(out_dir + "/" + figname, bbox_inches="tight")
plt.close(fig)
print(f"{figname} saved")


# ===============================
# Plot time series
figures_time_series = plot_time_series(df)

# energy time series
fige, axe = figures_time_series["energy"]["fig"], figures_time_series["energy"]["ax"]

axe.set_title(cuts_title)

figname_e = f"time_serie_energy_{parameters_str}_{cuts_str}.png"
fige.savefig(out_dir + "/" + figname_e, bbox_inches="tight")
plt.close(fige)
print(f"{figname_e} saved")

# rho35 time series
figr, axr = figures_time_series["rho35"]["fig"], figures_time_series["rho35"]["ax"]

axr.set_title(cuts_title)

figname_r = f"time_serie_rho35_{parameters_str}_{cuts_str}.png"
figr.savefig(out_dir + "/" + figname_r, bbox_inches="tight")
plt.close(figr)
print(f"{figname_r} saved")

# n events time series
fign, axn = figures_time_series["nr_events"]["fig"], figures_time_series["nr_events"]["ax"]

axn.set_title(cuts_title)

figname_n = f"time_serie_nevents_{parameters_str}_{cuts_str}.png"
fign.savefig(out_dir + "/" + figname_n, bbox_inches="tight")
plt.close(fign)
print(f"{figname_n} saved")

# ===============================
# Plot energy distribution
figures_energy = plot_energy_distribution(df, lge_min, lge_max)
fig, ax = figures_energy["hist"]["fig"], figures_energy["hist"]["ax"]
ax.set_title(cuts_title)

figname = f"energy_dist_{parameters_str}_{cuts_str}.png"
fig.savefig(out_dir + "/" + figname, bbox_inches="tight")
plt.close(fig)
print(f"{figname} saved")

fig, ax = figures_energy["hist_norm"]["fig"], figures_energy["hist_norm"]["ax"]
ax.set_title(cuts_title)

figname = f"energy_dist_norm_{parameters_str}_{cuts_str}.png"
fig.savefig(out_dir + "/" + figname, bbox_inches="tight")
plt.close(fig)
print(f"{figname} saved")

# ===============================
# Plot zenith angle distribution 
figures_zenith = plot_zenith_angle_distribution(df)

# theta distribution
fig, ax = figures_zenith["theta"]["fig"], figures_zenith["theta"]["ax"]

ax.set_title(cuts_title)

figname = f"theta_dist_{parameters_str}_{cuts_str}.png"
fig.savefig(out_dir + "/" + figname, bbox_inches="tight")
plt.close(fig)
print(f"{figname} saved")

# sin2 distribution
fig, ax = figures_zenith["sin2theta"]["fig"], figures_zenith["sin2theta"]["ax"]

ax.set_title(cuts_title)

figname = f"sin2theta_dist_{parameters_str}_{cuts_str}.png"
fig.savefig(out_dir + "/" + figname, bbox_inches="tight")
plt.close(fig)
print(f"{figname} saved")

# ===============================
## test fatt is correctly loaded

figures_att = test_attenuation_function(df)

fig, ax = figures_att["fig"], figures_att["ax"]

fig.suptitle(cuts_title)

figname = f"fatt_sanity_check_{parameters_str}_{cuts_str}.png"
fig.savefig(out_dir + "/" + figname, bbox_inches="tight")
plt.close(fig)
print(f"{figname} saved")

# =============================================
## test beta parameterization is corectly loaded

figures_beta = test_beta_parameterization(df)

fig, ax = figures_beta["fig"], figures_beta["ax"]

fig.suptitle(cuts_title)

figname = f"beta_sanity_check_{parameters_str}_{cuts_str}.png"
fig.savefig(out_dir + "/" + figname, bbox_inches="tight")
plt.close(fig)
print(f"{figname} saved")
