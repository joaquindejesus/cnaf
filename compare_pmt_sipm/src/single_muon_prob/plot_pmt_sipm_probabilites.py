import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

# Add custom module paths
sys.path.append('/storage/gpfs_data/pauger/users/jdejesus/my_python_modules/')
from myClasses import Profile
from myUsefulFunctions import generate_colormap, compute_weighted_mean, standard_error_of_weighted_mean
from myMplStyles import setStyle 

setStyle(False)

method = "half" # half or whole
size = "large" # large (only 10m2), small (only 5m2), all

pmt_dir = "/storage/gpfs_data/pauger/users/jdejesus/kit/analysis/single_muon_prob/pmt/outputs/csv/"
pmt_file_name = f"single_muon_prob_pmt_consecutive_3_data_{method}_excludeUC0_tmin0_tmax1024_{size}.csv"
pmt_path = os.path.join(pmt_dir, pmt_file_name)
df_pmt = pd.read_csv(pmt_path)

sipm_dir = "/storage/gpfs_data/pauger/users/jdejesus/kit/analysis/single_muon_prob/sipm/outputs/csv/"
sipm_file_name = f"single_muon_prob_sipm_consecutive_4_data_{method}_excludeUC0_tmin0_tmax2048.csv"
sipm_path = os.path.join(sipm_dir, sipm_file_name)
df_sipm = pd.read_csv(sipm_path)

theta_center_sipm = df_sipm["theta_center"].unique()
theta1_sipm = df_sipm["theta1"].unique()
theta2_sipm = df_sipm["theta2"].unique()

theta_center_pmt = df_pmt["theta_center"].unique()
theta1_pmt = df_pmt["theta1"].unique()
theta2_pmt = df_pmt["theta2"].unique()



ms = 10
alpha = 0.9

# --------------------------------
# p1mu_neigh PMT vs SiPM 
# --------------------------------


fig, ax = plt.subplots(1, 2, figsize=(17, 9))

# SiPM 
for i, th_center in enumerate(theta_center_sipm[:3]):
    th1, th2 = theta1_sipm[i], theta2_sipm[i]
    label = rf"{th1:.1f} < $\theta \, / \, ^\circ$ < {th2:.1f}"

    df_th = df_sipm[df_sipm["theta_center"] == th_center]

    dphi_center = df_th["dphi_center"].unique()
    sin_dphi = np.sin(np.deg2rad(dphi_center))
    

    # p1_neigh 
    p1_neigh = (df_th["p1mu_100"] + 
                df_th["p1mu_110"] + 
                df_th["p1mu_111"] +
                df_th["p1mu_101"] )
    
    p1_neigh_err = np.sqrt(df_th["p1mu_100_boot_err"]**2 + 
                df_th["p1mu_110_boot_err"]**2 + 
                df_th["p1mu_111_boot_err"]**2 +
                df_th["p1mu_101_boot_err"]**2 )

    ax[0].errorbar(sin_dphi, p1_neigh, yerr=p1_neigh_err, fmt="o", ms=ms,
                    alpha=alpha, label=label)

# pmt
for i, th_center in enumerate(theta_center_pmt):
    th1, th2 = theta1_pmt[i], theta2_pmt[i]
    label = rf"{th1:.1f} < $\theta \, / \, ^\circ$ < {th2:.1f}"

    df_th = df_pmt[df_pmt["theta_center"] == th_center]

    dphi_center = df_th["dphi_center"].unique()
    sin_dphi = np.sin(np.deg2rad(dphi_center))
    

    # p1_neigh 
    p1_neigh = (df_th["p1mu_10"] + 
                df_th["p1mu_11"]
                )
    
    p1_neigh_err = np.sqrt(df_th["p1mu_10_boot_err"]**2 + 
                df_th["p1mu_11_boot_err"]**2 )

    ax[1].errorbar(sin_dphi, p1_neigh, yerr=p1_neigh_err, fmt="o", ms=ms,
                    alpha=alpha, label=label)


ax[0].set_xlabel(r"$\sin\Delta\phi$")
ax[0].set_ylabel(r"$p_{1\mu}^\text{neigh}$ SiPM")
ax[0].set_xlim([0, 1.05])
ax[0].set_ylim([0, 0.1])
ax[0].legend(fontsize="medium")

ax[1].set_xlabel(r"$\sin\Delta\phi$")
ax[1].set_ylabel(r"$p_{1\mu}^\text{neigh}$ PMT")
ax[1].set_xlim([0, 1.05])
ax[1].set_ylim([0, 0.1])
ax[1].legend(fontsize="medium")
fig.tight_layout(pad=1.1)


figname = "p1mu_neigh_pmt_vs_sipm.png"
fig.savefig(figname, bbox_inches="tight")
plt.close(fig)
print(f"{figname} generated")


# --------------------------------
# Ratio R = n1mu_neigh / n1bar PMT vs SiPM 
# --------------------------------
fig, ax = plt.subplots(1, 2, figsize=(17, 9))

# SiPM 
for i, th_center in enumerate(theta_center_sipm):
    th1, th2 = theta1_sipm[i], theta2_sipm[i]
    label = rf"{th1:.1f} < $\theta \, / \, ^\circ$ < {th2:.1f}"

    df_th = df_sipm[df_sipm["theta_center"] == th_center]

    dphi_center = df_th["dphi_center"].unique()
    sin_dphi = np.sin(np.deg2rad(dphi_center))
    

    # n1_neigh 
    n1_neigh = (df_th["n1mu_100"] + 
                df_th["n1mu_110"] + 
                df_th["n1mu_111"] +
                df_th["n1mu_101"] )
    
        
    n1_neigh_err = np.sqrt(df_th["n1mu_100_boot_err"]**2 + 
                df_th["n1mu_110_boot_err"]**2 + 
                df_th["n1mu_111_boot_err"]**2 +
                df_th["n1mu_101_boot_err"]**2 )

    n1bar = df_th["n1bar"].to_numpy()

    # ratio 
    r = n1_neigh / n1bar 

    rerr = n1_neigh_err / n1bar

    ax[0].errorbar(sin_dphi, r, yerr=rerr, fmt="o", ms=ms,
                    alpha=alpha, label=label)


# pmt
for i, th_center in enumerate(theta_center_pmt):
    th1, th2 = theta1_pmt[i], theta2_pmt[i]
    label = rf"{th1:.1f} < $\theta \, / \, ^\circ$ < {th2:.1f}"

    df_th = df_pmt[df_pmt["theta_center"] == th_center]

    dphi_center = df_th["dphi_center"].unique()
    sin_dphi = np.sin(np.deg2rad(dphi_center))
    

    # p1_neigh 
    n1_neigh = (df_th["n1mu_10"] + 
                df_th["n1mu_11"]
                )
    
    n1_neigh_err = np.sqrt(df_th["n1mu_10_boot_err"]**2 + 
                df_th["n1mu_11_boot_err"]**2 )

    n1bar = df_th["n1bar"].to_numpy()

    # ratio 
    r = n1_neigh / n1bar 

    rerr = n1_neigh_err / n1bar

    ax[1].errorbar(sin_dphi, r, yerr=rerr, fmt="o", ms=ms,
                    alpha=alpha, label=label)


ax[0].set_xlabel(r"$\sin\Delta\phi$")
ax[0].set_ylabel(r"$N_{1\mu}^\text{neigh} \, / \, N_\text{1bar}$ SiPM")
ax[0].set_xlim([0, 1.05])
ax[0].set_ylim([0, 0.15])
ax[0].legend(fontsize="medium")

ax[1].set_xlabel(r"$\sin\Delta\phi$")
ax[1].set_ylabel(r"$N_{1\mu}^\text{neigh} \, / \, N_\text{1bar}$ PMT")
ax[1].set_xlim([0, 1.05])
ax[1].set_ylim([0, 0.15])
ax[1].legend(fontsize="medium")
fig.tight_layout(pad=1.1)


figname = "ratio_n1bar_neigh_pmt_vs_sipm.png"
fig.savefig(figname, bbox_inches="tight")
plt.close(fig)
print(f"{figname} generated")


# --------------------------------
# p1mu_non_neigh PMT vs SiPM 
# --------------------------------


fig, ax = plt.subplots(1, 2, figsize=(17, 9))

# SiPM 
p1values, p1values_err = [], []
for i, th_center in enumerate(theta_center_sipm[:3]):
    th1, th2 = theta1_sipm[i], theta2_sipm[i]
    label = rf"{th1:.1f} < $\theta \, / \, ^\circ$ < {th2:.1f}"

    df_th = df_sipm[df_sipm["theta_center"] == th_center]

    dphi_center = df_th["dphi_center"].unique()
    sin_dphi = np.sin(np.deg2rad(dphi_center))
    

    # p1_non_neigh 
    p1_non_neigh = (df_th["p1mu_000"] + 
                df_th["p1mu_010"] + 
                df_th["p1mu_011"] +
                df_th["p1mu_001"] )
    
    p1_non_neigh_err = np.sqrt(df_th["p1mu_000_boot_err"]**2 + 
                df_th["p1mu_010_boot_err"]**2 + 
                df_th["p1mu_011_boot_err"]**2 +
                df_th["p1mu_001_boot_err"]**2 )
    
    p1values += list(p1_non_neigh)
    p1values_err += list(p1_non_neigh_err)

    ax[0].errorbar(sin_dphi, p1_non_neigh, yerr=p1_non_neigh_err, fmt="o", ms=ms,
                    alpha=alpha, label=label)

wmean = compute_weighted_mean(p1values, p1values_err)
wmean_err = standard_error_of_weighted_mean(p1values, p1values_err)
ax[0].axhline(y=wmean, ls="--", color="k", 
        label=f"{wmean:.4f} +/- {wmean_err:.4f}")

# pmt
p1values, p1values_err = [], []
for i, th_center in enumerate(theta_center_pmt):
    th1, th2 = theta1_pmt[i], theta2_pmt[i]
    label = rf"{th1:.1f} < $\theta \, / \, ^\circ$ < {th2:.1f}"

    df_th = df_pmt[df_pmt["theta_center"] == th_center]

    dphi_center = df_th["dphi_center"].unique()
    sin_dphi = np.sin(np.deg2rad(dphi_center))
    

    # p1_non_neigh 
    p1_non_neigh = (df_th["p1mu_00"] + 
                df_th["p1mu_01"]
                )
    
    p1_non_neigh_err = np.sqrt(df_th["p1mu_00_boot_err"]**2 + 
                df_th["p1mu_01_boot_err"]**2 )

    p1values += list(p1_non_neigh)
    p1values_err += list(p1_non_neigh_err)


    ax[1].errorbar(sin_dphi, p1_non_neigh, yerr=p1_non_neigh_err, fmt="o", ms=ms,
                    alpha=alpha, label=label)


wmean = compute_weighted_mean(p1values, p1values_err)
wmean_err = standard_error_of_weighted_mean(p1values, p1values_err)
ax[1].axhline(y=wmean, ls="--", color="k", 
        label=f"{wmean:.4f} +/- {wmean_err:.4f}")

ax[0].set_xlabel(r"$\sin\Delta\phi$")
ax[0].set_ylabel(r"$p_{1\mu}^\text{non-neigh}$ SiPM")
ax[0].set_xlim([0, 1.05])
ax[0].set_ylim([-0.01, .04])
ax[0].legend(fontsize="medium")

ax[1].set_xlabel(r"$\sin\Delta\phi$")
ax[1].set_ylabel(r"$p_{1\mu}^\text{non-neigh}$ PMT")
ax[1].set_xlim([0, 1.05])
ax[1].set_ylim([-0.01, .04])
ax[1].legend(fontsize="medium")
fig.tight_layout(pad=1.1)


figname = "p1mu_non_neigh_pmt_vs_sipm.png"
fig.savefig(figname, bbox_inches="tight")
plt.close(fig)
print(f"{figname} generated")


# --------------------------------
# Ratio R = n1mu_non_neigh / n1bar PMT vs SiPM 
# --------------------------------
fig, ax = plt.subplots(1, 2, figsize=(17, 9))

# SiPM 
for i, th_center in enumerate(theta_center_sipm):
    th1, th2 = theta1_sipm[i], theta2_sipm[i]
    label = rf"{th1:.1f} < $\theta \, / \, ^\circ$ < {th2:.1f}"

    df_th = df_sipm[df_sipm["theta_center"] == th_center]

    dphi_center = df_th["dphi_center"].unique()
    sin_dphi = np.sin(np.deg2rad(dphi_center))
    

    # n1_non_neigh 
    n1_non_neigh = (df_th["n1mu_000"] + 
                df_th["n1mu_010"] + 
                df_th["n1mu_011"] +
                df_th["n1mu_001"] )
    
        
    n1_non_neigh_err = np.sqrt(df_th["n1mu_000_boot_err"]**2 + 
                df_th["n1mu_010_boot_err"]**2 + 
                df_th["n1mu_011_boot_err"]**2 +
                df_th["n1mu_001_boot_err"]**2 )

    n1bar = df_th["n1bar"].to_numpy()

    # ratio 
    r = n1_non_neigh / n1bar 

    rerr = n1_non_neigh_err / n1bar

    ax[0].errorbar(sin_dphi, r, yerr=rerr, fmt="o", ms=ms,
                    alpha=alpha, label=label)


# pmt
for i, th_center in enumerate(theta_center_pmt):
    th1, th2 = theta1_pmt[i], theta2_pmt[i]
    label = rf"{th1:.1f} < $\theta \, / \, ^\circ$ < {th2:.1f}"

    df_th = df_pmt[df_pmt["theta_center"] == th_center]

    dphi_center = df_th["dphi_center"].unique()
    sin_dphi = np.sin(np.deg2rad(dphi_center))
    

    # n1_non_neigh 
    n1_non_neigh = (df_th["n1mu_00"] + 
                df_th["n1mu_01"]
                )
    
    n1_non_neigh_err = np.sqrt(df_th["n1mu_00_boot_err"]**2 + 
                df_th["n1mu_01_boot_err"]**2 )

    n1bar = df_th["n1bar"].to_numpy()

    # ratio 
    r = n1_non_neigh / n1bar 

    rerr = n1_non_neigh_err / n1bar
    

    ax[1].errorbar(sin_dphi, r, yerr=rerr, fmt="o", ms=ms,
                    alpha=alpha, label=label)


ax[0].set_xlabel(r"$\sin\Delta\phi$")
ax[0].set_ylabel(r"$N_{1\mu}^\text{non-neigh} \, / \, N_\text{1bar}$ SiPM")
ax[0].set_xlim([0, 1.05])
ax[0].set_ylim([-0.02, .05])
ax[0].legend(fontsize="medium")

ax[1].set_xlabel(r"$\sin\Delta\phi$")
ax[1].set_ylabel(r"$N_{1\mu}^\text{non-neigh} \, / \, N_\text{1bar}$ PMT")
ax[1].set_xlim([0, 1.05])
ax[1].set_ylim([-0.02, .05])
ax[1].legend(fontsize="medium")
fig.tight_layout(pad=1.1)


figname = "ratio_n1bar_non_neigh_pmt_vs_sipm.png"
fig.savefig(figname, bbox_inches="tight")
plt.close(fig)
print(f"{figname} generated")


# --------------------------------
# Ratio R = n1mu_neigh / n1mu_non_neigh PMT vs SiPM 
# --------------------------------

fig, ax = plt.subplots(1, 2, figsize=(17, 9))

# SiPM 
ratios, ratios_err = [], []
for i, th_center in enumerate(theta_center_sipm[:3]):
    th1, th2 = theta1_sipm[i], theta2_sipm[i]
    label = rf"{th1:.1f} < $\theta \, / \, ^\circ$ < {th2:.1f}"

    df_th = df_sipm[df_sipm["theta_center"] == th_center]

    dphi_center = df_th["dphi_center"].unique()
    sin_dphi = np.sin(np.deg2rad(dphi_center))
    

    # n1_non_neigh 
    n1_non_neigh = (df_th["n1mu_000"] + 
                df_th["n1mu_010"] + 
                df_th["n1mu_011"] +
                df_th["n1mu_001"] )
    
        
    n1_non_neigh_err = np.sqrt(df_th["n1mu_000_boot_err"]**2 + 
                df_th["n1mu_010_boot_err"]**2 + 
                df_th["n1mu_011_boot_err"]**2 +
                df_th["n1mu_001_boot_err"]**2 )

    # n1_neigh 
    n1_neigh = (df_th["n1mu_100"] + 
                df_th["n1mu_110"] + 
                df_th["n1mu_111"] +
                df_th["n1mu_101"] )
    
        
    n1_neigh_err = np.sqrt(df_th["n1mu_100_boot_err"]**2 + 
                df_th["n1mu_110_boot_err"]**2 + 
                df_th["n1mu_111_boot_err"]**2 +
                df_th["n1mu_101_boot_err"]**2 )

    # ratio 
    r = n1_neigh / n1_non_neigh 

    rerr = r * np.sqrt(  (n1_neigh_err / n1_neigh)**2 + (n1_non_neigh_err / n1_non_neigh)**2 ) 

    ratios += list(r)
    ratios_err += list(rerr)

    ax[0].errorbar(sin_dphi, r, yerr=rerr, fmt="o", ms=ms,
                    alpha=alpha, label=label)

wmean = compute_weighted_mean(ratios, ratios_err)
wmean_err = standard_error_of_weighted_mean(ratios, ratios_err)
ax[0].axhline(y=wmean, ls="--", color="k", 
        label=f"{wmean:.1f} +/- {wmean_err:.1f}")

# pmt
ratios, ratios_err = [], []
for i, th_center in enumerate(theta_center_pmt):
    th1, th2 = theta1_pmt[i], theta2_pmt[i]
    label = rf"{th1:.1f} < $\theta \, / \, ^\circ$ < {th2:.1f}"

    df_th = df_pmt[df_pmt["theta_center"] == th_center]

    dphi_center = df_th["dphi_center"].unique()
    sin_dphi = np.sin(np.deg2rad(dphi_center))
    

    # n1_non_neigh 
    n1_non_neigh = (df_th["n1mu_00"] + 
                df_th["n1mu_01"]
                )
    
    n1_non_neigh_err = np.sqrt(df_th["n1mu_00_boot_err"]**2 + 
                df_th["n1mu_01_boot_err"]**2 )

    # n1_neigh
    n1_neigh = (df_th["n1mu_10"] + 
                df_th["n1mu_11"]
                )
    
    n1_neigh_err = np.sqrt(df_th["n1mu_10_boot_err"]**2 + 
                df_th["n1mu_11_boot_err"]**2 )

    # ratio 
    r = n1_neigh / n1_non_neigh 

    rerr = r * np.sqrt(  (n1_neigh_err / n1_neigh)**2 + (n1_non_neigh_err / n1_non_neigh)**2 ) 

    ratios += list(r)
    ratios_err += list(rerr)

    ax[1].errorbar(sin_dphi, r, yerr=rerr, fmt="o", ms=ms,
                    alpha=alpha, label=label)

wmean = compute_weighted_mean(ratios, ratios_err)
wmean_err = standard_error_of_weighted_mean(ratios, ratios_err)
ax[1].axhline(y=wmean, ls="--", color="k", 
        label=f"{wmean:.1f} +/- {wmean_err:.1f}")

ax[0].set_xlabel(r"$\sin\Delta\phi$")
ax[0].set_ylabel(r"$N_{1\mu}^\text{neigh} \, / \, N_{1\mu}^\text{non-neigh}$ SiPM")
ax[0].set_xlim([0, 1.05])
#ax[0].set_ylim([-0.02, 1])
ax[0].legend(fontsize="medium")

ax[1].set_xlabel(r"$\sin\Delta\phi$")
ax[1].set_ylabel(r"$N_{1\mu}^\text{neigh} \, / \, N_{1\mu}^\text{non-neigh}$ PMT")
ax[1].set_xlim([0, 1.05])
#ax[1].set_ylim([-0.02, 1])
ax[1].legend(fontsize="medium")
fig.tight_layout(pad=1.1)


figname = "ratio_neigh_non_neigh_pmt_vs_sipm.png"
fig.savefig(figname, bbox_inches="tight")
plt.close(fig)
print(f"{figname} generated")



# --------------------------------
# Ratios of probabilites p1mu_neigh / p1mu_non_neigh PMT vs SiPM 
# --------------------------------
fig, ax = plt.subplots(1, 2, figsize=(17, 9))

# SiPM 
ratios, ratios_err = [], []
for i, th_center in enumerate(theta_center_sipm[:3]):
    th1, th2 = theta1_sipm[i], theta2_sipm[i]
    label = rf"{th1:.1f} < $\theta \, / \, ^\circ$ < {th2:.1f}"

    df_th = df_sipm[df_sipm["theta_center"] == th_center]

    dphi_center = df_th["dphi_center"].unique()
    sin_dphi = np.sin(np.deg2rad(dphi_center))
    

    # p1_non_neigh 
    p1_non_neigh = (df_th["p1mu_000"] + 
                df_th["p1mu_010"] + 
                df_th["p1mu_011"] +
                df_th["p1mu_001"] )
    
        
    p1_non_neigh_err = np.sqrt(df_th["p1mu_000_boot_err"]**2 + 
                df_th["p1mu_010_boot_err"]**2 + 
                df_th["p1mu_011_boot_err"]**2 +
                df_th["p1mu_001_boot_err"]**2 )

    # p1_neigh 
    p1_neigh = (df_th["p1mu_100"] + 
                df_th["p1mu_110"] + 
                df_th["p1mu_111"] +
                df_th["p1mu_101"] )
    
        
    p1_neigh_err = np.sqrt(df_th["p1mu_100_boot_err"]**2 + 
                df_th["p1mu_110_boot_err"]**2 + 
                df_th["p1mu_111_boot_err"]**2 +
                df_th["p1mu_101_boot_err"]**2 )

    # ratio 
    r = p1_neigh / p1_non_neigh 

    rerr = r * np.sqrt(  (p1_neigh_err / p1_neigh)**2 + (p1_non_neigh_err / p1_non_neigh)**2 ) 

    ratios += list(r)
    ratios_err += list(rerr)

    ax[0].errorbar(sin_dphi, r, yerr=rerr, fmt="o", ms=ms,
                    alpha=alpha, label=label)

wmean = compute_weighted_mean(ratios, ratios_err)
wmean_err = standard_error_of_weighted_mean(ratios, ratios_err)
ax[0].axhline(y=wmean, ls="--", color="k", 
        label=f"{wmean:.1f} +/- {wmean_err:.1f}")

# pmt
ratios, ratios_err = [], []
for i, th_center in enumerate(theta_center_pmt):
    th1, th2 = theta1_pmt[i], theta2_pmt[i]
    label = rf"{th1:.1f} < $\theta \, / \, ^\circ$ < {th2:.1f}"

    df_th = df_pmt[df_pmt["theta_center"] == th_center]

    dphi_center = df_th["dphi_center"].unique()
    sin_dphi = np.sin(np.deg2rad(dphi_center))
    

    # p1_non_neigh 
    p1_non_neigh = (df_th["p1mu_00"] + 
                df_th["p1mu_01"]
                )
    
    p1_non_neigh_err = np.sqrt(df_th["p1mu_00_boot_err"]**2 + 
                df_th["p1mu_01_boot_err"]**2 )

    # p1_neigh
    p1_neigh = (df_th["p1mu_10"] + 
                df_th["p1mu_11"]
                )
    
    p1_neigh_err = np.sqrt(df_th["p1mu_10_boot_err"]**2 + 
                df_th["p1mu_11_boot_err"]**2 )

    # ratio 
    r = p1_neigh / p1_non_neigh 

    rerr = r * np.sqrt(  (p1_neigh_err / p1_neigh)**2 + (p1_non_neigh_err / p1_non_neigh)**2 ) 

    ratios += list(r)
    ratios_err += list(rerr)

    ax[1].errorbar(sin_dphi, r, yerr=rerr, fmt="o", ms=ms,
                    alpha=alpha, label=label)

wmean = compute_weighted_mean(ratios, ratios_err)
wmean_err = standard_error_of_weighted_mean(ratios, ratios_err)
ax[1].axhline(y=wmean, ls="--", color="k", 
        label=f"{wmean:.1f} +/- {wmean_err:.1f}")

ax[0].set_xlabel(r"$\sin\Delta\phi$")
ax[0].set_ylabel(r"$p_{1\mu}^\text{neigh} \, / \, p_{1\mu}^\text{non-neigh}$ SiPM")
ax[0].set_xlim([0, 1.05])
#ax[0].set_ylim([-0.02, 1])
ax[0].legend(fontsize="medium")

ax[1].set_xlabel(r"$\sin\Delta\phi$")
ax[1].set_ylabel(r"$p_{1\mu}^\text{neigh} \, / \, p_{1\mu}^\text{non-neigh}$ PMT")
ax[1].set_xlim([0, 1.05])
#ax[1].set_ylim([-0.02, 1])
ax[1].legend(fontsize="medium")
fig.tight_layout(pad=1.1)


figname = "ratio_p1mu_neigh_non_neigh_pmt_vs_sipm.png"
fig.savefig(figname, bbox_inches="tight")
plt.close(fig)
print(f"{figname} generated")