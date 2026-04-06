"""
TODO fill
"""
import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import poisson

# Add custom module paths
sys.path.append('/storage/gpfs_data/pauger/users/jdejesus/my_python_modules/')
from myClasses import Profile
from myMplStyles import setStyle
from myUsefulFunctions import fit_linear

setStyle(False)

strategy = "consecutive" # consectuive 'gap' or 'strategy'
method = "half"

gpath = "/storage/gpfs_data/pauger/users/jdejesus/kit/analysis/single_muon_prob/pmt/outputs/csv/"
fname = f"adst_dt_pairs_pmt_{strategy}_3_data_{method}.csv"

csv_path = os.path.join(gpath, fname)

if not os.path.isfile(csv_path):
    raise FileNotFoundError(f"{csv_path} does not exists..")

df = pd.read_csv(csv_path)

# cuts
# module size
cut_module_size = df["small_module"] == False

# k
cut_k = df["k"] <= 1

# module
cut_module = False


# energy?

if cut_module:
    cut_mod = (df["counter"] == 1764) & (df["module"] == 3)
    cuts = cut_module_size & cut_k & cut_mod
else:
    cuts = cut_module_size & cut_k

print(f"len(df) before cuts = {len(df)}")
df = df[cuts]
print(f"len(df) after cuts = {len(df)}")

# profile w vs sec_theta
w = df["w1"].to_numpy()
sec = 1 / np.cos(np.deg2rad(df["theta"].to_numpy()))

sec_bin_edges = np.linspace(1, 1/np.cos(np.deg2rad(45)), 7)

p = Profile(sec, w).doProfile(sec_bin_edges)

# linear fit

fit = fit_linear(x=p.x_mean, y=p.y_mean,
                 yerr=p.y_std_of_mean, x_ref=1, p0=(1, 1))
sec_plot = np.linspace(sec_bin_edges.min(), sec_bin_edges.max(), 100)
fit_plot = fit["popt"][0]*(sec_plot-1) + fit["popt"][1]
label = (r"$y = m \left( \sec\theta -1 \right) + b$" + "\n" +
         f"m = {fit['popt'][0]:.3f} +/- {fit['popt_err'][0]:.3f}\n" +
         f"b = {fit['popt'][1]:.3f} +/- {fit['popt_err'][1]:.3f}")

fig, ax = plt.subplots(1, 1, figsize=(14, 8))
ax.errorbar(p.x_mean, p.y_mean, yerr=p.y_std_of_mean, fmt="o")
ax.plot(sec_plot, fit_plot, "--", label=label)

ax.set_xlabel(r"$\sec\theta$")
ax.set_ylabel(r"$\langle \text{width} \, / \, (3.125 \text{ns}) \rangle$")

ax.legend()

figname = f"width_vs_sec_theta.png"

fig.savefig(figname, bbox_inches="tight")

plt.close(fig)
print(f"{figname} generated")

# print profile secTheta vs w
#print("Profile of sec_theta vs w")
#print(f" x = {p.x_mean}")
#print(f" y = {p.y_mean}")

# ------------------------------------------------------
# Method 1 -Extrapolating efficiency lab measurement with theta = 0 to
# larger zenith using the slope of the fit to w vs sec_theta

eps0 = 0.87
eff_extrapolated = fit["popt"][0]*(sec_plot-1) + eps0

fig, ax = plt.subplots(1, 1, figsize=(14, 8))

ax.plot(sec_plot, eff_extrapolated, "-.",
        label="extrapolated with linear fit")
ax.axhline(y=1, ls="--", color="k")





# ------------------------------------------------------
# Method 2 -Extrapolating efficiency lab measurement with theta = 0 to
# larger zenith using the probability of w=3 (suggsted by chatGPT)


w_bin_edges =  np.arange(0, 16, step=1) - 0.5
w_bin_centers = w_bin_edges[:-1] + np.diff(w_bin_edges)/2

eff_est = []
for dist in p.y_full_dist:
    hist, _ = np.histogram(dist, bins=w_bin_edges, density=True)
    ee = 1 / (1+hist[w_bin_centers==3])
    eff_est.append(ee)

# normalize to the lab measurement at theta = 0
eff_est_norm = np.array(eff_est) * eps0 / eff_est[0]

ax.errorbar(p.x_mean, eff_est, fmt="s", mfc="None", label="eff est")
ax.errorbar(p.x_mean, eff_est_norm, fmt="o", label="eff est normalized to lab")

ax.set_xlabel(r"$\sec\theta$")
ax.set_ylabel(r"Efficiency")

ax.legend()

figname = f"eff_extrapol_vs_sec_theta.png"

fig.savefig(figname, bbox_inches="tight")

plt.close(fig)
print(f"{figname} generated")

# -----------------------
# Distributions
fig, ax = plt.subplots(1, 1, figsize=(14, 8))

# most vertical bin


n0, bins0, _ = ax.hist(p.y_full_dist[0], histtype="step", bins=w_bin_edges,
        label=f"most vertical bin ({len(p.y_full_dist[0])} entries)", density=True,
        color="b")

# poisson
#ax.plot(w_bin_centers, poisson(mu=p.y_mean[0]).pmf(w_bin_centers), "-o",
        #color="b")

n_incl, bins_incl, _ = ax.hist(p.y_full_dist[-1], histtype="stepfilled", alpha=0.3,
        bins=w_bin_edges,
        label=f"most inclined bin ({len(p.y_full_dist[-1])} entries)", density=True,
        color="r")
#ax.plot(w_bin_centers, poisson(mu=p.y_mean[-1]).pmf(w_bin_centers), "--o",
        #color="r")

#print("Printing normalized probabilites for vertical-most bin...")
#for i, pp in enumerate(n0):
    #print(f"P(w={i}) = {pp:.3f}")

ax.set_xlabel("width / (3.125 ns)")
ax.set_ylabel("Probability")
ax.legend()

figname = f"width_distributions.png"

fig.savefig(figname, bbox_inches="tight")

plt.close(fig)
print(f"{figname} generated")
