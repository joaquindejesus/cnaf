"""
The idea of this script is to use the module-level muon measurements of the PMT and SiPM dataset 
to compare the mean ldf of the two datasets in the same energy and zenith angle bin.
Ideally, to avoid uncertainties about the different corner-clipping corrections in the
two datasets, this comparison should be made in a vertical bin (e.g., sec_theta < 1.1).

Before doing the LDF comparison, a check is made on the underlying energy and zenith distributions
of the PMT and SiPM datasets, to be sure that they share the same energy scale and sec_theta 
(comparing datasets with different energy scales would distort the comparison of the muon measurements)

The module-level measurements were obtained with the following scripts:
I) adst_to_pickle.py in /cr/data01/joaquin/analysis/data_analysis_pmt_paper_repro/src/adst_to_pickle/ 
for PMT in the KIT cluster. In CNAF, the .csv files can be found in 

II) extract_module_level_info.py in /cr/data01/joaquin/analysis/data_analysis/src/module_level_info/
for SiPM

Both for PMT and SiPM, the extraction of the module-level info relies on the function 
get_module_level_info() from the module augerADSTgeneral.py in my_python_modules 

This file (and functions.py) were built upon the KIT file 
/storage/gpfs_data/pauger/users/jdejesus/kit/analysis/pmt_paper_reanalysis/src/mean_ldf/compare_ldf_sipm_pmt.py

"""

import os
import sys
from pprint import pprint

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
from myUsefulFunctions import generate_colormap, compute_weighted_mean, standard_error_of_weighted_mean

sys.path.append('/storage/gpfs_data/pauger/users/jdejesus/my_python_modules/auger')
from augerADSTgeneral import extract_sd_info
from augerDetector import get_sd_tank_coordinates, extract_channels_broken

from functions import *

# since I used an older version in which many broken channels (mainly for SiPM) were not loaded
# in the xml, I have to externally correct the segmentation of each module reading the XML here
mmodel_confing_xml = "/storage/gpfs_data/pauger/users/jdejesus/auger_software/offline/src/Framework/MDetector/MModelConfigIdeal.xml.in"
broken_channels_pmt = extract_channels_broken(xml_path=mmodel_confing_xml, photodetector="pmt", 
                    as_dataframe=True)
broken_channels_sipm = extract_channels_broken(xml_path=mmodel_confing_xml, photodetector="sipm", 
                    as_dataframe=True)

import numpy as np
from scipy.optimize import minimize
from scipy.stats import binom

class EfficiencyRatioEstimator:
    """
    Estima el ratio de eficiencias R entre dos detectores y el flujo medio μ
    usando el método de máxima verosimilitud considerando todos los k_i.

    Parámetros
    ----------
    K1 : array-like
        Mediciones del detector 1 (p.ej. PMT)
    K2 : array-like
        Mediciones del detector 2 (p.ej. SiPM)
    n_segments1 : int
        Número de segmentos del detector 1
    n_segments2 : int
        Número de segmentos del detector 2
    """

    def __init__(self, K1, K2, n_segments1, n_segments2):
        self.K1 = np.array(K1)
        self.K2 = np.array(K2)
        self.n1 = n_segments1
        self.n2 = n_segments2
        self.result = None

    def _neg_log_likelihood(self, params):
        """
        Función de log-verosimilitud negativa para optimización.
        params = [mu, R]
        """
        mu, R = params
        if mu <= 0 or R <= 0 or R > 10:
            return 1e10  # penalización para valores no físicos
        
        p1 = 1 - np.exp(-mu / self.n1)
        p2 = 1 - np.exp(-R * mu / self.n2)

        ll1 = binom.logpmf(self.K1, self.n1, p1).sum()
        ll2 = binom.logpmf(self.K2, self.n2, p2).sum()

        return - (ll1 + ll2)

    def fit(self, mu_guess=1.0, R_guess=1.0):
        """
        Optimiza la verosimilitud y devuelve estimaciones de mu y R.
        """
        res = minimize(self._neg_log_likelihood, x0=[mu_guess, R_guess], 
                       bounds=[(1e-6, None), (1e-6, 10)], method="L-BFGS-B")
        if not res.success:
            #raise RuntimeError("Optimization failed: " + res.message)
            print("[WARNING] Optimization failed: " + res.message)
            return np.nan, np.nan 
        self.result = res
        self.mu_hat, self.R_hat = res.x
        return self.mu_hat, self.R_hat

    def get_errors(self):
        """
        Aproxima errores de mu y R usando la matriz Hessiana de la log-verosimilitud.
        """
        if self.result is None:
            #raise RuntimeError("Fit first with .fit()")
            print("[WARNING] get_errors() will return NaNs because there is no fit available")
            return np.array([np.nan, np.nan])

        eps = 1e-5
        params = self.result.x
        hessian = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                p1 = params.copy()
                p2 = params.copy()
                p3 = params.copy()
                p4 = params.copy()

                p1[i] += eps; p1[j] += eps
                p2[i] += eps; p2[j] -= eps
                p3[i] -= eps; p3[j] += eps
                p4[i] -= eps; p4[j] -= eps

                f1 = self._neg_log_likelihood(p1)
                f2 = self._neg_log_likelihood(p2)
                f3 = self._neg_log_likelihood(p3)
                f4 = self._neg_log_likelihood(p4)

                hessian[i, j] = (f1 - f2 - f3 + f4) / (4 * eps**2)

        cov = np.linalg.inv(hessian)
        return np.sqrt(np.diag(cov))  # errores: [sigma_mu, sigma_R]


    

def plot_time_series(df_pmt, df_sipm):
    """
    Generate time-evolution and comparison plots for a single UMD module
    operated during both the PMT and SiPM eras.

    The function computes time-binned profiles of several module-level
    observables using GPS time as the independent variable, and produces
    a collection of diagnostic plots comparing PMT and SiPM data. In addition,
    it evaluates relative scaling factors between the PMT and SiPM eras using
    (i) the mean local muon density LDF and (ii) the correlation between WCD
    signal and muon density.

    All selections (energy, zenith angle, core distance, quality cuts, etc.)
    must be applied prior to calling this function. This routine assumes that
    the input DataFrames correspond to the same physical UMD module.

    Parameters
    ----------
    df_pmt : pandas.DataFrame
        Module-level measurements from the PMT era. The DataFrame must contain
        at least the following columns:

        - ``gps`` : GPS time in seconds
        - ``sd_signal`` : WCD signal in VEM
        - ``k_1x1`` : number of activated bars (infinite window definition)
        - ``rho_shower_plane`` : muon density projected onto the shower plane
        - ``r`` : distance to the shower axis (m)

    df_sipm : pandas.DataFrame
        Same as ``df_pmt``, but corresponding to the SiPM era. Expected columns
        include:

        - ``gps``
        - ``sd_signal``
        - ``k`` : number of activated bars
        - ``rho_shower_plane``
        - ``r``

    Returns
    -------
    dict
        Dictionary containing Matplotlib figures and axes for all generated
        plots, together with the estimated scaling factors and uncertainties.
        The returned keys are:

        - ``fig``, ``ax``     : WCD signal vs GPS time
        - ``fig2``, ``ax2``   : mean k vs GPS time
        - ``fig3``, ``ax3``   : mean muon density vs GPS time
        - ``fig4``, ``ax4``   : WCD signal distributions
        - ``fig5``, ``ax5``   : muon density distributions
        - ``fig6``, ``ax6``   : mean LDF comparison and scaling
        - ``fig7``, ``ax7``   : WCD signal vs muon density correlation
        - ``fig8``, ``ax8``   : k distributions
        - ``scaling_factor_ldf`` :
            Scaling factor matching the SiPM and PMT mean LDFs
        - ``scaling_factor_ldf_err`` :
            Uncertainty on the LDF scaling factor
        - ``scaling_factor_sd_rho`` :
            Scaling factor from the WCD signal vs muon density correlation
        - ``scaling_factor_sd_rho_err`` :
            Uncertainty on the corresponding scaling factor

    Notes
    -----
    - Time profiles are computed in fixed bins of four weeks using a custom
      ``Profile`` object.
    - Only time bins with non-zero mean k are shown in time-evolution plots.
    - Scaling factors are obtained via ``fit_scaling_factor`` using profile
      means and uncertainties.
    - The function produces several figures and can be slow for large datasets.
    - No cuts or data validation are performed internally.

    """

    assert len(set(df_pmt["counter_id"])) == 1 
    assert len(set(df_pmt["module_id"])) == 1 

    assert len(set(df_sipm["counter_id"])) == 1 
    assert len(set(df_sipm["module_id"])) == 1 

    assert len(set(df_sipm["event_id"])) == len(df_sipm["event_id"])
    assert len(set(df_pmt["event_id"])) == len(df_pmt["event_id"])

    
    seconds_in_day = 3600 * 24 
    seconds_in_week = 7 * seconds_in_day
    
    gps_min = min(df_pmt["gps"])
    gps_max = max(df_sipm["gps"])
    
    gps_bin_edges = np.arange(gps_min, gps_max+1, step=4*seconds_in_week)
    
    # -----------------------
    ## PMT profiles 
    
        
    # WCD signal 
    p_sd_signal_pmt = Profile(x=df_pmt["gps"], y=df_pmt["sd_signal"]).doProfile(gps_bin_edges)
        
    # k - # activated bars inifinte window 
    p_k_pmt = Profile(x=df_pmt["gps"], y=df_pmt["k_1x1"]).doProfile(gps_bin_edges)
        
    # rho_mu 
    p_rho_pmt = Profile(x=df_pmt["gps"], y=df_pmt["rho_shower_plane"]).doProfile(gps_bin_edges)
        
    # -----------------------
    ## SiPM profiles 
    
    # WCD signal 
    p_sd_signal_sipm = Profile(x=df_sipm["gps"], y=df_sipm["sd_signal"]).doProfile(gps_bin_edges)
        
    # k - # activated bars inifinte window 
    p_k_sipm = Profile(x=df_sipm["gps"], y=df_sipm["k"]).doProfile(gps_bin_edges)
        
    # rho_mu 
    p_rho_sipm = Profile(x=df_sipm["gps"], y=df_sipm["rho_shower_plane"]).doProfile(gps_bin_edges)
        
    color_pmt = "b"
    color_sipm = "g"
    
    ## PLOTS 
    # WCD signal vs time 
    fig, ax = plt.subplots(1, 1, figsize=(20, 11))
    
    non_zero_bool_pmt = np.array(p_k_pmt.y_mean) > 0
    non_zero_bool_sipm = np.array(p_k_sipm.y_mean) > 0
    
    ax.errorbar(np.array(p_sd_signal_pmt.x_mean)[non_zero_bool_pmt], 
                np.array(p_sd_signal_pmt.y_mean)[non_zero_bool_pmt], 
                yerr=np.array(p_sd_signal_pmt.y_std_of_mean)[non_zero_bool_pmt], fmt="o", ms=10, 
                color=color_pmt,
                label="PMT")
    ax.errorbar(np.array(p_sd_signal_sipm.x_mean)[non_zero_bool_sipm], 
                np.array(p_sd_signal_sipm.y_mean)[non_zero_bool_sipm], 
                yerr=np.array(p_sd_signal_sipm.y_std_of_mean)[non_zero_bool_sipm], fmt="o", ms=10, 
                color=color_sipm,
                label="SiPM")
    
    ax.set_xlabel("GPS time / s")
    ax.set_ylabel(r"$\langle \text{WCD signal} \rangle / \text{VEM}$")
    ax.set_ylim([0, 250])
    ax.legend()
    
    
    # k vs time 
    fig2, ax2 = plt.subplots(1, 1, figsize=(20, 11))
    
    ax2.errorbar(np.array(p_k_pmt.x_mean)[non_zero_bool_pmt], 
                 np.array(p_k_pmt.y_mean)[non_zero_bool_pmt], 
                yerr=np.array(p_k_pmt.y_std_of_mean)[non_zero_bool_pmt], fmt="o", ms=10, 
                color=color_pmt,
                label="PMT")
    ax2.errorbar(np.array(p_k_sipm.x_mean)[non_zero_bool_sipm], 
                 np.array(p_k_sipm.y_mean)[non_zero_bool_sipm], 
                yerr=np.array(p_k_sipm.y_std_of_mean)[non_zero_bool_sipm], fmt="o", ms=10, color=color_sipm,
                label="SiPM")
    ax2.set_ylim([0, 15])
    
    ax2.set_xlabel("GPS time / s")
    ax2.set_ylabel(r"$\langle k \rangle$")
    ax2.legend()
    
    # rho_mu vs time 
    fig3, ax3 = plt.subplots(1, 1, figsize=(20, 11))
    
    ax3.errorbar(np.array(p_rho_pmt.x_mean)[non_zero_bool_pmt], 
                 np.array(p_rho_pmt.y_mean)[non_zero_bool_pmt], 
                yerr=np.array(p_rho_pmt.y_std_of_mean)[non_zero_bool_pmt], fmt="o", ms=10, color=color_pmt,
                label="PMT")
    ax3.errorbar(np.array(p_rho_sipm.x_mean)[non_zero_bool_sipm], 
                 np.array(p_rho_sipm.y_mean)[non_zero_bool_sipm], 
                yerr=np.array(p_rho_sipm.y_std_of_mean)[non_zero_bool_sipm], fmt="o", ms=10, color=color_sipm,
                label="SiPM")
    
    ax3.set_ylim([0, 3])
    ax3.set_xlabel("GPS time / s")
    ax3.set_ylabel(r"$\langle \rho_{\mu} \rangle / \text{m}^{-2}$")
    ax3.legend()
    
    ## WCD signal distribution
    fig4, ax4 = plt.subplots(1, 1, figsize=(20, 11))
    
    sd_signal_bins = 10**np.linspace(0.5, 2.3, 11)

    
    ax4.hist(df_pmt["sd_signal"], bins=sd_signal_bins, density=True, histtype="step", lw=2,
             color=color_pmt, label="PMT")
    ax4.axvline(x=np.mean(df_pmt["sd_signal"].to_numpy()), ls="-", color=color_pmt, lw=1.5)
    ax4.axvline(x=np.median(df_pmt["sd_signal"].to_numpy()), ls="--", color=color_pmt, lw=1.5)
    
    ax4.hist(df_sipm["sd_signal"], bins=sd_signal_bins, density=True, histtype="stepfilled",
             lw=2, alpha=0.5,
             color=color_sipm, label="SiPM")
    
    ax4.axvline(x=np.mean(df_sipm["sd_signal"].to_numpy()), ls="-", color=color_sipm, lw=1.5)
    ax4.axvline(x=np.median(df_sipm["sd_signal"].to_numpy()), ls="--", color=color_sipm, lw=1.5)
    
    ax4.set_xscale("log")
    ax4.set_xlabel("WCD signal / VEM")
    ax4.set_ylabel("pdf")
    ax4.legend()
    
    ## rho_mu_dist 
    fig5, ax5 = plt.subplots(1, 1, figsize=(20, 11))
    
    rho_bins = 10**np.linspace(-1, 2.3, 12)
    
    rho_bins = np.arange(-1, 20, step=0.5)
    
    ax5.hist(df_pmt["rho_shower_plane"], bins=rho_bins, density=True, histtype="step", lw=2,
             color=color_pmt, label="PMT")
    ax5.axvline(x=np.mean(df_pmt["rho_shower_plane"].to_numpy()), ls="-", color=color_pmt, lw=1.5)
    ax5.axvline(x=np.median(df_pmt["rho_shower_plane"].to_numpy()), ls="--", color=color_pmt, lw=1.5)
    
    ax5.hist(df_sipm["rho_shower_plane"], bins=rho_bins, density=True, histtype="stepfilled",
             lw=2, alpha=0.5,
             color=color_sipm, label="SiPM")
    
    ax5.axvline(x=np.mean(df_sipm["rho_shower_plane"].to_numpy()), ls="-", color=color_sipm, lw=1.5)
    ax5.axvline(x=np.median(df_sipm["rho_shower_plane"].to_numpy()), ls="--", color=color_sipm, lw=1.5)
    
    #ax5.set_xscale("log")
    ax5.set_xlabel(r"$\rho_{\mu} / \text{m}^{-2}$")
    ax5.set_ylabel("pdf")
    ax5.legend()
    
    ## mean ldf observed for this particular module 
    fig6, ax6 = plt.subplots(1, 1, figsize=(20, 11))
    
    r_bin_edges = np.arange(200, 801, step=150)
    p_ldf_pmt = Profile(df_pmt["r"], df_pmt["rho_shower_plane"]).doProfile(bins=list(r_bin_edges))
    n_ldf_pmt = p_ldf_pmt.n_per_bin
    
    p_ldf_sipm = Profile(df_sipm["r"], df_sipm["rho_shower_plane"]).doProfile(bins=list(r_bin_edges))
    n_ldf_sipm = p_ldf_sipm.n_per_bin
    
    # scaling factor ldf 
    fit_scaling_ldf = fit_scaling_factor(y1=p_ldf_sipm.y_mean, y2=p_ldf_pmt.y_mean, 
                       y1_err=p_ldf_sipm.y_std_of_mean, y2_err=p_ldf_pmt.y_std_of_mean)
    
        
    ax6.errorbar(p_ldf_pmt.x_mean, p_ldf_pmt.y_mean, yerr=p_ldf_pmt.y_std_of_mean,
                 fmt="o", color=color_pmt, label=f"PMT ({n_ldf_pmt} entries)")
    ax6.errorbar(p_ldf_sipm.x_mean, p_ldf_sipm.y_mean, yerr=p_ldf_sipm.y_std_of_mean,
                 fmt="s", color=color_sipm, label=f"SiPM {n_ldf_sipm} entries")
    
    ax6.set_title(f"scaling factor = {fit_scaling_ldf['scaling_factor']:.3f} +- {fit_scaling_ldf['scaling_factor_err']:.3f}")
    ax6.set_xlabel("r / m")
    ax6.set_ylabel(r"$\rho_{\mu} / \text{m}^{-2}$")
    
    ax6.set_xscale("log")
    ax6.set_yscale("log")
    ax6.legend()
    
    ## correlation WCD signal vs k or rho_mu
    fig7, ax7 = plt.subplots(1, 1, figsize=(20, 11))
    
    p_sd_rho_pmt = Profile(df_pmt["sd_signal"], df_pmt["rho_shower_plane"]).doProfile(bins=list(sd_signal_bins))
        
    p_sd_rho_sipm = Profile(df_sipm["sd_signal"], df_sipm["rho_shower_plane"]).doProfile(bins=list(sd_signal_bins))
    
    n_sd_rho_pmt = np.array(p_sd_rho_pmt.n_per_bin)
    n_sd_rho_sipm = np.array(p_sd_rho_sipm.n_per_bin)

    mask_sd_rho_pmt = n_sd_rho_pmt > 3
    mask_sd_rho_sipm = n_sd_rho_sipm > 3

    mask_all = (mask_sd_rho_pmt) & (mask_sd_rho_sipm)

    # scaling factor wcd signal vs rho_mu 
    fit_scaling_sd_rho = fit_scaling_factor(y1=np.array(p_sd_rho_sipm.y_mean)[mask_all], 
                                            y2=np.array(p_sd_rho_pmt.y_mean)[mask_all], 
                                            y1_err=np.array(p_sd_rho_sipm.y_std_of_mean)[mask_all], 
                                            y2_err=np.array(p_sd_rho_pmt.y_std_of_mean)[mask_all])
    
    ax7.set_title(f"scaling factor = {fit_scaling_sd_rho['scaling_factor']:.3f} +- {fit_scaling_sd_rho['scaling_factor_err']:.3f}")
    ax7.errorbar(p_sd_rho_pmt.x_mean, p_sd_rho_pmt.y_mean, yerr=p_sd_rho_pmt.y_std_of_mean,
                 fmt="o", color=color_pmt, label="PMT")
    ax7.errorbar(p_sd_rho_sipm.x_mean, p_sd_rho_sipm.y_mean, yerr=p_sd_rho_sipm.y_std_of_mean,
                 fmt="s", color=color_sipm, label="SiPM")
    
    ax7.set_xlabel("WCD signal / VEM")
    ax7.set_ylabel(r"$\rho_{\mu} / \text{m}^{-2}$")
    
    ax7.set_xscale("log")
    ax7.set_yscale("log")
    ax7.legend()
    
    ## k distribution 
    fig8, ax8 = plt.subplots(1, 1, figsize=(20, 11))
    
    k_bins = np.arange(0, 40, step=1) - 0.5 
    
    ax8.hist(df_pmt["k_1x1"], histtype="step", lw=2, color=color_pmt,
                density=True, bins=k_bins, label=f"PMT ({len(df_pmt)} entries)")
    
    ax8.hist(df_sipm["k"], bins=k_bins, density=True, histtype="stepfilled",
             lw=2, alpha=0.5,
             color=color_sipm, label=f"SiPM ({len(df_sipm)} entries)")
    
    ax8.set_xlabel("k")
    ax8.set_ylabel("Probability")
    ax8.legend()
    
    ####################################################################
    # test on the relative efficiency of SiPM / PMT as a function of r

    ns_pmt = df_pmt["n_segments"].to_numpy() 
    assert len(set(ns_pmt)) == 1 
    ns_pmt = ns_pmt[0]

    
    p_k_ldf_pmt = Profile(df_pmt["r"], df_pmt["k_1x1"]).doProfile(r_bin_edges)
    mean_ks_pmt = np.array(p_k_ldf_pmt.y_mean)

    eps_pmt = np.log(1-mean_ks_pmt / ns_pmt)

    ns_sipm = df_sipm["n_segments"].to_numpy() 
    assert len(set(ns_sipm)) == 1 
    ns_sipm = ns_sipm[0]

    p_k_ldf_sipm = Profile(df_sipm["r"], df_sipm["k"]).doProfile(r_bin_edges)
    mean_ks_sipm = np.array(p_k_ldf_sipm.y_mean)

    eps_sipm = np.log(1-mean_ks_sipm / ns_sipm)

    ratio = eps_pmt / eps_sipm
    
    # error in numerator (assuming k is binomial(ns, p=1-exp(-mu*epsilon/ns)))
    ratio_error = []

    for i, (kpmt, ksipm) in enumerate(zip(p_k_ldf_pmt.y_mean, p_k_ldf_sipm.y_mean)):
        npmt = p_k_ldf_pmt.n_per_bin[i] 
        nsipm = p_k_ldf_sipm.n_per_bin[i]

        p_pmt = kpmt / ns_pmt 
        p_sipm = ksipm / ns_sipm 

        num_err = np.sqrt( p_pmt / (npmt * ns_pmt * (1-p_pmt)) )
        denom_err = np.sqrt( p_sipm / (nsipm * ns_sipm * (1-p_sipm)) )

        rerr = ratio[i] * np.sqrt( (num_err / eps_pmt[i])**2 + (denom_err / eps_sipm[i])**2 )

        ratio_error.append(rerr)
    
    fig9, ax9 = plt.subplots(1, 1, figsize=(20, 11))

    ax9.errorbar(p_k_ldf_sipm.x_center, ratio, yerr=ratio_error,
                 fmt="o", color="k")

    ax9.set_xlabel(r"$r / \text{m}$")
    ax9.set_ylabel(r"$\epsilon_\text{PMT} / \epsilon_\text{SiPM}$")

    ax9.axhline(y=1, ls="--", color="grey", alpha=0.8)


    
    return {"fig": fig, "ax": ax,
            "fig2": fig2, "ax2": ax2,
            "fig3": fig3, "ax3": ax3,
            "fig4": fig4, "ax4": ax4,
            "fig5": fig5, "ax5": ax5,
            "fig6": fig6, "ax6": ax6,
            "fig7": fig7, "ax7": ax7,
            "fig8": fig8, "ax8": ax8,
            "fig9": fig9, "ax9": ax9,
            "scaling_factor_ldf": fit_scaling_ldf["scaling_factor"],
            "scaling_factor_ldf_err": fit_scaling_ldf["scaling_factor_err"],
            "scaling_factor_sd_rho": fit_scaling_sd_rho["scaling_factor"],
            "scaling_factor_sd_rho_err": fit_scaling_sd_rho["scaling_factor_err"]}
    
def plot_relative_efficiency_module_level(df_pmt, df_sipm):
    """
    """
    
    assert len(set(df_pmt["counter_id"])) == 1 
    assert len(set(df_pmt["module_id"])) == 1 
    
    assert len(set(df_sipm["counter_id"])) == 1 
    assert len(set(df_sipm["module_id"])) == 1 

    assert len(set(df_sipm["event_id"])) == len(df_sipm["event_id"])
    assert len(set(df_pmt["event_id"])) == len(df_pmt["event_id"])

    #assert len(set(ns_pmt)) == 1, f"{set(ns_pmt)}, {set(df_pmt['counter_id'])}, {set(df_pmt['module_id'])}" 
    #ns_pmt = ns_pmt[0]

    #ns_sipm = df_sipm["n_segments"].to_numpy() 
    #assert len(set(ns_sipm)) == 1, f"{set(ns_sipm)}" 
    #ns_sipm = ns_sipm[0]

    # segmentation for PMT era of this module
    counter_id_pmt = df_pmt["counter_id"].iloc[0]
    mod_id_pmt = df_pmt["module_id"].iloc[0]
    n_broken_pmt = 0
    mask_pmt = (broken_channels_pmt["counter_id"] == counter_id_pmt) & (broken_channels_pmt["module_id"] == mod_id_pmt)
    if mask_pmt.any():
        channels_broken_list_pmt = np.unique(broken_channels_pmt[mask_pmt].copy()["channelsBroken_list"].iloc[0])

        print("")
        print(f"channels_broken_list_pmt for {counter_id_pmt}, module_id_pmt = {mod_id_pmt}")
        print(f"{channels_broken_list_pmt}")
        print("")

        n_broken_pmt = len(channels_broken_list_pmt) 
    
    ns_pmt = 64 - n_broken_pmt

    # segmentation for SiPM era of this module
    counter_id_sipm = df_sipm["counter_id"].iloc[0]
    mod_id_sipm = df_sipm["module_id"].iloc[0]
    n_broken_sipm = 0
    mask_sipm = (broken_channels_sipm["counter_id"] == counter_id_sipm) & (broken_channels_sipm["module_id"] == mod_id_sipm)
    if mask_sipm.any():
        channels_broken_list_sipm = np.unique(broken_channels_sipm[mask_sipm].copy()["channelsBroken_list"].iloc[0])

        print("")
        print(f"channels_broken_list_sipm for {counter_id_sipm}, module_id_sipm = {mod_id_sipm}")
        print(f"{channels_broken_list_sipm}")
        print("")
        n_broken_sipm = len(channels_broken_list_sipm) 
    
    ns_sipm = 64 - n_broken_sipm

    sd_signal_bins = 10**np.linspace(1., 2.3, 7)
    #sd_signal_bins = np.arange(10, 200, step=10)

    p_k_sd_pmt = Profile(df_pmt["sd_signal"], df_pmt["k_1x1"]).doProfile(bins=list(sd_signal_bins))
    p_k_sd_sipm = Profile(df_sipm["sd_signal"], df_sipm["k"]).doProfile(bins=list(sd_signal_bins))

    ratio, ratio_err = [], []
    ratio_mle, ratio_mle_err = [], []

    for i, (kpmt, ksipm) in enumerate(zip(p_k_sd_pmt.y_mean, p_k_sd_sipm.y_mean)):
        p_pmt = kpmt / ns_pmt
        p_sipm = ksipm / ns_sipm

        eps_pmt = np.log(1-p_pmt)
        eps_sipm = np.log(1-p_sipm)

        n_data_pmt = p_k_sd_pmt.n_per_bin[i]
        n_data_sipm = p_k_sd_sipm.n_per_bin[i]

        r = eps_pmt / eps_sipm if kpmt != 0 else 0

        sigma_pmt = np.sqrt( p_pmt / (n_data_pmt * ns_pmt * (1-p_pmt)) )
        sigma_sipm = np.sqrt( p_sipm / (n_data_sipm * ns_sipm * (1-p_sipm)) )

        rerr = r * np.sqrt( (sigma_pmt/eps_pmt)**2 + (sigma_sipm/eps_sipm)**2 )

        ratio.append(r)
        ratio_err.append(rerr)

        # full distribution of k for PMT for this bin
        k_dist_pmt = p_k_sd_pmt.y_full_dist[i]
        # full distribution of k for SiPM for this bin
        k_dist_sipm = p_k_sd_sipm.y_full_dist[i]

        est = EfficiencyRatioEstimator(K1=k_dist_sipm, K2=k_dist_pmt, n_segments1=ns_sipm, n_segments2=ns_pmt)
        mu_hat, R_hat = est.fit()
        sigma_mu, sigma_R = est.get_errors()
        # R = eff2 / eff1 = eff_PMT / eff_SiPM

        print(f"mu = {mu_hat:.3f} ± {sigma_mu:.3f}")
        print(f"R  = {R_hat:.3f} ± {sigma_R:.3f}")

        ratio_mle.append(R_hat)
        ratio_mle_err.append(sigma_R)

    # masking 
    #mask = (np.array(p_k_sd_pmt.n_per_bin)) > 5 (np.array(p_k_sd_sipm.n_per_bin)) > 5

    ratio = np.array(ratio)
    ratio_err = np.array(ratio_err)
    

    fig, ax = plt.subplots(1, 1, figsize=(13, 9))


    ax.errorbar(p_k_sd_sipm.x_center, ratio, yerr=ratio_err, fmt="o", 
                color="k")
    

    ax.errorbar(p_k_sd_sipm.x_center, ratio_mle, yerr=ratio_mle_err, fmt="s", 
                color="g", mfc="None", label="likelihood")
    
    ax.set_xlabel("WCD signal / VEM")
    ax.set_ylabel(r"$\epsilon_\text{PMT} / \epsilon_\text{SiPM}$")
    ax.set_title(f"ns_pmt = {ns_pmt}, ns_sipm = {ns_sipm}")

    ax.set_xscale("log")
    ax.axhline(y=1, ls="--", color="grey", alpha=0.9)

    return {"fig": fig, "ratio": ratio, "ratio_err": ratio_err}

def plot_relative_efficiency_per_module(df_pmt, df_sipm, theta_min, theta_max, 
                                        lge_min, out_dir):   
    """
    """
    # Mapping between module ids from the PMT era to the SiPM era. 
    # Only to those modules that were in acquisition in the two periods
    mod_id_pmt_to_sipm = {1773: {1: 101},
                          93: {1: 101, 2: 102, 3: 103, 4: 104},
                          1574: {1: 101, 2: 102, 3: 103, 4: 104},
                          1570: {1: 101},
                          688: {1: 101}, 
                          1764: {1: 101, 2: 102, 3: 103, 4: 104,
                                 5: 105, 6: 106, 7: 107, 8: 108},
                          1622: {1: 101, 2: 102, 3: 103, 4: 104, 
                                 5: 105, 6: 106}}

    cuts_pmt = ((theta_min < df_pmt["theta"]) & (df_pmt["theta"] < theta_max) & 
                (lge_min < df_pmt["lge"]))
    
    df_pmt_cut = df_pmt[cuts_pmt]
    
    cuts_sipm = ((theta_min < df_sipm["theta"]) & (df_sipm["theta"] < theta_max) & 
                (lge_min < df_sipm["lge"]))
    
    df_sipm_cut = df_sipm[cuts_sipm]

    rel_eff_dict = {}

    counter_module_string, r, r_err = [], [], []
    
    for counter_id in mod_id_pmt_to_sipm.keys():
        counter_pmt_bool = df_pmt_cut["counter_id"] == counter_id 
        if counter_id == 93:
            counter_sipm_bool = (df_sipm_cut["counter_id"] == 93) | (df_sipm_cut["counter_id"] == 710)
        else:
            counter_sipm_bool = df_sipm_cut["counter_id"] == counter_id 
        
        df_pmt_c = df_pmt_cut[counter_pmt_bool]
        df_sipm_c = df_sipm_cut[counter_sipm_bool]

        # creating output directory for each counter insider out_dir 
        out_dir_counter = out_dir + f"c{counter_id}"
        os.makedirs(out_dir_counter, exist_ok=True)
        
        for mod_id_pmt in mod_id_pmt_to_sipm[counter_id].keys():
            mod_id_sipm = mod_id_pmt_to_sipm[counter_id][mod_id_pmt]
            
            df_pmt_m = df_pmt_c[ df_pmt_c["module_id"] == mod_id_pmt ]
            df_sipm_m = df_sipm_c[ df_sipm_c["module_id"] == mod_id_sipm ]
            
            if len(df_pmt_m) == 0 or len(df_sipm_m) == 0:
                print(f"one of the dataframes is empty for c{counter_id} m{mod_id_pmt}.. skipped..")
                continue
            
            area = np.mean(df_pmt_m["active_area"])
                        
            
            title_cuts = (f"theta_min = {theta_min:.0f}, theta_max = {theta_max:.0f}, "
                  f"lge_min = {lge_min:.1f}, area = {area:.1f} m2\n")

            title_cuts += f"c{counter_id} m{mod_id_pmt} (PMT) / m{mod_id_sipm} (SiPM)"

            ## plot 
            fig_dict = plot_relative_efficiency_module_level(df_pmt_m, df_sipm_m)

            fig_dict["fig"].suptitle(title_cuts)
                       
            save_figure(fig_dict["fig"], (out_dir_counter + "/" + f"c{counter_id}m{mod_id_pmt}_th{theta_min:.0f}-{theta_max:.0f}_lgemin{lge_min:.1f}"
                                            "rel_eff_sd_signal.png"))
            
            ratio = fig_dict["ratio"]
            ratio_err = fig_dict["ratio_err"]
            #print(ratio)
            mask = ~np.isnan(ratio)

            rel_eff_dict[f"c{counter_id}m{mod_id_pmt}"] = compute_weighted_mean(ratio[mask], ratio_err[mask])

            counter_module_string.append(f"c{counter_id}-m{mod_id_pmt}")
            
            r.append(compute_weighted_mean(ratio[mask], ratio_err[mask]))
            r_err.append(standard_error_of_weighted_mean(ratio[mask], ratio_err[mask]))


    
    print("relative efficiency per module (as a function of sd signal): ")
    for cm, rr in rel_eff_dict.items():
        print(f"{cm}  --> {rr:.3f}")
            
    # figure with the ratio per module 
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    x_dummy = range(len(counter_module_string))

    print(f"{len(counter_module_string)}, {len(r)}, {len(r_err)}")

    ax.errorbar(x_dummy, r, yerr=r_err, fmt="o", mfc="None", color="k")
    ax.axhline(y=1, ls="--", color="grey")

    ax.set_xticks(x_dummy)
    ax.set_xticklabels(counter_module_string, rotation=45, fontsize="small")

    ax.set_xlabel("Module")
    ax.set_ylabel(r"R = $\epsilon_\text{PMT} \, / \, \epsilon_\text{SiPM}$")

    save_figure(fig, out_dir + f"rel_eff_sd_signal_all_modules_th{theta_min:.0f}-{theta_max:.0f}_lgemin{lge_min:.1f}.png")
            

def plot_modules_time_series(df_pmt, df_sipm, lge_min, lge_max,
                             theta_min, theta_max, 
                             rmin, rmax, out_dir):
    """
    Run time-series and comparison analyses for all UMD modules operated in
    both the PMT and SiPM eras, and save the resulting figures to disk.

    The function applies common physics cuts in energy, zenith angle, and
    core distance, identifies modules that were in acquisition during both
    eras using a predefined PMT→SiPM module mapping, and for each matched
    module pair calls :func:`plot_time_series`. All per-module figures are
    saved to ``out_dir``.

    In addition, the function collects the PMT/SiPM scaling factors obtained
    from each module (mean LDF scaling and WCD signal-muon density scaling),
    produces summary plots of these scaling factors across modules, and saves
    the summary figures. Modules with scaling factors below the median are
    reported to stdout.

    Parameters
    ----------
    df_pmt : pandas.DataFrame
        Module-level measurements for all UMD modules operating in the PMT era.
        The DataFrame must contain, at minimum, the columns used by
        :func:`plot_time_series` as well as:

        - ``counter_id`` : station identifier
        - ``module_id`` : module identifier within the station
        - ``theta`` : zenith angle (deg)
        - ``lge`` : log10(primary energy)
        - ``r`` : distance to the shower axis (m)
        - ``active_area`` : module active area (m^2)

    df_sipm : pandas.DataFrame
        Same as ``df_pmt``, but corresponding to the SiPM era.

    lge_min, lge_max : float
        Minimum and maximum log10(primary energy) used for event selection.

    theta_min, theta_max : float
        Minimum and maximum zenith angle (degrees).

    rmin, rmax : float
        Minimum and maximum distance to the shower axis (meters).

    out_dir : str
        Output directory where all figures are written. The directory is
        assumed to exist.

    Returns
    -------
    None
        This function does not return any object. Its output consists of
        figures written to disk and summary information printed to stdout.

    Notes
    -----
    - The PMT→SiPM module correspondence is hard-coded via an internal mapping
      and restricted to modules that operated in both eras.
    - All event-level cuts are applied internally; no additional cuts are
      performed inside :func:`plot_time_series`.
    - Per-module figures include time evolutions, distributions, LDFs, and
      correlation plots; summary figures show the distribution of scaling
      factors across modules.
    - Large-area modules (active area > 7 m^2) are highlighted in the summary
      scaling-factor plots.
    - This function can be slow and produce many files for large datasets.
    """
    
    
    # Mapping between module ids from the PMT era to the SiPM era. 
    # Only to those modules that were in acquisition in the two periods
    mod_id_pmt_to_sipm = {1773: {1: 101},
                          93: {1: 101, 2: 102, 3: 103, 4: 104},
                          1574: {1: 101, 2: 102, 3: 103, 4: 104},
                          1570: {1: 101},
                          688: {1: 101}, 
                          1764: {1: 101, 2: 102, 3: 103, 4: 104,
                                 5: 105, 6: 106, 7: 107, 8: 108},
                          1622: {1: 101, 2: 102, 3: 103, 4: 104, 
                                 5: 105, 6: 106}}
    
    
    cuts_pmt = ((theta_min < df_pmt["theta"]) & (df_pmt["theta"] < theta_max) & 
                (lge_min < df_pmt["lge"]) & (df_pmt["lge"] < lge_max) &
                (df_pmt["r"] > rmin) & (df_pmt["r"] < rmax))
    
    df_pmt_cut = df_pmt[cuts_pmt]
    
    cuts_sipm = ((theta_min < df_sipm["theta"]) & (df_sipm["theta"] < theta_max) & 
                (lge_min < df_sipm["lge"]) & (df_sipm["lge"] < lge_max) & 
                (df_sipm["r"] > rmin) & (df_sipm["r"] < rmax))
    
    df_sipm_cut = df_sipm[cuts_sipm]
    

    scaling_factors_ldf, scaling_factors_ldf_err = [], [] 
    scaling_factors_sd_rho, scaling_factors_sd_rho_err = [], [] 
    counter_mod_string = []
    area_mod = []

    string_cuts = f"th{theta_min:.0f}-{theta_max:.0f}_lge{lge_min}-{lge_max}"
    
    title_cuts_base = (f"theta_min = {theta_min:.0f}, theta_max = {theta_max:.0f}, "
                  f"lge_min = {lge_min:.1f}, lge_max = {lge_max:.1f}, rmin = {rmin:.0f} m, rmax = {rmax:.0f} m\n")

    for counter_id in mod_id_pmt_to_sipm.keys():
        counter_pmt_bool = df_pmt_cut["counter_id"] == counter_id 
        if counter_id == 93:
            counter_sipm_bool = (df_sipm_cut["counter_id"] == 93) | (df_sipm_cut["counter_id"] == 710)
        else:
            counter_sipm_bool = df_sipm_cut["counter_id"] == counter_id 
        
        df_pmt_c = df_pmt_cut[counter_pmt_bool]
        df_sipm_c = df_sipm_cut[counter_sipm_bool]

        # creating output directory for each counter insider out_dir 
        out_dir_counter = out_dir + f"c{counter_id}"
        os.makedirs(out_dir_counter, exist_ok=True)
        
        for mod_id_pmt in mod_id_pmt_to_sipm[counter_id].keys():
            mod_id_sipm = mod_id_pmt_to_sipm[counter_id][mod_id_pmt]
            
            df_pmt_m = df_pmt_c[ df_pmt_c["module_id"] == mod_id_pmt ]
            df_sipm_m = df_sipm_c[ df_sipm_c["module_id"] == mod_id_sipm ]
            
            if len(df_pmt_m) == 0 or len(df_sipm_m) == 0:
                print(f"one of the dataframes is empty for c{counter_id} m{mod_id_pmt}.. skipped..")
                continue
            
            area = np.mean(df_pmt_m["active_area"])
            area_mod.append(area)
            
          
            title_cuts = title_cuts_base + f"c{counter_id} m{mod_id_pmt} (PMT) / m{mod_id_sipm} (SiPM)"
            
            ## plot 
            fig_dict = plot_time_series(df_pmt_m, df_sipm_m)
            
            fig_dict["fig"].suptitle(title_cuts)
            fig_dict["fig2"].suptitle(title_cuts)
            fig_dict["fig3"].suptitle(title_cuts)
            fig_dict["fig4"].suptitle(title_cuts)
            fig_dict["fig5"].suptitle(title_cuts)
            fig_dict["fig6"].suptitle(title_cuts)
            fig_dict["fig7"].suptitle(title_cuts)
            fig_dict["fig8"].suptitle(title_cuts)
            fig_dict["fig9"].suptitle(title_cuts)

            
            
            save_figure(fig_dict["fig"], out_dir_counter + "/" + f"c{counter_id}m{mod_id_pmt}_sd_signal_{string_cuts}.png")
            save_figure(fig_dict["fig2"], out_dir_counter + "/" + f"c{counter_id}m{mod_id_pmt}_k_{string_cuts}.png")
            save_figure(fig_dict["fig3"], out_dir_counter + "/" + f"c{counter_id}m{mod_id_pmt}_rho_mu_{string_cuts}.png")
            save_figure(fig_dict["fig4"], out_dir_counter + "/" + f"c{counter_id}m{mod_id_pmt}_sd_signal_dist_{string_cuts}.png")
            save_figure(fig_dict["fig5"], out_dir_counter + "/" + f"c{counter_id}m{mod_id_pmt}_rho_dist_{string_cuts}.png")
            save_figure(fig_dict["fig6"], out_dir_counter + "/" + f"c{counter_id}m{mod_id_pmt}_ldf_{string_cuts}.png")
            save_figure(fig_dict["fig7"], out_dir_counter + "/" + f"c{counter_id}m{mod_id_pmt}_sd_rho_corr_{string_cuts}.png")
            save_figure(fig_dict["fig8"], out_dir_counter + "/" + f"c{counter_id}m{mod_id_pmt}_k_dist_{string_cuts}.png")
            save_figure(fig_dict["fig9"], out_dir_counter + "/" + f"c{counter_id}m{mod_id_pmt}_rel_eff_{string_cuts}.png")

            scaling_factors_ldf.append(fig_dict["scaling_factor_ldf"])
            scaling_factors_ldf_err.append(fig_dict["scaling_factor_ldf_err"])
            scaling_factors_sd_rho.append(fig_dict["scaling_factor_sd_rho"])
            scaling_factors_sd_rho_err.append(fig_dict["scaling_factor_sd_rho_err"])
            counter_mod_string.append(f"c{counter_id}m{mod_id_pmt}")
            
    
    # convert to numpy arrays 
    x_scaling = np.arange(len(scaling_factors_ldf))
    scaling_factors_ldf = np.array(scaling_factors_ldf)
    scaling_factors_ldf_err = np.array(scaling_factors_ldf_err)
    scaling_factors_sd_rho = np.array(scaling_factors_sd_rho)
    scaling_factors_sd_rho_err = np.array(scaling_factors_sd_rho_err)
    area_mod = np.array(area_mod)
    
    # scaling factors LDF 
    fig, ax = plt.subplots(2, 1, figsize=(18, 14))
    ax = ax.flatten() 
    
    
    ax[0].errorbar(x_scaling, 
                   scaling_factors_ldf,
                yerr=scaling_factors_ldf_err, fmt="o", mfc="None", color="b", ms=9,
                label="Mean LDF")
    
    ax[0].errorbar(x_scaling, 
                   scaling_factors_sd_rho,
                yerr=scaling_factors_sd_rho_err, fmt="s", mfc="None", color="g", ms=9,
                label="WCD signal vs rho")
    
    # large area detectors as full markers 
    large_bool = area_mod > 7
    
    ax[0].errorbar(x_scaling[large_bool], 
                   scaling_factors_ldf[large_bool],
                yerr=scaling_factors_ldf_err[large_bool], fmt="o", alpha=0.6, 
                color="b", ms=9)
    ax[0].errorbar(x_scaling[large_bool], 
                   scaling_factors_sd_rho[large_bool],
                yerr=scaling_factors_sd_rho_err[large_bool], fmt="s", alpha=0.6, color="g", ms=9,
                )
    
    ax[0].set_xticks(x_scaling)
    ax[0].set_xticklabels(counter_mod_string, rotation=45)
    
    ax[0].set_xlabel("Module")
    ax[0].set_ylabel("Scaling factor")
    ax[0].axhline(y=1, ls="--", color="grey", lw=1.5, alpha=0.6) 
    ax[0].legend()
    
    ax[1].hist(scaling_factors_ldf, histtype="step", lw=2, color="b", label="Mean LDF")
    mean_scaling_ldf = np.mean(scaling_factors_ldf)
    median_scaling_ldf = np.median(scaling_factors_ldf)
    ax[1].axvline(mean_scaling_ldf, ls="--", color="b", 
                  label=f"mean = {mean_scaling_ldf:.3f}, median = {median_scaling_ldf:.3f}")
    
    mean_scaling_sd_rho = np.mean(scaling_factors_sd_rho)
    median_scaling_sd_rho = np.median(scaling_factors_sd_rho)
    ax[1].hist(scaling_factors_sd_rho, histtype="step", lw=2, color="g", label="WCD signal vs rho")
    ax[1].axvline(mean_scaling_sd_rho, ls="-.", color="g",
                  label=f"mean = {mean_scaling_sd_rho:.3f}, median = {median_scaling_sd_rho:.3f}")
    
    
    ax[1].set_xlabel("Scaling factor")
    ax[1].legend()
    
    fig.suptitle(title_cuts_base)

    fig.tight_layout(pad=1.1)
    
    save_figure(fig, out_dir + f"scaling_factors_all_modules_{string_cuts}.png")
    
    # Printing modules that have scaling factor below median 
    
    print(f"Modules with scaling factor below median {median_scaling_ldf:.3f}: ")
    for i, val in enumerate(scaling_factors_ldf):
        if val < median_scaling_ldf:
            print(f"{counter_mod_string[i]},  scaling = {val:.3f}")

    

def save_figure(fig, fname):
    """Save a matplotlib figure and close it."""
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print(f"{fname} generated")
    

# ================================================
# Load the csv module-level measurements
# ================================================


pmt_path = ("/storage/gpfs_data/pauger/users/jdejesus/kit/datasets/pmt/reproduced_dataset_2025/module_level_info/"
            "module_level_stratGap_nOnes2_nGaps1_wSize7_rejSmall0_rejCorr1_effSmall1.04_effLarge0.95_rMin200_minCounters3_ldfSara1_fixBeta1.csv")

sipm_path1 = ("/storage/gpfs_data/pauger/users/jdejesus/kit/datasets/sipm/data/module_level_info/"
            "module_level_info_sipm_tmin0_tmax2048_gpsMin1230336019_gpsMax1261872018.csv")

sipm_path2 = ("/storage/gpfs_data/pauger/users/jdejesus/kit/datasets/sipm/data/module_level_info/"
            "module_level_info_sipm_tmin0_tmax2048_gpsMin1261872019_gpsMax1275004818.csv")

sipm_path3 = ("/storage/gpfs_data/pauger/users/jdejesus/kit/datasets/sipm/data/module_level_info/"
            "module_level_info_sipm_tmin0_tmax2048_gpsMin1198800019_gpsMax1230336018.csv")

sipm_path4 = ("/storage/gpfs_data/pauger/users/jdejesus/kit/datasets/sipm/data/module_level_info/"
            "module_level_info_sipm_tmin0_tmax2048_gpsMin1275004819_gpsMax1293494418.csv")

sipm_path5 = ("/storage/gpfs_data/pauger/users/jdejesus/kit/datasets/sipm/data/module_level_info/"
            "module_level_info_sipm_tmin0_tmax2048_gpsMin1293494419_gpsMax1306540818.csv")

df_pmt = pd.read_csv(pmt_path)
#print(df_pmt.columns)
#print(df_pmt.info())

df1 = pd.read_csv(sipm_path1)
df2 = pd.read_csv(sipm_path2)
df3 = pd.read_csv(sipm_path3)
df4 = pd.read_csv(sipm_path4)
df5 = pd.read_csv(sipm_path5)

df_sipm = pd.concat([df1, df2, df3, df4, df5])
assert len(df_sipm) == len(df1) + len(df2) + len(df3) + len(df4) + len(df5)

print(df_sipm.columns)


# Define cuts
core_in_uc = 0
use_energy_icrc2025 = False
lge_bins = np.array([17.3, 17.5, 17.7, 17.9, 18.1, 18.3])
lge_bin_centers = 0.5 * (lge_bins[:-1] + lge_bins[1:])

# Create output directory if necessary 
out_dir = "../../outputs/mean_ldf/"

if use_energy_icrc2025:
    out_dir += "energy_icrc2025/"

os.makedirs(out_dir, exist_ok=True)

# =============================================
# Unique event-level quantities 
# =============================================
print("Unique event-level quantities...")


event_dist = plot_event_distributions(df_pmt=df_pmt, df_sipm=df_sipm, 
                                      core_in_uc=core_in_uc, lge_bins=lge_bins,
                                      use_energy_icrc2025=use_energy_icrc2025)


save_figure(event_dist["fig"], out_dir + f"compare_sipm_pmt_event_dist_coreInUC{int(core_in_uc)}.png")
save_figure(event_dist["fig2"], out_dir + f"compare_sipm_pmt_event_dist2_coreInUC{int(core_in_uc)}.png")
save_figure(event_dist["fig3"], out_dir + f"compare_sipm_pmt_event_dist3_coreInUC{int(core_in_uc)}.png")
save_figure(event_dist["fig4"], out_dir + f"compare_sipm_pmt_event_dist4_coreInUC{int(core_in_uc)}.png")

# ============================================
# Module-level profiles 
# ===========================================
print("")
print("Module-level quantities..")

# evntually trying alternative energy binning for the scaling factor
#lge_bins = np.array([17.3, 17.6, 17.9, 18.3])
#lge_bin_centers = 0.5 * (lge_bins[:-1] + lge_bins[1:])

# Define cuts
theta_min, theta_max = 0, 25
#theta_min, theta_max = 25, 45
module_size = "large"
scale_rho_mu = False

# scaling factors as a function of energy
scaling = {"scaling_mean_ldf": [], "scaling_mean_ldf_err": [],
            "scaling_rho_wcd": [], "scaling_rho_wcd_err": [],
            "k_rho_scaling": [], "k_rho_scaling_err": []}

for i, lge_center in enumerate(lge_bin_centers):
    lge1 = lge_bins[i]
    lge2 = lge_bins[i+1]


    fig_mean_ldf = plot_mean_ldf(
        df_pmt=df_pmt,
        df_sipm=df_sipm,
        theta_min=theta_min,
        theta_max=theta_max,
        lge_min=lge1,
        lge_max=lge2,
        module_size=module_size,
        core_in_uc=core_in_uc,
        use_energy_icrc2025=use_energy_icrc2025,
        scale_rho_mu=scale_rho_mu
    )

    # Saving figures
    save_figure(fig_mean_ldf["rho_vs_r"]["fig"], 
               (out_dir + f"rho_vs_r_th{theta_min:.0f}-{theta_max:.0f}_"
               f"lge{lge1:.1f}-{lge2:.1f}_{module_size}_coreInUC{int(core_in_uc)}.png"))

    save_figure(fig_mean_ldf["k_profiles"]["fig"], (out_dir + f"k_profiles_th{theta_min:.0f}-{theta_max:.0f}_"
                f"lge{lge1:.1f}-{lge2:.1f}_{module_size}_coreInUC{int(core_in_uc)}.png"))
   
    save_figure(fig_mean_ldf["rho_vs_sd"]["fig"], 
               (out_dir + f"rho_vs_sd_th{theta_min:.0f}-{theta_max:.0f}_"
               f"lge{lge1:.1f}-{lge2:.1f}_{module_size}_coreInUC{int(core_in_uc)}.png"))
    
    save_figure(fig_mean_ldf["k_vs_rho"]["fig"], 
               (out_dir + f"k_vs_rho_th{theta_min:.0f}-{theta_max:.0f}_"
               f"lge{lge1:.1f}-{lge2:.1f}_{module_size}_coreInUC{int(core_in_uc)}.png"))
    
    save_figure(fig_mean_ldf["k_vs_nmu"]["fig"], 
               (out_dir + f"k_vs_nmu_th{theta_min:.0f}-{theta_max:.0f}_"
               f"lge{lge1:.1f}-{lge2:.1f}_{module_size}_coreInUC{int(core_in_uc)}.png"))
    
    save_figure(fig_mean_ldf["k_fraction"]["fig"], 
               (out_dir + f"k_fraction_th{theta_min:.0f}-{theta_max:.0f}_"
               f"lge{lge1:.1f}-{lge2:.1f}_{module_size}_coreInUC{int(core_in_uc)}.png"))

    # Appending scaling factors 
    scaling["scaling_mean_ldf"].append(fig_mean_ldf["scaling_factors"]["mean_ldf_scaling"])
    scaling["scaling_mean_ldf_err"].append(fig_mean_ldf["scaling_factors"]["mean_ldf_scaling_err"])
    scaling["scaling_rho_wcd"].append(fig_mean_ldf["scaling_factors"]["rho_wcd_scaling"])
    scaling["scaling_rho_wcd_err"].append(fig_mean_ldf["scaling_factors"]["rho_wcd_scaling_err"])
    scaling["k_rho_scaling"].append(fig_mean_ldf["scaling_factors"]["k_rho_scaling"])
    scaling["k_rho_scaling_err"].append(fig_mean_ldf["scaling_factors"]["k_rho_scaling_err"])


    # relative efficiency 

    rel_eff_dict = plot_relative_efficiency(df_pmt=df_pmt,
                    df_sipm=df_sipm,
                    theta_min=theta_min,
                    theta_max=theta_max,
                    lge_min=lge1,
                    lge_max=lge2,
                    module_size=module_size,
                    core_in_uc=core_in_uc,
                    use_energy_icrc2025=use_energy_icrc2025)

    save_figure(rel_eff_dict["fig"], 
               (out_dir + f"relative_eff_th{theta_min:.0f}-{theta_max:.0f}_"
               f"lge{lge1:.1f}-{lge2:.1f}_{module_size}_coreInUC{int(core_in_uc)}.png"))


# el scaling en funcion de k no tiene mucho sentido!
# Si PMT y SiPM tienen diferentes respuestas, eficiencias, etc
# (como creemos que tenian), entonces es esperable que frente al mismo k
# la conversion a densidad de muones sea distinta

# scaling factors as a function of energy 
fig, ax = plt.subplots(1, 1, figsize=(14, 11))

ax.errorbar(lge_bin_centers, scaling["scaling_mean_ldf"], 
            yerr=scaling["scaling_mean_ldf_err"], fmt="o", color="b", alpha=0.9,
            label=r"$\rho_{\mu}$ vs r")

wmean_scaling = compute_weighted_mean(scaling["scaling_mean_ldf"], scaling["scaling_mean_ldf_err"])
wmean_scaling_err = standard_error_of_weighted_mean(scaling["scaling_mean_ldf"], scaling["scaling_mean_ldf_err"])
ax.axhline(y=wmean_scaling, color="b", ls="--", label=f"{wmean_scaling:.2f} +/- {wmean_scaling_err:.2f}")

#ax.errorbar(lge_bin_centers, scaling["scaling_rho_wcd"], 
            #yerr=scaling["scaling_rho_wcd_err"], fmt="s", color="orange", mfc="None", alpha=0.9,
            #label=r"$\rho_{\mu}$ vs WCD signal")

wmean_scaling_wcd = compute_weighted_mean(scaling["scaling_rho_wcd"],scaling["scaling_rho_wcd_err"])
wmean_scaling_wcd_err = standard_error_of_weighted_mean(scaling["scaling_rho_wcd"], scaling["scaling_rho_wcd_err"])
#ax.axhline(y=wmean_scaling_wcd, color="orange", ls="-.", label=f"{wmean_scaling_wcd:.2f} +/- {wmean_scaling_wcd_err:.2f}")

#ax.errorbar(lge_bin_centers, scaling["k_rho_scaling"], 
            #yerr=scaling["k_rho_scaling_err"], fmt="v", color="r", mfc="None", alpha=0.9,
            #label=r"$\rho_{\mu}$ vs k")

ax.set_xlabel(r"$\lg(E/\text{eV})$")
ax.set_ylabel("scaling factor")
ax.set_ylim([0.75, 1.03])
ax.legend()
ax.axhline(y=1, ls="-.", color="grey")
title_cuts = (
        f"{theta_min:.0f}° < theta < {theta_max:.0f}°, "
        f"module_size = {module_size}, core_in_uc = {core_in_uc}, use_energy_icrc2025 = {use_energy_icrc2025}, scale_rho_mu = {scale_rho_mu}"
    )
fig.suptitle(title_cuts)

save_figure(fig, 
               (out_dir + f"scaling_vs_lge_th{theta_min:.0f}-{theta_max:.0f}_"
               f"{module_size}_coreInUC{int(core_in_uc)}.png"))

plt.close(fig)


# ========================================
# Profile of rho450 / (E/10**18)**0.9, where rho450 is the mean rho_mu for
# 400 < r < 500

print("")
print(" Mean rho_mu around 450 as a function of energy...")

fig_rho450 = plot_rho450_vs_lge(df_pmt,
    df_sipm,
    theta_min,
    theta_max,
    lge_bin_edges=lge_bins,
    module_size=module_size,
    core_in_uc=core_in_uc,
    use_energy_icrc2025=use_energy_icrc2025)

save_figure(fig_rho450["fig"], 
               (out_dir + f"rho450_vs_lge_th{theta_min:.0f}-{theta_max:.0f}_"
               f"{module_size}_coreInUC{int(core_in_uc)}.png"))


# =========================================
print("")
print(" Nmu and Nmu_uncorr as a funciton of k...")

figs_k_nmu = plot_k_vs_nmu(df_pmt,
    df_sipm,
    theta_min,
    theta_max,
    lge_min=lge_bins.min(),
    lge_max=lge_bins.max(),
    module_size=module_size,
    core_in_uc=core_in_uc,
    use_energy_icrc2025=use_energy_icrc2025)

save_figure(figs_k_nmu["fig"], 
               (out_dir + f"nmu_corr_vs_uncorr_th{theta_min:.0f}-{theta_max:.0f}_"
               f"{module_size}_coreInUC{int(core_in_uc)}.png"))

csv_name = (out_dir + f"suspect_overcount_pmt_th{theta_min:.0f}-{theta_max:.0f}_"
               f"{module_size}_coreInUC{int(core_in_uc)}.csv")
df_suspect = figs_k_nmu["df_suspect_overcounting"]

df_suspect.to_csv(csv_name, index=False)
print(f"{csv_name} generated")


"""
theta_min = 0 
theta_max = 25 

lge_min = 17.5 
lge_max = 17.8

lge_min = 17.5 
lge_max = 18.3



fig_mean_ldf = plot_mean_ldf(df_pmt=df_pmt, df_sipm=df_sipm, 
                             theta_min=theta_min, theta_max=theta_max, 
                             lge_min=lge_min, lge_max=lge_max,
                             module_size="large")

save_figure(fig_mean_ldf["fig"], "compare_sipm_pmt_mean_ldf.png")
save_figure(fig_mean_ldf["fig2"], "compare_sipm_pmt_mean_ldf2.png")
save_figure(fig_mean_ldf["fig3"], "compare_sipm_pmt_mean_ldf3.png")
save_figure(fig_mean_ldf["fig4"], "compare_sipm_pmt_mean_ldf4.png")
save_figure(fig_mean_ldf["fig5"], "compare_sipm_pmt_mean_ldf5.png")
save_figure(fig_mean_ldf["fig6"], "compare_sipm_pmt_mean_ldf6.png")
save_figure(fig_mean_ldf["fig7"], "compare_sipm_pmt_mean_ldf7.png")
"""

print("")
print("Time series for each module ... ")
## Time series 
rmin, rmax = 200, 800
#theta_min = 0 
#theta_max = 25 
lge_min, lge_max = 17.5, 17.8

plot_modules_time_series(df_pmt=df_pmt, df_sipm=df_sipm, 
                             theta_min=theta_min, theta_max=theta_max, 
                             lge_min=lge_min, lge_max=lge_max, 
                             rmin=rmin, rmax=rmax,
                             out_dir=out_dir)



print("")
print("Relative efficiency for each module as a funciton of SD signal ... ")
plot_relative_efficiency_per_module(df_pmt=df_pmt, df_sipm=df_sipm, theta_min=theta_min, theta_max=theta_max, 
                                lge_min=lge_min, out_dir=out_dir)
