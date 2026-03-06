import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib.colors import LogNorm
from scipy.stats import ks_2samp, cramervonmises_2samp
from scipy.optimize import minimize_scalar
from astropy.time import Time
import ROOT

# Add custom module paths
sys.path.append('/storage/gpfs_data/pauger/users/jdejesus/my_python_modules/')

from myClasses import Profile
from myUsefulFunctions import generate_colormap, compute_weighted_mean, standard_error_of_weighted_mean


sys.path.append('/storage/gpfs_data/pauger/users/jdejesus/my_python_modules/auger')
from augerADSTgeneral import extract_sd_info
from augerDetector import get_sd_tank_coordinates

# ================================================
# Analysis of event-level quantities
# ================================================



def extract_energy_icrc_2017(
    events_id,
    adst_path="./adst_sd_umd_icrc-2017/",
    adst_files=("adst_sd_umd.root", "adst_sd_umd2.root"),
):
    """
    Extract SD energy (log10(E/eV)) for PMT events using the ICRC 2017 Observer production.

    This function maps PMT event identifiers (SD+MD) to their corresponding SD
    event IDs and retrieves the SD energy estimate from the ICRC 2017 ADST files.
    This is presumably the energy estimate used in the PMT paper.

    NOTE
    ----
    After testing, using the ICRC 2017 energy does not change the PMT results
    significantly.

    Parameters
    ----------
    events_id : array-like of str
        PMT event identifiers of the form:
        'auger_162670441700__sd_39643837__md_228'.

    adst_path : str, optional
        Path to the directory containing the ICRC 2017 ADST files.

    adst_files : tuple of str, optional
        Names of the ADST files to be scanned.

    Returns
    -------
    dict
        Dictionary with keys:
            - 'event_id': PMT event IDs (SD+MD)
            - 'lge_icrc_2017': SD log10(E/eV) from ICRC 2017 reconstruction

    Requirements
    ------------
    - Auger Offline environment configured
    - AUGEROFFLINEROOT environment variable set
    - extract_sd_info(event) available in the namespace
    """

    # ------------------------------------------------------------------
    # Build SD -> MD event ID mapping
    # ------------------------------------------------------------------
    sd_to_md = {}
    for evt in events_id:
        sd_id = evt.split("__md")[0]
        sd_to_md[sd_id] = evt

    sd_event_ids = set(sd_to_md.keys())

    # ------------------------------------------------------------------
    # Check environment and load Offline libraries
    # ------------------------------------------------------------------
    if "AUGEROFFLINEROOT" not in os.environ:
        raise RuntimeError("AUGEROFFLINEROOT environment variable is not set")

    auger_offline_root = os.environ["AUGEROFFLINEROOT"]
    ROOT.gSystem.Load(os.path.join(auger_offline_root, "lib/libRecEventKG.so"))

    # ------------------------------------------------------------------
    # Check input files
    # ------------------------------------------------------------------
    adst_file_paths = []
    for fname in adst_files:
        full_path = os.path.join(adst_path, fname)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"ADST file not found: {full_path}")
        adst_file_paths.append(full_path)

    # ------------------------------------------------------------------
    # Output container
    # ------------------------------------------------------------------
    outdict = {
        "event_id": [],
        "lge_icrc_2017": [],
    }

    # ------------------------------------------------------------------
    # Loop over ADST files and events
    # ------------------------------------------------------------------
    for fname in adst_file_paths:

        files = ROOT.std.vector("string")()
        files.push_back(fname)

        rec_file = ROOT.RecEventFile(files)
        event = ROOT.RecEvent()
        geo = ROOT.DetectorGeometry()

        rec_file.ReadDetectorGeometry(geo)
        rec_file.SetBuffers(event)

        while rec_file.ReadNextEvent() == ROOT.RecEventFile.eSuccess:

            sd_event_id = str(event.GetEventId())

            # Keep only requested SD events
            if sd_event_id not in sd_event_ids:
                continue

            # Avoid duplicates if event appears in multiple files
            md_event_id = sd_to_md[sd_event_id]
            if md_event_id in outdict["event_id"]:
                continue

            sd_info = extract_sd_info(event)

            outdict["event_id"].append(md_event_id)
            outdict["lge_icrc_2017"].append(sd_info["lgE"])

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"n_input_events_id           = {len(events_id)}")
    print(f"n_events_found_icrc_2017    = {len(outdict['event_id'])}")
    print(f"n_unique_events_icrc_2017  = {len(set(outdict['event_id']))}")

    return outdict

def _reduce_to_event_level(df):
    """
    Helper function for plot_event_level_distributions()

    Reduce a module-level DataFrame to event-level quantities.

    Assumes that all event-level quantities are constant within an event.
    """
    grouped = df.groupby("event_id")

    out = grouped.agg(
        lge=("lge", "first"),
        theta=("theta", "first"),
        phi=("phi", "first"),
        s450=("sd_shower_size", "first"),
        s35=("sd_shower_size_att", "first"),
        xc=("xc", "first"),
        yc=("yc", "first")
    )

    return out.reset_index()

def sync_lge_from_2025(
    df,
    df_2025,
    event_col="event_id",
    lge_df="lge",
    lge_2025="lge_icrc_2025",
    verbose=False
):
    """
    Keep only events in df (dataframe with module-level measurements) that are 
    present in df_2025 (dataframe containing the energy estimate of the ICRC2025) (preserving duplicates)
    and replace the lge column in df using values from df_2025.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with possibly duplicated event_id (ie. allows for module-level dataframes)
    df_2025 : pandas.DataFrame
        DataFrame with unique event_id and corresponding lge from the ICRC 2025
        Observer production
    """

    # --- rename for consistency
    df_2025 = df_2025.rename(columns={lge_2025: lge_df})

    # --- diagnostics
    events_df = set(df[event_col])
    events_2025 = set(df_2025[event_col])

    only_df = events_df - events_2025
    only_2025 = events_2025 - events_df
    common = events_df & events_2025

    if verbose:
        print(f"Events only in df:       {len(only_df)}")
        print(f"Events only in df_2025:  {len(only_2025)}")
        print(f"Events in both:          {len(common)}")

    n_before = len(df)

    # --- keep only common events (duplicates preserved)
    df = df[df[event_col].isin(events_2025)].copy()

    if verbose:
        print(f"Rows in df before filter: {n_before}")
        print(f"Rows in df after filter:  {len(df)}")

    # --- replace lge via mapping
    lge_map = df_2025.set_index(event_col)[lge_df]
    df[lge_df] = df[event_col].map(lge_map)

    return df



def plot_event_distributions(df_pmt, df_sipm, 
                        lge_bins=np.array([17.3, 17.5, 17.7, 17.9, 18.1, 18.3]),
                        theta_max=45,
                        core_in_uc=False,
                        use_energy_icrc2025=True):
    """
    Compare PMT and SiPM datasets at event level.
    Plot event-level distributions and cross-checks for PMT and SiPM datasets.

    Produces:
      - Energy distributions (raw and ICRC 2017)
      - Energy calibration checks
      - Attenuation factor comparisons
      - KS tests and ECDFs
      - Energy-scale shift scan

    Returns
    -------
    dict
        Dictionary containing matplotlib figures and axes.
    """
    

    # ------------------------------------------------------------
    # Cuts
    # ------------------------------------------------------------
    
    lge_min =  lge_bins.min()
    lge_max =  lge_bins.max()

    
    df_pmt = df_pmt.query("theta < @theta_max and lge > @lge_min and lge < @lge_max")
    df_sipm = df_sipm.query("theta < @theta_max and lge > @lge_min and lge < @lge_max")

    xml_path = ("/storage/gpfs_data/pauger/users/jdejesus/auger_software/"
                    "offline/src/Framework/SDetector/SStationList.xml.in")
    df_tanks = get_sd_tank_coordinates(xml_path)

    if core_in_uc:
       
        df_indexed = df_tanks.set_index("station_id")

        # Do a square centered in the unitary cell
        # easting limits
        xleft = df_indexed.loc[1574, "easting"]
        xright = df_indexed.loc[1764, "easting"]

        # northing limits
        ybottom = df_indexed.loc[1773, "northing"]
        yup = df_indexed.loc[1570, "northing"]

        mask_pmt_core_uc = ((xleft < df_pmt["xc"]) & (df_pmt["xc"] < xright) & 
                            (ybottom < df_pmt["yc"]) & (df_pmt["yc"] < yup) )
        
        mask_sipm_core_uc = ((xleft < df_sipm["xc"]) & (df_sipm["xc"] < xright) & 
                            (ybottom < df_sipm["yc"]) & (df_sipm["yc"] < yup) )
        
        df_pmt = df_pmt[mask_pmt_core_uc]
        df_sipm = df_sipm[mask_sipm_core_uc]

    if use_energy_icrc2025:
        fname_pmt_icrc25 = ("/storage/gpfs_data/pauger/users/jdejesus/analysis/"
                            "compare_pmt_sipm/src/get_energy_icrc2025/energy_pmt_icrc2025.csv")
        df_pmt_icrc25 = pd.read_csv(fname_pmt_icrc25)

        fname_sipm_icrc25 = ("/storage/gpfs_data/pauger/users/jdejesus/analysis/"
                            "compare_pmt_sipm/src/get_energy_icrc2025/energy_sipm_icrc2025.csv")
        df_sipm_icrc25 = pd.read_csv(fname_sipm_icrc25)
        
        # Overwriting the lge column using the energies of the ICRC2025
        df_pmt = sync_lge_from_2025(df=df_pmt, df_2025=df_pmt_icrc25)
        df_sipm = sync_lge_from_2025(df=df_sipm, df_2025=df_sipm_icrc25)



    # ------------------------------------------------------------
    # Reduce to event level
    # ------------------------------------------------------------

    evt_pmt = _reduce_to_event_level(df_pmt)
    evt_sipm = _reduce_to_event_level(df_sipm)

    # ------------------------------------------------------------
    # ICRC 2017 energy for PMT
    # ------------------------------------------------------------
    icrc = extract_energy_icrc_2017(evt_pmt["event_id"].to_numpy())
    icrc_map = dict(zip(icrc["event_id"], icrc["lge_icrc_2017"]))

    evt_pmt["lge_icrc_2017"] = evt_pmt["event_id"].map(icrc_map)
    evt_pmt = evt_pmt.dropna(subset=["lge_icrc_2017"])


    # ------------------------------------------------------------
    # Numpy arrays
    # ------------------------------------------------------------
    lge_pmt = evt_pmt["lge"].to_numpy()
    lge_pmt_icrc_2017 = evt_pmt["lge_icrc_2017"].to_numpy()
    theta_pmt = evt_pmt["theta"].to_numpy()
    s450_pmt = evt_pmt["s450"].to_numpy()
    s35_pmt = evt_pmt["s35"].to_numpy()
    xc_pmt = evt_pmt["xc"].to_numpy()
    yc_pmt = evt_pmt["yc"].to_numpy()

    lge_sipm = evt_sipm["lge"].to_numpy()
    theta_sipm = evt_sipm["theta"].to_numpy()
    s450_sipm = evt_sipm["s450"].to_numpy()
    s35_sipm = evt_sipm["s35"].to_numpy()
    xc_sipm = evt_sipm["xc"].to_numpy()
    yc_sipm = evt_sipm["yc"].to_numpy()

    fatt_pmt = s450_pmt / s35_pmt
    fatt_sipm = s450_sipm / s35_sipm

    # ------------------------------------------------------------
    # Binning
    # ------------------------------------------------------------
    
    lge_centers = 0.5 * (lge_bins[:-1] + lge_bins[1:])
    theta_bins = np.linspace(0, theta_max, 12)

    mask_pmt = (lge_bins.min() < lge_pmt) & (lge_pmt < lge_bins.max())
    mask_pmt_icrc_2017 = (lge_bins.min() < lge_pmt_icrc_2017) & (lge_pmt_icrc_2017 < lge_bins.max())
    mask_sipm = (lge_bins.min() < lge_sipm) & (lge_sipm < lge_bins.max())

    color_pmt = "b"
    color_sipm = "g"

    title_cuts = (f"theta < {theta_max:.2f}, {lge_bins.min():.1f} < lge < {lge_bins.max():.1f} ,"
                  f" core_in_uc = {core_in_uc}, use_energy_icrc2025 = {use_energy_icrc2025}")
    # ------------------------------------------------------------
    # FIGURE 1 — Energy distributions + calibration
    # ------------------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(18, 14))
    ax = ax.flatten()

    counts_pmt, _ = np.histogram(lge_pmt, bins=lge_bins)
    counts_sipm, _ = np.histogram(lge_sipm, bins=lge_bins)

    ax[0].hist(lge_pmt, bins=lge_bins, histtype="step", lw=2,
               label=f"PMT ({mask_pmt.sum()} events)", color=color_pmt)
    ax[0].hist(lge_pmt_icrc_2017, bins=lge_bins, histtype="step", lw=2,
               label=f"PMT ICRC 2017 ({mask_pmt_icrc_2017.sum()} events)", ls="--", color="grey")
    ax[0].hist(lge_sipm, bins=lge_bins, histtype="step", lw=2,
               label=f"SiPM ({mask_sipm.sum()} events)", color=color_sipm)
    ax[0].set_yscale("log")
    ax[0].set_xlabel(r"$\lg(E/\mathrm{eV})$")
    ax[0].legend()

    valid = (counts_pmt > 0) & (counts_sipm > 0)
    scale = counts_sipm[valid][1] / counts_pmt[valid][1]
    counts_pmt_scaled = counts_pmt * scale
    counts_pmt_scaled_err = np.sqrt(counts_pmt) * scale

    ax[1].errorbar(lge_centers, counts_pmt_scaled,
                   yerr=counts_pmt_scaled_err,
                   fmt="o", mfc="None", label="PMT (scaled)",
                   color=color_pmt)
    ax[1].errorbar(lge_centers, counts_sipm,
                   yerr=np.sqrt(counts_sipm),
                   fmt="s", alpha=0.6, label="SiPM", color=color_sipm)
    ax[1].set_yscale("log")
    ax[1].legend()

    ax[1].set_xlabel(r"$\lg(E/\mathrm{eV})$")
    ax[1].set_ylabel("Counts")

    ratio = counts_pmt_scaled / counts_sipm
    ratio_err = ratio * np.sqrt(
        (counts_pmt_scaled_err / counts_pmt) ** 2 +
        (np.sqrt(counts_sipm) / counts_sipm) ** 2
    )

    ax[2].errorbar(lge_centers, ratio, yerr=ratio_err, fmt="o", mfc="None")
    ax[2].axhline(1, ls="--", color="grey")
    ax[2].set_xlabel(r"$\lg(E/\mathrm{eV})$")
    ax[2].set_ylabel("nr of events PMT (scaled) /  nr of events SiPM")

    ax[3].errorbar(s35_pmt, lge_pmt, fmt="o", mfc="None", label="PMT", color=color_pmt)
    ax[3].errorbar(s35_sipm, lge_sipm, fmt="s", alpha=0.6, label="SiPM", color=color_sipm)
    ax[3].set_xscale("log")
    ax[3].set_xlabel(r"$S_{35}$")
    ax[3].set_ylabel(r"$\lg(E/\mathrm{eV})$")
    ax[3].legend()

    fig.suptitle(title_cuts)

    fig.tight_layout()

    # ------------------------------------------------------------
    # FIGURE 2 — f_att, S450 vs S35, theta
    # ------------------------------------------------------------
    fig2, ax2 = plt.subplots(2, 2, figsize=(18, 14))
    ax2 = ax2.flatten()

    ax2[0].errorbar(theta_pmt[mask_pmt], fatt_pmt[mask_pmt],
                    fmt="o", mfc="None", label="PMT", color=color_pmt)
    ax2[0].errorbar(theta_sipm[mask_sipm], fatt_sipm[mask_sipm],
                    fmt="s", alpha=0.6, label="SiPM", color=color_sipm)
    ax2[0].axvline(35, ls="--", color="grey")
    ax2[0].axhline(1, ls="--", color="grey")
    ax2[0].set_xlabel(r"$\theta$")
    ax2[0].set_ylabel(r"$f_{\mathrm{att,SD}}$")
    ax2[0].legend()

    ax2[1].errorbar(s450_pmt, s35_pmt, fmt="o", mfc="None", label="PMT", color=color_pmt)
    ax2[1].errorbar(s450_sipm, s35_sipm, fmt="s", alpha=0.6, label="SiPM", color=color_sipm)
    ax2[1].set_xscale("log")
    ax2[1].set_yscale("log")
    ax2[1].set_xlabel(r"$S(450)$")
    ax2[1].set_ylabel(r"$S_{35}$")
    ax2[1].legend()

    ax2[2].hist(theta_pmt[mask_pmt], bins=theta_bins,
                histtype="step", lw=2, density=True, label="PMT", color=color_pmt)
    ax2[2].hist(theta_sipm[mask_sipm], bins=theta_bins,
                histtype="step", lw=2, density=True, label="SiPM", color=color_sipm)
    ax2[2].set_xlabel(r"$\theta$")
    ax2[2].set_ylabel("pdf")
    ax2[2].legend()

    fig2.suptitle(title_cuts)
    fig2.tight_layout()

    # ------------------------------------------------------------
    # FIGURE 3 — PDFs + ECDF + KS, and core disdtributions
    # ------------------------------------------------------------
    fig3, ax3 = plt.subplots(2, 2, figsize=(18, 14))
    ax3 = ax3.flatten()

    ax3[0].hist(lge_pmt, bins=lge_bins, histtype="step", lw=2,
                density=True, label="PMT", color=color_pmt)
    ax3[0].hist(lge_pmt_icrc_2017, bins=lge_bins, histtype="step", lw=2,
                density=True, ls="--", label="PMT ICRC2017", color="grey")
    ax3[0].hist(lge_sipm, bins=lge_bins, histtype="step", lw=2,
                density=True, label="SiPM", color=color_sipm)
    ax3[0].set_xlabel(r"$\lg(E/\mathrm{eV})$")
    ax3[0].set_ylabel("pdf")
    ax3[0].legend()

    def ecdf(x):
        x = np.sort(x)
        return x, np.arange(1, len(x) + 1) / len(x)

    x_pmt, y_pmt = ecdf(lge_pmt)
    x_pmt_icrc_2017, y_pmt_icrc_2017 = ecdf(lge_pmt_icrc_2017)
    x_sipm, y_sipm = ecdf(lge_sipm)

    ks_lge = ks_2samp(lge_pmt, lge_sipm)
    ks_icrc_2017 = ks_2samp(lge_pmt_icrc_2017, lge_sipm)

    ax3[1].plot(x_pmt, y_pmt, label="PMT", color=color_pmt)
    ax3[1].plot(x_pmt_icrc_2017, y_pmt_icrc_2017, label="PMT ICRC2017", color="grey")
    ax3[1].plot(x_sipm, y_sipm, label="SiPM", color=color_sipm)
    ax3[1].set_xlabel(r"$\lg(E/\mathrm{eV})$")
    ax3[1].set_ylabel("cdf")
    ax3[1].set_title(
        f"KS p-value = {ks_lge.pvalue:.3f}; "
        f"ICRC2017 = {ks_icrc_2017.pvalue:.3f}"
    )
    ax3[1].legend()

    # cores
    ax3[2].errorbar(df_tanks["easting"], df_tanks["northing"], fmt="o", mfc="None",
                    color="k", ms=12)
    ax3[2].errorbar(xc_pmt, yc_pmt, fmt="o",alpha=0.6,
                    color=color_pmt, ms=4, label=f"{len(xc_pmt)} PMT events")
    
    ax3[2].set_xlabel("Easting / m")
    ax3[2].set_ylabel("Northing / m")
    ax3[2].legend()
    
    
    ax3[3].errorbar(df_tanks["easting"], df_tanks["northing"], fmt="o", mfc="None",
                    color="k", ms=12)
    ax3[3].errorbar(xc_sipm, yc_sipm, fmt="o",alpha=0.6,
                    color=color_sipm, ms=4, label=f"{len(xc_sipm)} SiPM events")
    ax3[3].set_xlabel("Easting / m")
    ax3[3].set_ylabel("Northing / m")
    ax3[3].legend()

    fig3.suptitle(title_cuts)
    fig3.tight_layout()

    # ------------------------------------------------------------
    # FIGURE 4 — Energy-scale scan
    # ------------------------------------------------------------
    shifts = np.arange(-0.2, 0.2, 0.01)
    p_scan = [ks_2samp(lge_pmt + s, lge_sipm).pvalue for s in shifts]

    fig4, ax4 = plt.subplots()
    ax4.plot(shifts, p_scan, color="k")
    ax4.axhline(0.05, ls="--", color="grey")
    ax4.set_yscale("log")
    ax4.set_xlabel("Energy shift")
    ax4.set_ylabel("KS p-value")
    ax4.set_title(f"max p-value = {np.max(p_scan):.3f}")

    fig4.suptitle(title_cuts)

    return {
        "fig": fig, "ax": ax,
        "fig2": fig2, "ax2": ax2,
        "fig3": fig3, "ax3": ax3,
        "fig4": fig4, "ax4": ax4,
    }

            
# ================================================
# Analysis of mean LDF using all modules at once
# ================================================


def fit_scaling_factor(y1, y2, y1_err, y2_err):
    """
    Fit a scaling factor k such that y1 ~ k * y2
    minimizing the chi^2:
    
    chi2(k) = sum_i (y1_i - k*y2_i)^2 / (y1_err_i^2 + (k*y2_err_i)^2)
    
    Returns best k, chi2, and uncertainty on k from curvature.
    
    """
    
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)
    y1_err = np.asarray(y1_err)
    y2_err = np.asarray(y2_err)

    def chi2(k):
        denom = y1_err**2 + (k * y2_err)**2
        return np.sum((y1 - k*y2)**2 / denom)

    # Search for k > 0 (usually physical)
    result = minimize_scalar(chi2, bounds=(0, 10), method='bounded')

    k_best = result.x
    chi2_best = result.fun

    # ---- Compute second derivative of chi2(k) at k_best ----
    def second_derivative(f, x, h=1e-5):
        # Central finite difference for f''(x)
        return (f(x + h) - 2*f(x) + f(x - h)) / (h*h)

    chi2_second = second_derivative(chi2, k_best)

    # Convert curvature to uncertainty
    sigma_k = np.sqrt(2.0 / chi2_second)

    return {
        "scaling_factor": k_best,
        "scaling_factor_err": sigma_k,
        "chi2_best": chi2_best,
        "result_minimize": result
    }

def plot_k_fractions(
    df_pmt,
    df_sipm,
    lge_min,
    lge_max,
    theta_min,
    theta_max,
    r_bin_edges,
    sd_signal_bin_edges,
    use_energy_icrc2025,
    k_values=(0,),
):
    """
    Plot the fraction of events with a given k value as a function of
    (1) core distance r and (2) WCD signal.

    Parameters
    ----------
    df_pmt, df_sipm : pandas.DataFrame
        DataFrames containing PMT and SiPM reconstructions.
        Required columns:
            - theta
            - lge
            - r
            - sd_signal
            - k_1x1 (PMT)
            - k (SiPM)

    lge_min, lge_max : float
        Log10(E/eV) selection range.

    theta_min, theta_max : float
        Zenith angle selection range (degrees).

    r_bin_edges : array-like
        Bin edges for core distance r (meters).

    sd_signal_bin_edges : array-like
        Bin edges for WCD signal (VEM).

    k_values : iterable of int, optional
        k values for which the fractions are computed.
        Default is (0,).

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """

    # ------------------------------------------------------------
    # Event selection
    # ------------------------------------------------------------
    if use_energy_icrc2025:
        fname_pmt_icrc25 = ("/storage/gpfs_data/pauger/users/jdejesus/analysis/"
                            "compare_pmt_sipm/src/get_energy_icrc2025/energy_pmt_icrc2025.csv")
        df_pmt_icrc25 = pd.read_csv(fname_pmt_icrc25)

        fname_sipm_icrc25 = ("/storage/gpfs_data/pauger/users/jdejesus/analysis/"
                            "compare_pmt_sipm/src/get_energy_icrc2025/energy_sipm_icrc2025.csv")
        df_sipm_icrc25 = pd.read_csv(fname_sipm_icrc25)
        
        # Overwriting the lge column using the energies of the ICRC2025
        df_pmt = sync_lge_from_2025(df=df_pmt, df_2025=df_pmt_icrc25)
        df_sipm = sync_lge_from_2025(df=df_sipm, df_2025=df_sipm_icrc25)

    sel = (
        (theta_min < df_pmt["theta"]) & (df_pmt["theta"] < theta_max) &
        (lge_min   < df_pmt["lge"])   & (df_pmt["lge"]   < lge_max)
    )
    df_pmt = df_pmt[sel]

    sel = (
        (theta_min < df_sipm["theta"]) & (df_sipm["theta"] < theta_max) &
        (lge_min   < df_sipm["lge"])   & (df_sipm["lge"]   < lge_max)
    )
    df_sipm = df_sipm[sel]

    # ------------------------------------------------------------
    # Bin centers
    # ------------------------------------------------------------
    r_bin_edges = np.asarray(r_bin_edges)
    r_centers = 0.5 * (r_bin_edges[:-1] + r_bin_edges[1:])

    sd_signal_bin_edges = np.asarray(sd_signal_bin_edges)
    sd_centers = 0.5 * (sd_signal_bin_edges[:-1] + sd_signal_bin_edges[1:])

    # ------------------------------------------------------------
    # Helper to compute fractions
    # ------------------------------------------------------------
    def compute_fraction(df, k_col, kval, mask):
        n_total = mask.sum()
        if n_total == 0:
            return 0.0, 0.0

        n_k = (df.loc[mask, k_col] == kval).sum()
        f = n_k / n_total
        ferr = np.sqrt(f * (1.0 - f) / n_total)  # binomial error
        return f, ferr

    # ------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------
    fractions = {
        k: {
            "r": {"pmt": [], "pmt_err": [], "sipm": [], "sipm_err": []},
            "sd": {"pmt": [], "pmt_err": [], "sipm": [], "sipm_err": []},
        }
        for k in k_values
    }

    # ------------------------------------------------------------
    # Fractions vs r
    # ------------------------------------------------------------
    for r1, r2 in zip(r_bin_edges[:-1], r_bin_edges[1:]):
        mask_pmt = (r1 < df_pmt["r"]) & (df_pmt["r"] < r2)
        mask_sipm = (r1 < df_sipm["r"]) & (df_sipm["r"] < r2)

        for k in k_values:
            f, e = compute_fraction(df_pmt, "k_1x1", k, mask_pmt)
            fractions[k]["r"]["pmt"].append(f)
            fractions[k]["r"]["pmt_err"].append(e)

            f, e = compute_fraction(df_sipm, "k", k, mask_sipm)
            fractions[k]["r"]["sipm"].append(f)
            fractions[k]["r"]["sipm_err"].append(e)

    # ------------------------------------------------------------
    # Fractions vs WCD signal
    # ------------------------------------------------------------
    for s1, s2 in zip(sd_signal_bin_edges[:-1], sd_signal_bin_edges[1:]):
        mask_pmt = (s1 < df_pmt["sd_signal"]) & (df_pmt["sd_signal"] < s2)
        mask_sipm = (s1 < df_sipm["sd_signal"]) & (df_sipm["sd_signal"] < s2)

        for k in k_values:
            f, e = compute_fraction(df_pmt, "k_1x1", k, mask_pmt)
            fractions[k]["sd"]["pmt"].append(f)
            fractions[k]["sd"]["pmt_err"].append(e)

            f, e = compute_fraction(df_sipm, "k", k, mask_sipm)
            fractions[k]["sd"]["sipm"].append(f)
            fractions[k]["sd"]["sipm_err"].append(e)

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(18, 11))

    markers = ["o", "s", "^", "v", "*"]

    for i, k in enumerate(k_values):
        ax[0].errorbar(
            r_centers,
            fractions[k]["r"]["pmt"],
            yerr=fractions[k]["r"]["pmt_err"],
            fmt=f"-{markers[i]}",
            mfc="none",
            label=f"PMT k={k}",
        )
        ax[0].errorbar(
            r_centers,
            fractions[k]["r"]["sipm"],
            yerr=fractions[k]["r"]["sipm_err"],
            fmt=f"--{markers[i]}",
            label=f"SiPM k={k}",
        )

        ax[1].errorbar(
            sd_centers,
            fractions[k]["sd"]["pmt"],
            yerr=fractions[k]["sd"]["pmt_err"],
            fmt=f"-{markers[i]}",
            mfc="none",
        )
        ax[1].errorbar(
            sd_centers,
            fractions[k]["sd"]["sipm"],
            yerr=fractions[k]["sd"]["sipm_err"],
            fmt=f"--{markers[i]}",
            label=f"SiPM k={k}",
        )

    ax[0].set(xlabel="r / m", ylabel="Fraction")
    ax[0].legend()

    ax[1].set(xlabel="WCD signal / VEM", ylabel="Fraction", xscale="log")

    return fig, ax


def plot_mean_ldf(
    df_pmt,
    df_sipm,
    theta_min,
    theta_max,
    lge_min,
    lge_max,
    module_size="all",
    core_in_uc=False,
    use_energy_icrc2025=False,
    scale_rho_mu=False
):
    """
    Produce comparison plots between PMT- and SiPM-era UMD module-level data.

    The function:
      - applies common cuts in zenith angle and energy,
      - optionally selects modules by active area,
      - profiles muon density (projected onto the shower plane) as a function of
        distance to the shower axis,
      - derives mean LDFs and their ratios,
      - computes and compares mean k-parameters as a function of distance and
        SD signal.

    Muon density is projected onto the shower plane as:
        rho_shower_plane = rho / cos(theta)

    Parameters
    ----------
    df_pmt, df_sipm : pandas.DataFrame
        Module-level PMT and SiPM datasets. Required columns include:
        ``theta``, ``lge``, ``r``, ``rho``, ``rho_shower_plane``,
        ``active_area``, ``k_1x1`` (PMT), ``k`` (SiPM), ``sd_signal``.

    theta_min, theta_max : float
        Zenith angle selection (degrees).

    lge_min, lge_max : float
        log10(E / eV) selection.

    module_size : {"all", "large", "small"}, optional
        Module selection by active area:
          - "all": no selection
          - "large": active_area >= 7 m^2
          - "small": active_area < 7 m^2


    pmt_method : {"paper", "inf_window_uncorr", "inf_window_corr"} str 
        Which muon estimator to use for the PMT dataset.
            - "paper": same muon estimation as in the paper, this is, with 
                time windows of 7 bins, f_clip(theta; dPhi) and eff(w=7)
            - "inf_window_uncorr": using infinite window uncorrected, ie., just 
                use -n*ln(1-k/n)
            - "inf_window_corr": infinte window corrected by efficiency, ie , eff(w=1024).
            which is 87% for 10 m2 and 95% for 5m2

    scale_rho_mu : bool
        If true, it will scale each module-level measurement by (E/10**18)**0.9

    Returns
    -------
    dict
        Dictionary with figures and axes:
        {
            "rho_vs_r": {"fig": fig1, "ax": ax1},
            "k_profiles": {"fig": fig2, "ax": ax2},
        }

    Raises
    ------
    ValueError
        If ``module_size`` is invalid.
    """

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if module_size not in {"all", "small", "large"}:
        raise ValueError("module_size must be 'all', 'small', or 'large'")

    # TODO IMPLEMENTAR!
    #if pmt_method not in {"paper", "inf_window_uncorr", "inf_window_corr"}:
        #raise ValueError("pmt_method must be 'paper', 'inf_window_uncorr' or 'inf_window_corr'")

    # ------------------------------------------------------------------
    # Common cuts
    # ------------------------------------------------------------------
    if use_energy_icrc2025:
        fname_pmt_icrc25 = ("/storage/gpfs_data/pauger/users/jdejesus/analysis/"
                            "compare_pmt_sipm/src/get_energy_icrc2025/energy_pmt_icrc2025.csv")
        df_pmt_icrc25 = pd.read_csv(fname_pmt_icrc25)

        fname_sipm_icrc25 = ("/storage/gpfs_data/pauger/users/jdejesus/analysis/"
                            "compare_pmt_sipm/src/get_energy_icrc2025/energy_sipm_icrc2025.csv")
        df_sipm_icrc25 = pd.read_csv(fname_sipm_icrc25)
        
        # Overwriting the lge column using the energies of the ICRC2025
        df_pmt = sync_lge_from_2025(df=df_pmt, df_2025=df_pmt_icrc25)
        df_sipm = sync_lge_from_2025(df=df_sipm, df_2025=df_sipm_icrc25)

    def apply_cuts(df):
        mask = (
            (theta_min < df["theta"])
            & (df["theta"] < theta_max)
            & (lge_min < df["lge"])
            & (df["lge"] < lge_max)
        )
        return df.loc[mask].copy()

    df_pmt_cut = apply_cuts(df_pmt)
    df_sipm_cut = apply_cuts(df_sipm)

    if module_size == "large":
        df_pmt_cut = df_pmt_cut[df_pmt_cut["active_area"] >= 7]
        df_sipm_cut = df_sipm_cut[df_sipm_cut["active_area"] >= 7]
    elif module_size == "small":
        df_pmt_cut = df_pmt_cut[df_pmt_cut["active_area"] < 7]
        df_sipm_cut = df_sipm_cut[df_sipm_cut["active_area"] < 7]

    
    if core_in_uc:
        xml_path = ("/storage/gpfs_data/pauger/users/jdejesus/auger_software/"
                    "offline/src/Framework/SDetector/SStationList.xml.in")
        df_tanks = get_sd_tank_coordinates(xml_path)
        
        df_indexed = df_tanks.set_index("station_id")

        # Do a square centered in the unitary cell
        # easting limits
        xleft = df_indexed.loc[1574, "easting"]
        xright = df_indexed.loc[1764, "easting"]

        # northing limits
        ybottom = df_indexed.loc[1773, "northing"]
        yup = df_indexed.loc[1570, "northing"]

        mask_pmt_core_uc = ((xleft < df_pmt_cut["xc"]) & (df_pmt_cut["xc"] < xright) & 
                            (ybottom < df_pmt_cut["yc"]) & (df_pmt_cut["yc"] < yup) )
        
        mask_sipm_core_uc = ((xleft < df_sipm_cut["xc"]) & (df_sipm_cut["xc"] < xright) & 
                            (ybottom < df_sipm_cut["yc"]) & (df_sipm_cut["yc"] < yup) )
        
        df_pmt_cut = df_pmt_cut[mask_pmt_core_uc]
        df_sipm_cut = df_sipm_cut[mask_sipm_core_uc]


    title_cuts = (
        f"{theta_min:.0f}° < theta < {theta_max:.0f}°, "
        f"{lge_min:.1f} < lgE < {lge_max:.1f}, "
        f"module_size = {module_size}, core_in_uc = {core_in_uc}, use_energy_icrc2025 = {use_energy_icrc2025}, scale_rho_mu = {scale_rho_mu}"
    )

    # ------------------------------------------------------------------
    # Physics definitions and checks
    # ------------------------------------------------------------------
    r_bin_edges = np.arange(200, 801, 100)

    rho_pmt_sp = df_pmt_cut["rho"].to_numpy() / np.cos(
        np.deg2rad(df_pmt_cut["theta"].to_numpy())
    )
    rho_sipm_sp = df_sipm_cut["rho"].to_numpy() / np.cos(
        np.deg2rad(df_sipm_cut["theta"].to_numpy())
    )

    # Sanity check against stored column
    assert np.allclose(
        rho_pmt_sp, df_pmt_cut["rho_shower_plane"]
    ), "PMT rho_shower_plane inconsistency"
    assert np.allclose(
        rho_sipm_sp, df_sipm_cut["rho_shower_plane"]
    ), "SiPM rho_shower_plane inconsistency"

    if scale_rho_mu:
        scale_energy_pmt = 10**((df_pmt_cut["lge"]-18)*0.9)
        rho_pmt_sp = rho_pmt_sp / scale_energy_pmt

        scale_energy_sipm = 10**((df_sipm_cut["lge"]-18)*0.9)
        rho_sipm_sp = rho_sipm_sp / scale_energy_sipm

    # Profiles rho vs r
    p_pmt = Profile(x=df_pmt_cut["r"], y=rho_pmt_sp).doProfile(r_bin_edges)
    p_sipm = Profile(x=df_sipm_cut["r"], y=rho_sipm_sp).doProfile(r_bin_edges)
    
    # -------------------------------------------------
    # Alternative estimation of rho_mu with PMT using inf window
    # 1st: pass to shower plane
    rho_inf_window_pmt_1x1_shower_plane = df_pmt_cut["rho_inf_uncorrected_1x1"].to_numpy() / np.cos(np.deg2rad(df_pmt_cut["theta"].to_numpy()))

    p_pmt_alt_uncorr = Profile(df_pmt_cut["r"], rho_inf_window_pmt_1x1_shower_plane).doProfile(r_bin_edges)

    # 2nd: correct by efficiency 
    eff = 0.87 if module_size == "large" else 0.94
    rho_inf_window_pmt_1x1_shower_plane /= eff
    # ToDo: should I correct by CornerClipping?

    p_pmt_alt = Profile(df_pmt_cut["r"], rho_inf_window_pmt_1x1_shower_plane).doProfile(r_bin_edges)

    # ------------------------------------------------------------------
    # FIGURE 1 — rho_mu vs r
    # ------------------------------------------------------------------
    color_pmt = "b"
    color_sipm = "g"

    fig1, ax1 = plt.subplots(2, 2, figsize=(18, 14))
    ax1 = ax1.flatten()

    # Unscaled
    ax1[0].errorbar(
        p_pmt.x_center,
        p_pmt.y_mean,
        yerr=p_pmt.y_std_of_mean,
        fmt="o",
        ms=10,
        color=color_pmt,
        label="PMT",
    )

    ax1[0].errorbar(
        p_pmt_alt.x_center,
        p_pmt_alt.y_mean,
        yerr=p_pmt_alt.y_std_of_mean,
        fmt="^",
        ms=10,
        color="r",
        mfc="None",
        label=f"PMT w/ inf window (corrected by eff={eff:.2f})",
    )

    ax1[0].errorbar(
        p_pmt_alt_uncorr.x_center,
        p_pmt_alt_uncorr.y_mean,
        yerr=p_pmt_alt_uncorr.y_std_of_mean,
        fmt="v",
        ms=10,
        color="m",
        mfc="None",
        label=f"PMT w/ inf window (uncorrected)",
    )

    
    ax1[0].errorbar(
        p_sipm.x_center,
        p_sipm.y_mean,
        yerr=p_sipm.y_std_of_mean,
        fmt="s",
        ms=10,
        color=color_sipm,
        label="SiPM",
    )

    ylabel = r"$\rho_\mu\ / (E / 10^{18} \text{eV})^{0.9}$" if scale_rho_mu else r"$\rho_\mu\ /\ \mathrm{m}^{-2}$"
    ax1[0].set(xscale="log", yscale="log", xlabel="r / m",
               ylabel=ylabel)
    ax1[0].legend()

    # Scaling
    # Scaling
    valid_mask_ldf = (np.array(p_pmt.y_mean) > 0) & (np.array(p_sipm.y_mean) > 0)
    if valid_mask_ldf.sum() != len(p_sipm.y_mean):
        print((f" the rho_mu vs r profile had some bins "
                f"with invalid entries (lge_min={lge_min:.1f}, lgemax={lge_max:.1f}, theta_min={theta_min:.0f}, theta_max={theta_max:.0f})"))
    fit = fit_scaling_factor(
        y1=np.array(p_sipm.y_mean)[valid_mask_ldf],
        y2=np.array(p_pmt.y_mean)[valid_mask_ldf],
        y1_err=np.array(p_sipm.y_std_of_mean)[valid_mask_ldf],
        y2_err=np.array(p_pmt.y_std_of_mean)[valid_mask_ldf],
    )
    scale = fit["scaling_factor"]

    rho_pmt_scaled = scale * np.array(p_pmt.y_mean)
    rho_pmt_scaled_err = scale * np.array(p_pmt.y_std_of_mean)

    # Scaling using PMT w/inf window (corrected by efficiency)- suffix 'alt' for alternative
    valid_mask_ldf_alt = (np.array(p_pmt_alt.y_mean) > 0) & (np.array(p_sipm.y_mean) > 0)
    if valid_mask_ldf_alt.sum() != len(p_sipm.y_mean):
        print((f" the rho_mu vs r profile had some bins "
                f"with invalid entries (lge_min={lge_min:.1f}, lgemax={lge_max:.1f}, theta_min={theta_min:.0f}, theta_max={theta_max:.0f})"))
    fit_alt = fit_scaling_factor(
        y1=np.array(p_sipm.y_mean)[valid_mask_ldf_alt],
        y2=np.array(p_pmt_alt.y_mean)[valid_mask_ldf_alt],
        y1_err=np.array(p_sipm.y_std_of_mean)[valid_mask_ldf_alt],
        y2_err=np.array(p_pmt_alt.y_std_of_mean)[valid_mask_ldf_alt],
    )
    scale_alt = fit_alt["scaling_factor"]

    rho_pmt_scaled_alt = scale_alt * np.array(p_pmt_alt.y_mean)
    rho_pmt_scaled_alt_err = scale_alt * np.array(p_pmt_alt.y_std_of_mean)

    
    
    ax1[1].errorbar(
        p_pmt.x_center,
        rho_pmt_scaled,
        yerr=rho_pmt_scaled_err,
        fmt="o",
        ms=10,
        color=color_pmt,
        mfc="None",
        label=f"PMT x {scale:.3f}",
    )

    ax1[1].errorbar(
        p_pmt.x_center,
        rho_pmt_scaled_alt,
        yerr=rho_pmt_scaled_alt_err,
        fmt="^",
        ms=10,
        color="r",
        mfc="None",
        label=f"PMT (inf window + correction eff) x {scale_alt:.3f}",
    )

    ax1[1].errorbar(
        p_sipm.x_center,
        p_sipm.y_mean,
        yerr=p_sipm.y_std_of_mean,
        fmt="s",
        ms=10,
        color=color_sipm,
        label="SiPM",
    )
    ax1[1].set(xscale="log", yscale="log", xlabel="r / m",
               ylabel=ylabel)
    ax1[1].legend()

    # Ratios
    with np.errstate(divide="ignore", invalid="ignore"):
        rho_sipm = np.array(p_sipm.y_mean)
        rho_sipm_err = np.array(p_sipm.y_std_of_mean)
        
        rho_pmt = np.array(p_pmt.y_mean)
        rho_pmt_err = np.array(p_pmt.y_std_of_mean)

        ratio = rho_pmt / rho_sipm 
        ratio_err = ratio * np.sqrt( (rho_pmt_err / rho_pmt)**2 + (rho_sipm_err / rho_sipm)**2 )
        
        ratio_scaled = rho_pmt_scaled / rho_sipm 
        ratio_scaled_err = ratio_scaled * np.sqrt( (rho_pmt_scaled_err / rho_pmt_scaled)**2 + (rho_sipm_err / rho_sipm)**2 )  

        # alternative PMT calculation with infinite window
        rho_pmt_alt = np.array(p_pmt_alt.y_mean)
        rho_pmt_alt_err = np.array(p_pmt_alt.y_std_of_mean)

        ratio_alt = rho_pmt_alt / rho_sipm 
        ratio_alt_err = ratio_alt * np.sqrt( (rho_pmt_alt_err / rho_pmt_alt)**2 + (rho_sipm_err / rho_sipm)**2 )

        ratio_scaled_alt = rho_pmt_scaled_alt / rho_sipm 
        ratio_scaled_alt_err = ratio_scaled_alt * np.sqrt( (rho_pmt_scaled_alt_err / rho_pmt_scaled_alt)**2 + (rho_sipm_err / rho_sipm)**2 )  


    wmean_ratio = compute_weighted_mean(ratio, ratio_err)
    wmean_ratio_err = standard_error_of_weighted_mean(ratio, ratio_err)

    ax1[2].errorbar(p_sipm.x_center, ratio, yerr=ratio_err, fmt="o", color="k")
    ax1[2].errorbar(p_sipm.x_center, ratio_alt, yerr=ratio_alt_err, fmt="^", color="r")
    ax1[2].axhline(wmean_ratio, ls="-.", color="k", label=f"mean = {wmean_ratio:.2f} +/- {wmean_ratio_err:.2f}")
    ax1[2].axhline(1, ls="--", color="grey")
    ylabel2 = r"$\left( \rho_\mathrm{PMT} / (E_\text{PMT} / 10^{18} \text{eV})^{0.9} \right) / \left( \rho_\mathrm{SiPM} / (E_\text{PMT} / 10^{18} \text{eV})^{0.9} \right)$" if scale_rho_mu else r"$\rho_\mathrm{PMT} /  \rho_\mathrm{SiPM}$"
    ax1[2].set(xlabel="r / m",
               ylabel=ylabel2)
    ax1[2].legend()

    ax1[3].errorbar(
        p_sipm.x_center, ratio_scaled, yerr=ratio_scaled_err, fmt="o", color="k"
    )
    ax1[3].errorbar(
        p_sipm.x_center, ratio_scaled_alt, yerr=ratio_scaled_alt_err, fmt="^", color="r"
    )
    ax1[3].axhline(1, ls="--", color="grey")
    ax1[3].set(
        xlabel="r / m",
        ylabel=r"$\rho_\mathrm{PMT}^{\mathrm{scaled}} / \rho_\mathrm{SiPM}$",
    )

    fig1.suptitle(title_cuts)
    fig1.tight_layout()

    # ------------------------------------------------------------------
    # FIGURE 2 — k profiles
    # ------------------------------------------------------------------
    fig2, ax2 = plt.subplots(2, 2, figsize=(18, 14))
    ax2 = ax2.flatten()

    k_pmt = Profile(df_pmt_cut["r"], df_pmt_cut["k_1x1"]).doProfile(r_bin_edges)
    k_sipm = Profile(df_sipm_cut["r"], df_sipm_cut["k"]).doProfile(r_bin_edges)

    ax2[0].errorbar(
        k_pmt.x_center, k_pmt.y_mean, yerr=k_pmt.y_std_of_mean,
        fmt="o", ms=10, color=color_pmt, label="PMT"
    )
    ax2[0].errorbar(
        k_sipm.x_center, k_sipm.y_mean, yerr=k_sipm.y_std_of_mean,
        fmt="s", ms=10, color=color_sipm, label="SiPM"
    )
    ax2[0].set(xscale="log", xlabel="r / m", ylabel=r"$k$")
    ax2[0].legend()


    k_pmt_mean = np.array(k_pmt.y_mean)
    k_pmt_mean_err = np.array(k_pmt.y_std_of_mean)

    k_sipm_mean = np.array(k_sipm.y_mean)
    k_sipm_mean_err = np.array(k_sipm.y_std_of_mean)

    ratio_k =  k_pmt_mean / k_sipm_mean
    ratio_k_err = ratio_k * np.sqrt( (k_sipm_mean_err / k_sipm_mean)**2 + 
                                     (k_pmt_mean_err / k_pmt_mean)**2 )

    ax2[1].errorbar(k_sipm.x_center, ratio_k, yerr=ratio_k_err, fmt="o", color="k")
    ax2[1].axhline(1, ls="--", color="grey")
    ax2[1].set(xlabel="r / m",
               ylabel=r"$\langle k\rangle_\mathrm{PMT} / \langle k\rangle_\mathrm{SiPM}$")

    # k vs SD signal
    sd_bins = 10**np.linspace(0.9, 2.2, 11)
    k_sd_pmt = Profile(df_pmt_cut["sd_signal"], df_pmt_cut["k_1x1"]).doProfile(sd_bins)
    k_sd_sipm = Profile(df_sipm_cut["sd_signal"], df_sipm_cut["k"]).doProfile(sd_bins)

    ax2[2].errorbar(
        k_sd_pmt.x_center, k_sd_pmt.y_mean, yerr=k_sd_pmt.y_std_of_mean,
        fmt="o", ms=10, color=color_pmt, label="PMT"
    )
    ax2[2].errorbar(
        k_sd_sipm.x_center, k_sd_sipm.y_mean, yerr=k_sd_sipm.y_std_of_mean,
        fmt="s", ms=10, color=color_sipm, label="SiPM"
    )
    ax2[2].set(xscale="log", xlabel="WCD signal / VEM", ylabel=r"$k$")
    ax2[2].legend()

    k_sd_pmt_mean = np.array(k_sd_pmt.y_mean)
    k_sd_pmt_mean_err = np.array(k_sd_pmt.y_std_of_mean)

    k_sd_sipm_mean = np.array(k_sd_sipm.y_mean)
    k_sd_sipm_mean_err = np.array(k_sd_sipm.y_std_of_mean)
    ratio_k_sd =  k_sd_pmt_mean / k_sd_sipm_mean
    ratio_k_sd_err = ratio_k_sd * np.sqrt( (k_sd_sipm_mean_err / k_sd_sipm_mean)**2 + (k_sd_pmt_mean_err / k_sd_pmt_mean)**2 )

    ax2[3].errorbar(
        k_sd_sipm.x_center, ratio_k_sd, yerr=ratio_k_sd_err, fmt="o", color="k"
    )
    ax2[3].axhline(1, ls="--", color="grey")
    ax2[3].set(
        xlabel="WCD signal / VEM",
        ylabel=r"$\langle k\rangle_\mathrm{PMT} / \langle k\rangle_\mathrm{SiPM}$",
    )

    fig2.suptitle(title_cuts)
    fig2.tight_layout()

    # ------------------------------------------------------------------
    # FIGURE 3 — rho_mu vs WCD signal profiles
    # ------------------------------------------------------------------
    fig3, ax3 = plt.subplots(2, 2, figsize=(18, 14))
    ax3 = ax3.flatten()

    ## sd signal vs r profiles, for PMT and SiPM datasets
    sd_pmt = Profile(df_pmt_cut["r"], df_pmt_cut["sd_signal"]).doProfile(r_bin_edges)
    sd_sipm = Profile(df_sipm_cut["r"], df_sipm_cut["sd_signal"]).doProfile(r_bin_edges)


    ax3[0].errorbar(df_pmt_cut["r"], df_pmt_cut["sd_signal"], fmt="o", alpha=0.05, ms=1, color=color_pmt)
    ax3[0].errorbar(df_sipm_cut["r"], df_sipm_cut["sd_signal"], fmt="o", alpha=0.05, ms=1, color=color_sipm)

    ax3[0].errorbar(sd_pmt.x_center, sd_pmt.y_mean, yerr=sd_pmt.y_std_of_mean,
                    fmt="o", mfc="None", color=color_pmt, label="PMT dataset")
    ax3[0].errorbar(sd_sipm.x_center, sd_sipm.y_mean, yerr=sd_sipm.y_std_of_mean,
                    fmt="s", alpha=0.5, color=color_sipm, label="SiPM dataset")
    ax3[0].set(xscale="log", yscale="log", xlabel="r / m", ylabel="WCD signal / VEM")

    ax3[0].set_xlim(left=150)
    ax3[0].set_ylim(top=600)

    ax3[0].axhline(y=sd_bins.max(), ls="--", color="k")
    ax3[0].axvline(x=r_bin_edges.min(), ls="--", color="k")

    # horizontal and vertical lines in the sd signal bins 
    for s in sd_bins:
        ax3[0].axhline(y=s, ls="-.", color="grey", alpha=0.2, lw=.8)
        ax3[1].axvline(x=s, ls="-.", color="grey", alpha=0.2, lw=.8)

    ax3[0].legend()

    # rho_mu vs WCD signal 
    rho_sd_pmt = Profile(df_pmt_cut["sd_signal"], rho_pmt_sp).doProfile(sd_bins)
    rho_sd_sipm = Profile(df_sipm_cut["sd_signal"], rho_sipm_sp).doProfile(sd_bins)

    # alternative estimation with PMT using inf window
    rho_sd_pmt_alt = Profile(df_pmt_cut["sd_signal"], rho_inf_window_pmt_1x1_shower_plane).doProfile(sd_bins)

    ax3[1].errorbar(rho_sd_pmt.x_center, rho_sd_pmt.y_mean, yerr=rho_sd_pmt.y_std_of_mean,
                    fmt="o", mfc="None", color=color_pmt, label="PMT")
    ax3[1].errorbar(rho_sd_pmt_alt.x_center, rho_sd_pmt_alt.y_mean, yerr=rho_sd_pmt_alt.y_std_of_mean,
                    fmt="^", mfc="None", color="r", label=f"PMT w/ inf window (corrected by eff={eff:.2f})")

    ax3[1].errorbar(rho_sd_sipm.x_center, rho_sd_sipm.y_mean, yerr=rho_sd_sipm.y_std_of_mean,
                    fmt="s", alpha=0.5, color=color_sipm, label="SiPM")
    ax3[1].set(xscale="log", yscale="log", xlabel="WCD signal / VEM", ylabel=r"$\rho_{\mu} \, / \, \text{m}^{-2}$")
    ax3[1].legend()

    # ratio
    rho_sd_pmt_mean = np.array(rho_sd_pmt.y_mean)
    rho_sd_pmt_mean_err = np.array(rho_sd_pmt.y_std_of_mean)
    rho_sd_sipm_mean = np.array(rho_sd_sipm.y_mean)
    rho_sd_sipm_mean_err = np.array(rho_sd_sipm.y_std_of_mean)

    ratio_rho_sd = rho_sd_pmt_mean / rho_sd_sipm_mean 
    ratio_rho_sd_err = ratio_rho_sd * np.sqrt( (rho_sd_pmt_mean_err / rho_sd_pmt_mean)**2 +
                                              (rho_sd_sipm_mean_err / rho_sd_sipm_mean)**2 )

    ax3[2].errorbar(rho_sd_sipm.x_center, ratio_rho_sd, yerr=ratio_rho_sd_err,
                    fmt="o", color="k", ms=9)
    ax3[2].axhline(y=1, ls="-.", color="grey")
    ax3[2].set(xlabel="WCD signal / VEM", xscale="log", 
        ylabel=r"$\langle \rho_{\mu} \rangle_\text{PMT} / \langle \rho_{\mu} \rangle_\text{SiPM}$")

    # Scaling
    valid_mask = (np.array(rho_sd_pmt.y_mean) > 0) & (np.array(rho_sd_sipm.y_mean) > 0)
    if valid_mask.sum() != len(rho_sd_sipm.y_mean):
        print((f" the rho_mu vs sd_signal profile had some bins "
                f"with invalid entries (lge_min={lge_min:.1f}, lgemax={lge_max:.1f}, theta_min={theta_min:.0f}, theta_max={theta_max:.0f})"))
    fit_rho_sd = fit_scaling_factor(
        y1=np.array(rho_sd_sipm.y_mean)[valid_mask],
        y2=np.array(rho_sd_pmt.y_mean)[valid_mask],
        y1_err=np.array(rho_sd_sipm.y_std_of_mean)[valid_mask],
        y2_err=np.array(rho_sd_pmt.y_std_of_mean)[valid_mask],
    )
    
    scale_rho_sd = fit_rho_sd["scaling_factor"]
    scale_rho_sd_err = fit_rho_sd["scaling_factor_err"]

    rho_sd_pmt_scaled = scale_rho_sd * rho_sd_pmt_mean
    rho_sd_pmt_scaled_err = scale_rho_sd * rho_sd_pmt_mean_err

    # Ratio scaled
    with np.errstate(divide="ignore", invalid="ignore"):
        
        
        ratio_rho_sd_scaled = rho_sd_pmt_scaled / rho_sd_sipm_mean
        ratio_rho_sd_scaled_err = ratio_rho_sd_scaled * np.sqrt( (rho_sd_pmt_scaled_err / rho_sd_pmt_scaled)**2 + 
                                                                 (rho_sd_sipm_mean_err / rho_sd_sipm_mean)**2  )  


    ax3[3].errorbar(rho_sd_sipm.x_center, ratio_rho_sd_scaled, yerr=ratio_rho_sd_scaled_err,
                    fmt="o", color="k", ms=9)
    ax3[3].axhline(y=1, ls="-.", color="grey")
    ax3[3].set(xlabel="WCD signal / VEM", xscale="log", 
        ylabel=r"$\langle \rho_{\mu} \rangle_\text{PMT}^\text{scaled} / \langle \rho_{\mu} \rangle_\text{SiPM}$")
    ax3[3].set_title(f"scaling = {scale_rho_sd:.3f} +- {scale_rho_sd_err:.3f}")

    fig3.suptitle(title_cuts)
    fig3.tight_layout()


    # ------------------------------------------------------------------
    # FIGURE 4 — k vs rho_mu profiles (the mapping k -> rho_mu really tells us wether the time windows and efficiency corrections are playing a role)
    # ------------------------------------------------------------------

    k_bin_edges = np.arange(1, 25, step=1) - 0.5
    k_bin_centers =  0.5 * (k_bin_edges[:-1] + k_bin_edges[1:])

    k_rho_pmt = Profile(df_pmt_cut["k_1x1"], rho_pmt_sp).doProfile(k_bin_edges)
    k_rho_sipm = Profile(df_sipm_cut["k"], rho_sipm_sp).doProfile(k_bin_edges)

    fig4, ax4 = plt.subplots(2, 2, figsize=(18, 14))
    ax4 = ax4.flatten()

    ax4[0].errorbar(k_rho_pmt.x_center, k_rho_pmt.y_mean, yerr=k_rho_pmt.y_std_of_mean,
                    fmt="s", mfc="None", color=color_pmt, label="PMT")
    ax4[0].errorbar(k_rho_sipm.x_center, k_rho_sipm.y_mean, yerr=k_rho_sipm.y_std_of_mean,
                    fmt="o", alpha=0.8, color=color_sipm, label="SiPM")

    ax4[0].set_xlabel("k")
    ax4[0].set_ylabel(r"$\rho_{\mu} \, / \, \text{m}^{-2}$")
    ax4[0].legend()


    # Ratios
    with np.errstate(divide="ignore", invalid="ignore"):
        k_rho_pmt_mean = np.array(k_rho_pmt.y_mean)
        k_rho_pmt_mean_err = np.array(k_rho_pmt.y_std_of_mean)
        
        k_rho_sipm_mean = np.array(k_rho_sipm.y_mean)
        k_rho_sipm_mean_err = np.array(k_rho_sipm.y_std_of_mean)
        
        ratio = k_rho_pmt_mean / k_rho_sipm_mean 
        ratio_err = ratio * np.sqrt( (k_rho_pmt_mean_err / k_rho_pmt_mean)**2 + (k_rho_sipm_mean_err / k_rho_sipm_mean)**2 )
        
        #ratio_scaled = rho_pmt_scaled / rho_sipm 
        #ratio_scaled_err = ratio_scaled * np.sqrt( (rho_pmt_scaled_err / rho_pmt_scaled)**2 + (rho_sipm_err / rho_sipm)**2 )  

    ax4[1].errorbar(k_bin_centers, ratio, yerr=ratio_err, fmt="o", color="k")
    ax4[1].axhline(1, ls="--", color="grey")
    ax4[1].set(xlabel="k",
               ylabel=r"$\rho_\mathrm{PMT} / \rho_\mathrm{SiPM}$")

    # scaling 
    valid_mask_k_rho = (k_rho_pmt_mean > 0) & (k_rho_sipm_mean > 0) & (np.array(k_rho_pmt.n_per_bin) > 8)
    fit_k_rho = fit_scaling_factor(y1=k_rho_sipm_mean[valid_mask_k_rho],
        y2=k_rho_pmt_mean[valid_mask_k_rho],
        y1_err=k_rho_sipm_mean_err[valid_mask_k_rho],
        y2_err=k_rho_pmt_mean_err[valid_mask_k_rho])

    k_rho_pmt_mean_scaled = fit_k_rho["scaling_factor"] * k_rho_pmt_mean
    k_rho_pmt_mean_scaled_err = fit_k_rho["scaling_factor"] * k_rho_pmt_mean_err

    # Ratio scaled
    with np.errstate(divide="ignore", invalid="ignore"):
        
        
        ratio_k_rho_scaled = k_rho_pmt_mean_scaled / k_rho_sipm_mean
        ratio_k_rho_scaled_err = ratio_k_rho_scaled * np.sqrt( (k_rho_pmt_mean_scaled_err / k_rho_pmt_mean_scaled)**2 + 
                                                                 (k_rho_sipm_mean_err / k_rho_sipm_mean)**2  )  

    ax4[2].errorbar(k_bin_centers, ratio_k_rho_scaled, yerr=ratio_k_rho_scaled_err,
                    fmt="o", color="k", ms=9)
    ax4[2].axhline(y=1, ls="-.", color="grey")
    ax4[2].set(xlabel="k",  
        ylabel=r"$\langle \rho_{\mu} \rangle_\text{PMT}^\text{scaled} / \langle \rho_{\mu} \rangle_\text{SiPM}$")
    ax4[2].set_title(f"scaling = {fit_k_rho['scaling_factor'] :.3f} +- {fit_k_rho['scaling_factor_err'] :.3f}")

    fig4.suptitle(title_cuts)
    fig4.tight_layout()


    # ------------------------------------------------------------------
    # FIGURE 5 — k vs nmu (both corrected and uncorrected)
    # -----
    fig5, ax5 = plt.subplots(2, 2, figsize=(18, 14))
    ax5 = ax5.flatten()
    
    # number of muons with corrections 

    k_nmu_pmt = Profile(df_pmt_cut["k_1x1"], df_pmt_cut["nmu"]).doProfile(k_bin_edges)
    k_nmu_sipm = Profile(df_sipm_cut["k"], df_sipm_cut["nmu"]).doProfile(k_bin_edges)
      

    ax5[0].errorbar(k_nmu_pmt.x_center, k_nmu_pmt.y_mean, yerr=k_nmu_pmt.y_std_of_mean,
                    fmt="s", mfc="None", color=color_pmt, label="PMT")

    ax5[0].errorbar(k_nmu_sipm.x_center, k_nmu_sipm.y_mean, yerr=k_nmu_sipm.y_std_of_mean,
                    fmt="o", alpha=0.8, color=color_sipm, label="SiPM")


    ax5[0].errorbar(df_pmt_cut["k_1x1"], df_pmt_cut["nmu"],
                    fmt="s", mfc="None", color=color_pmt, ms=3, alpha=0.3)
    
    ax5[0].errorbar(df_sipm_cut["k"], df_sipm_cut["nmu"],
                    fmt="o", color=color_sipm, ms=3, alpha=0.3)
    ax5[0].set_xlim(right=k_bin_edges.max()+1)

    ax5[0].set_ylim(bottom=-2, top=40)

    ax5[0].set_xlabel("k")
    ax5[0].set_ylabel(r"$\hat{N}_{\mu}$")
    ax5[0].legend()
    
    # ratio 
    # Ratios
    with np.errstate(divide="ignore", invalid="ignore"):
        k_nmu_pmt_mean = np.array(k_nmu_pmt.y_mean)
        k_nmu_pmt_mean_err = np.array(k_nmu_pmt.y_std_of_mean)
        
        k_nmu_sipm_mean = np.array(k_nmu_sipm.y_mean)
        k_nmu_sipm_mean_err = np.array(k_nmu_sipm.y_std_of_mean)
        
        ratio = k_nmu_pmt_mean / k_nmu_sipm_mean 
        ratio_err = ratio * np.sqrt( (k_nmu_pmt_mean_err / k_nmu_pmt_mean)**2 + (k_nmu_sipm_mean_err / k_nmu_sipm_mean)**2 )

    median_ratio = np.median(ratio) 
    mean_ratio = np.mean(ratio)

    ax5[1].errorbar(k_bin_centers, ratio, yerr=ratio_err, fmt="o", color="k")
    ax5[1].axhline(1, ls="--", color="grey")
    ax5[1].axhline(median_ratio, ls="-.", color="k", label=f"median_ratio = {median_ratio:.2f}\nmean_ratio = {mean_ratio:.2f}")
    ax5[1].set(xlabel="k",
               ylabel=r"$\hat{N}_{\mu}^\mathrm{PMT} / \hat{N}_{\mu}\mathrm{SiPM}$")
    
    ax5[1].legend()

    #################################
    # same but with nmu uncorrected 
    k_nmu_pmt_unc = Profile(df_pmt_cut["k_1x1"], df_pmt_cut["nmu_uncorrected"]).doProfile(k_bin_edges)
    k_nmu_sipm_unc = Profile(df_sipm_cut["k"], df_sipm_cut["nmu_uncorrected"]).doProfile(k_bin_edges)

    ax5[2].errorbar(k_nmu_pmt_unc.x_center, k_nmu_pmt_unc.y_mean, yerr=k_nmu_pmt_unc.y_std_of_mean,
                    fmt="s", mfc="None", color=color_pmt, label="PMT")
    ax5[2].errorbar(k_nmu_sipm_unc.x_center, k_nmu_sipm_unc.y_mean, yerr=k_nmu_sipm_unc.y_std_of_mean,
                    fmt="o", alpha=0.8, color=color_sipm, label="SiPM")

    ax5[2].errorbar(df_pmt_cut["k_1x1"], df_pmt_cut["nmu_uncorrected"],
                    fmt="s", mfc="None", color=color_pmt, ms=3, alpha=0.3)
    
    ax5[2].errorbar(df_sipm_cut["k"], df_sipm_cut["nmu_uncorrected"],
                    fmt="o", color=color_sipm, ms=3, alpha=0.3)

    ax5[2].set_xlabel("k")
    ax5[2].set_ylabel(r"$\hat{N}_{\mu}$ uncorrected")
    ax5[2].set_xlim(right=k_bin_edges.max()+1)
    ax5[2].set_ylim(bottom=-2, top=40)
    ax5[2].legend()
    ax5[2].set_title((f"set(n_segments_sipm) = {np.unique(df_sipm_cut['n_segments'])}\n"
                      f"set(n_segments_pmt) = {np.unique(df_pmt_cut['n_segments'])}"))

    xplot = np.linspace(0, max(k_bin_centers), 200)
    yplot = -64*np.log(1-xplot/64)
    ax5[2].plot(xplot, yplot, color="grey", alpha=0.7, ls="--")

    # Ratios
    with np.errstate(divide="ignore", invalid="ignore"):
        k_nmu_unc_pmt_mean = np.array(k_nmu_pmt_unc.y_mean)
        k_nmu_unc_pmt_mean_err = np.array(k_nmu_pmt_unc.y_std_of_mean)
        
        k_nmu_unc_sipm_mean = np.array(k_nmu_sipm_unc.y_mean)
        k_nmu_unc_sipm_mean_err = np.array(k_nmu_sipm_unc.y_std_of_mean)
        
        ratio = k_nmu_unc_pmt_mean / k_nmu_unc_sipm_mean 
        ratio_err = ratio * np.sqrt( (k_nmu_unc_pmt_mean_err / k_nmu_unc_pmt_mean)**2 + (k_nmu_unc_sipm_mean_err / k_nmu_unc_sipm_mean)**2 )

    median_ratio = np.median(ratio) 
    mean_ratio = np.mean(ratio)

    ax5[3].axhline(median_ratio, ls="-.", color="k", label=f"median_ratio = {median_ratio:.2f}\nmean_ratio = {mean_ratio:.2f}")
    ax5[3].errorbar(k_bin_centers, ratio, yerr=ratio_err, fmt="o", color="k")
    ax5[3].axhline(1, ls="--", color="grey")
    ax5[3].set(xlabel="k",
               ylabel=r"$\hat{N}_{\mu}^\mathrm{PMT}  \text{ uncorr.} \, / \, \hat{N}_{\mu}^\mathrm{SiPM} \text{ uncorr.}$")
    ax5[3].legend()
    fig5.suptitle(title_cuts)
    fig5.tight_layout()
  
    # ------------------------------------------------------------------
    # FIGURE 6 — k-fraction (fraction of modules with k=k_fixed) as a function of r and WCD signal
    # -----
    fig6, ax6 = plot_k_fractions(df_pmt=df_pmt_cut, 
                     df_sipm=df_sipm_cut, 
                     lge_min=lge_min, 
                     lge_max=lge_max, 
                     theta_min=theta_min,
                     theta_max=theta_max, 
                     r_bin_edges=r_bin_edges, 
                     sd_signal_bin_edges=sd_bins,
                     use_energy_icrc2025=use_energy_icrc2025 
                     )

    fig6.suptitle(title_cuts)
    fig6.tight_layout()

    return {
        "rho_vs_r": {"fig": fig1, "ax": ax1},
        "k_profiles": {"fig": fig2, "ax": ax2},
        "rho_vs_sd": {"fig": fig3, "ax": ax3},
        "k_vs_rho": {"fig": fig4, "ax": ax4},
        "k_vs_nmu": {"fig": fig5, "ax": ax5},
        "k_fraction": {"fig": fig6, "ax": ax6},
        "scaling_factors": {"mean_ldf_scaling": fit["scaling_factor"], 
                            "mean_ldf_scaling_err": fit["scaling_factor_err"],
                            "rho_wcd_scaling": fit_rho_sd["scaling_factor"],
                            "rho_wcd_scaling_err": fit_rho_sd["scaling_factor_err"],
                            "k_rho_scaling": fit_k_rho["scaling_factor"],
                            "k_rho_scaling_err": fit_k_rho["scaling_factor_err"]}

    }


def plot_k_vs_nmu(df_pmt,
    df_sipm,
    theta_min,
    theta_max,
    lge_min,
    lge_max,
    module_size="all",
    core_in_uc=False,
    use_energy_icrc2025=False):
    """
    Plot Nmu_corrected / Nmu_uncorr as a function of k 
    for SiPM and PMT
    """

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if module_size not in {"all", "small", "large"}:
        raise ValueError("module_size must be 'all', 'small', or 'large'")

    # ------------------------------------------------------------------
    # Common cuts
    # ------------------------------------------------------------------
    if use_energy_icrc2025:
        fname_pmt_icrc25 = ("/storage/gpfs_data/pauger/users/jdejesus/analysis/"
                            "compare_pmt_sipm/src/get_energy_icrc2025/energy_pmt_icrc2025.csv")
        df_pmt_icrc25 = pd.read_csv(fname_pmt_icrc25)

        fname_sipm_icrc25 = ("/storage/gpfs_data/pauger/users/jdejesus/analysis/"
                            "compare_pmt_sipm/src/get_energy_icrc2025/energy_sipm_icrc2025.csv")
        df_sipm_icrc25 = pd.read_csv(fname_sipm_icrc25)
        
        # Overwriting the lge column using the energies of the ICRC2025
        df_pmt = sync_lge_from_2025(df=df_pmt, df_2025=df_pmt_icrc25)
        df_sipm = sync_lge_from_2025(df=df_sipm, df_2025=df_sipm_icrc25)

    def apply_cuts(df):
        mask = (
            (theta_min < df["theta"])
            & (df["theta"] < theta_max)
            & (lge_min < df["lge"])
            & (df["lge"] < lge_max)
        )
        return df.loc[mask].copy()

    df_pmt_cut = apply_cuts(df_pmt)
    df_sipm_cut = apply_cuts(df_sipm)

    if module_size == "large":
        df_pmt_cut = df_pmt_cut[df_pmt_cut["active_area"] >= 7]
        df_sipm_cut = df_sipm_cut[df_sipm_cut["active_area"] >= 7]
    elif module_size == "small":
        df_pmt_cut = df_pmt_cut[df_pmt_cut["active_area"] < 7]
        df_sipm_cut = df_sipm_cut[df_sipm_cut["active_area"] < 7]

    
    if core_in_uc:
        xml_path = ("/storage/gpfs_data/pauger/users/jdejesus/auger_software/"
                    "offline/src/Framework/SDetector/SStationList.xml.in")
        df_tanks = get_sd_tank_coordinates(xml_path)
        
        df_indexed = df_tanks.set_index("station_id")

        # Do a square centered in the unitary cell
        # easting limits
        xleft = df_indexed.loc[1574, "easting"]
        xright = df_indexed.loc[1764, "easting"]

        # northing limits
        ybottom = df_indexed.loc[1773, "northing"]
        yup = df_indexed.loc[1570, "northing"]

        mask_pmt_core_uc = ((xleft < df_pmt_cut["xc"]) & (df_pmt_cut["xc"] < xright) & 
                            (ybottom < df_pmt_cut["yc"]) & (df_pmt_cut["yc"] < yup) )
        
        mask_sipm_core_uc = ((xleft < df_sipm_cut["xc"]) & (df_sipm_cut["xc"] < xright) & 
                            (ybottom < df_sipm_cut["yc"]) & (df_sipm_cut["yc"] < yup) )
        
        df_pmt_cut = df_pmt_cut[mask_pmt_core_uc]
        df_sipm_cut = df_sipm_cut[mask_sipm_core_uc]


    title_cuts = (
        f"{theta_min:.0f}° < theta < {theta_max:.0f}°, "
        f"{lge_min:.1f} < lgE < {lge_max:.1f}, "
        f"module_size = {module_size}, core_in_uc = {core_in_uc}, use_energy_icrc2025 = {use_energy_icrc2025}"
    )

    # ------------------------------------------------------------------
    # FIGURE 1 — k vs nmu_corr / nmu_uncorr for PMT and SiPM separately
    # --------------------------------------
    
    fig, ax = plt.subplots(2, 2, figsize=(18,14))
    ax = ax.flatten()

    color_pmt = "b"
    color_sipm = "g"
    k_bin_edges = np.arange(1, 30) - 0.5
    k_bin_centers =  0.5 * (k_bin_edges[:-1] + k_bin_edges[1:])


    # PMT
    mask_valid_pmt = df_pmt_cut["nmu_uncorrected"].to_numpy() != 0
    ratio_nmu_pmt = df_pmt_cut["nmu"].to_numpy()[mask_valid_pmt] / df_pmt_cut["nmu_uncorrected"].to_numpy()[mask_valid_pmt]
    k_pmt = df_pmt_cut["k_1x1"].to_numpy()[mask_valid_pmt]

    k_nmu_ratio_pmt = Profile(k_pmt, ratio_nmu_pmt).doProfile(k_bin_edges)

    # SiPM
    mask_valid_sipm = df_sipm_cut["nmu_uncorrected"].to_numpy() != 0
    ratio_nmu_sipm = df_sipm_cut["nmu"].to_numpy()[mask_valid_sipm] / df_sipm_cut["nmu_uncorrected"].to_numpy()[mask_valid_sipm]
    k_sipm = df_sipm_cut["k"].to_numpy()[mask_valid_sipm]

    k_nmu_ratio_sipm = Profile(k_sipm, ratio_nmu_sipm).doProfile(k_bin_edges)

    # ratio Nmu_corr / Nmu uncorr as a function of k, both for PMT and SiPM
    ax[0].errorbar(k_nmu_ratio_pmt.x_center, k_nmu_ratio_pmt.y_mean, yerr=k_nmu_ratio_pmt.y_std_of_mean,
                fmt="s", mfc="None", color=color_pmt, ms=10, label="PMT (with time windows)")
    ax[0].errorbar(k_nmu_ratio_sipm.x_center, k_nmu_ratio_sipm.y_mean, yerr=k_nmu_ratio_sipm.y_std_of_mean,
                fmt="o", color=color_sipm, ms=10, label="SiPM")
    ax[0].set(xlabel="k", ylabel=r"$\langle  \hat{N}_{\mu} \, / \, \hat{N}_{\mu}^\text{uncorr} \rangle$")

    ax[0].axhline(y=1, ls="--", color="grey", alpha=0.7)
    ax[0].legend()

    # Comparison of Nmu uncorrected with and without time windows (ie inf window) 
    # only for the PMT case

    ratio_windows_pmt = df_pmt_cut["nmu_inf_uncorrected_1x1"].to_numpy()[mask_valid_pmt] / df_pmt_cut["nmu_uncorrected"].to_numpy()[mask_valid_pmt]

    k_nmu_windows_pmt = Profile(k_pmt, ratio_windows_pmt).doProfile(k_bin_edges)

    ax[1].errorbar(k_nmu_windows_pmt.x_center, k_nmu_windows_pmt.y_mean, 
                    yerr=k_nmu_windows_pmt.y_std_of_mean, fmt="s", color=color_pmt,
                    ms=10)
    # nmu_inf / nmu_window should be equal to epsilon_inf / epsilon_w = 0.87 / 0.95; at least to first order
    if module_size == "large":
        ratio_efficiencies = 0.87 / 0.95
    elif module_size == "small":
        ratio_efficiencies = 0.94 / 1.04
    else:
        ratio_efficiencies = 1
    ax[1].axhline(y=ratio_efficiencies, color="darkturquoise", ls="-.", lw=0.8, alpha=0.7,
                label=r"$\epsilon(w=1024) / \epsilon(w=7)$")
    
    ax[1].set(xlabel="k", ylabel=r"$\langle  \hat{N}_{\mu}^\text{inf} \, / \, \hat{N}_{\mu}^\text{windows} \rangle$")
    ax[1].legend()
    ax[1].axhline(y=1, ls="--", color="grey")
    ax[1].set_title("uncorrected values; only for PMT")


    # distribution of Nmu for various k. Using only modules
    # with no masked channels (n_segments = 64). ONLY PMT 
    def plot_nmu_fixed_k(df_pmt_cut, k_value, ax, normalize=False,
                        color="b"):
        mask_bool = (df_pmt_cut["k_1x1"].to_numpy() == k_value) & (df_pmt_cut["n_segments"].to_numpy() == 64)
        values, counts = np.unique(df_pmt_cut["nmu_uncorrected"].to_numpy()[mask_bool], return_counts=True)
        values_inf_window, counts_inf_window = np.unique(df_pmt_cut["nmu_inf_uncorrected_1x1"].to_numpy()[mask_bool], return_counts=True)
        if normalize:
            counts = counts / np.sum(counts)
            counts_inf_window = counts_inf_window / np.sum(counts_inf_window)

        ax.bar(values, counts, width=0.8, align="center", color=color, alpha=0.6)
        ax.axvline(x=np.mean(df_pmt_cut["nmu_uncorrected"].to_numpy()[mask_bool]), color=color, alpha=0.8)

        ax.bar(values_inf_window, counts_inf_window, width=0.8, align="center", label=f"k = {k_value} (window = 1024)", 
            alpha=0.5, edgecolor=color, facecolor="none")

    # nmu_uncorr distribution for k = 1
    plot_nmu_fixed_k(df_pmt_cut, k_value=1, ax=ax[2], normalize=False, color="b")
    plot_nmu_fixed_k(df_pmt_cut, k_value=1, ax=ax[3], normalize=True, color="b")

    # nmu_uncorr distribution for k = 5
    plot_nmu_fixed_k(df_pmt_cut, k_value=5, ax=ax[2], normalize=False, color="orange")
    plot_nmu_fixed_k(df_pmt_cut, k_value=5, ax=ax[3], normalize=True, color="orange")

    # nmu_uncorr distribution for k = 10
    plot_nmu_fixed_k(df_pmt_cut, k_value=10, ax=ax[2], normalize=False, color="g")
    plot_nmu_fixed_k(df_pmt_cut, k_value=10, ax=ax[3], normalize=True, color="g")
     
    

    ax[2].set(xlabel=r"$ \hat{N}_{\mu}^\text{uncorr}$", ylabel="Counts", title="PMT only",
        yscale="log")
    ax[2].legend()

    ax[3].set(xlabel=r"$ \hat{N}_{\mu}^\text{uncorr}$", ylabel="Probability", title="PMT only",
        )
    ax[3].legend()


    ## Printing modules with possible over-counting, 
    # defined as those in which nmu_uncorr (w=7) > nmu_uncorr (w=1024)
    # I will keep only up to k = 2 for now 
    def get_pmt_modules_suspected_overcounting(df_pmt):
        k_bool = (df_pmt["k_1x1"].to_numpy() <= 2) & (df_pmt["n_segments"].to_numpy() == 64)
        suspect_bool = (df_pmt["nmu_uncorrected"].to_numpy() > df_pmt["nmu_inf_uncorrected_1x1"].to_numpy())

        df_suspect = df_pmt[(k_bool) & (suspect_bool)]
        
        return df_suspect   

    df_suspect = get_pmt_modules_suspected_overcounting(df_pmt_cut)


    fig.suptitle(title_cuts)
    fig.tight_layout()

    return {"fig": fig,
            "df_suspect_overcounting": df_suspect}



def plot_rho450_vs_lge( df_pmt,
    df_sipm,
    theta_min,
    theta_max,
    lge_bin_edges,
    module_size="all",
    core_in_uc=False,
    use_energy_icrc2025=False):
    """
    """
    
    lge_bin_centers = 0.5 * (lge_bin_edges[:-1] + lge_bin_edges[1:])
    rmin, rmax = 400, 500
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if module_size not in {"all", "small", "large"}:
        raise ValueError("module_size must be 'all', 'small', or 'large'")

    # ------------------------------------------------------------------
    # Common cuts
    # ------------------------------------------------------------------
    if use_energy_icrc2025:
        fname_pmt_icrc25 = ("/storage/gpfs_data/pauger/users/jdejesus/analysis/"
                            "compare_pmt_sipm/src/get_energy_icrc2025/energy_pmt_icrc2025.csv")
        df_pmt_icrc25 = pd.read_csv(fname_pmt_icrc25)

        fname_sipm_icrc25 = ("/storage/gpfs_data/pauger/users/jdejesus/analysis/"
                            "compare_pmt_sipm/src/get_energy_icrc2025/energy_sipm_icrc2025.csv")
        df_sipm_icrc25 = pd.read_csv(fname_sipm_icrc25)
        
        # Overwriting the lge column using the energies of the ICRC2025
        df_pmt = sync_lge_from_2025(df=df_pmt, df_2025=df_pmt_icrc25)
        df_sipm = sync_lge_from_2025(df=df_sipm, df_2025=df_sipm_icrc25)

    def apply_cuts(df):
        mask = (
            (theta_min < df["theta"])
            & (df["theta"] < theta_max) 
            & (rmin < df["r"])
            & (df["r"] < rmax) 
        )
        return df.loc[mask].copy()

    df_pmt_cut = apply_cuts(df_pmt)
    df_sipm_cut = apply_cuts(df_sipm)

    if module_size == "large":
        df_pmt_cut = df_pmt_cut[df_pmt_cut["active_area"] >= 7]
        df_sipm_cut = df_sipm_cut[df_sipm_cut["active_area"] >= 7]
    elif module_size == "small":
        df_pmt_cut = df_pmt_cut[df_pmt_cut["active_area"] < 7]
        df_sipm_cut = df_sipm_cut[df_sipm_cut["active_area"] < 7]

    
    if core_in_uc:
        xml_path = ("/storage/gpfs_data/pauger/users/jdejesus/auger_software/"
                    "offline/src/Framework/SDetector/SStationList.xml.in")
        df_tanks = get_sd_tank_coordinates(xml_path)
        
        df_indexed = df_tanks.set_index("station_id")

        # Do a square centered in the unitary cell
        # easting limits
        xleft = df_indexed.loc[1574, "easting"]
        xright = df_indexed.loc[1764, "easting"]

        # northing limits
        ybottom = df_indexed.loc[1773, "northing"]
        yup = df_indexed.loc[1570, "northing"]

        mask_pmt_core_uc = ((xleft < df_pmt_cut["xc"]) & (df_pmt_cut["xc"] < xright) & 
                            (ybottom < df_pmt_cut["yc"]) & (df_pmt_cut["yc"] < yup) )
        
        mask_sipm_core_uc = ((xleft < df_sipm_cut["xc"]) & (df_sipm_cut["xc"] < xright) & 
                            (ybottom < df_sipm_cut["yc"]) & (df_sipm_cut["yc"] < yup) )
        
        df_pmt_cut = df_pmt_cut[mask_pmt_core_uc]
        df_sipm_cut = df_sipm_cut[mask_sipm_core_uc]


    title_cuts = (
        f"{theta_min:.0f}° < theta < {theta_max:.0f}°, "
        f"{rmin:.0f} < r / m < {rmax:.0f}, "
        f"module_size = {module_size}, core_in_uc = {core_in_uc}, use_energy_icrc2025 = {use_energy_icrc2025}"
    )


    # normalizing the rho_mu by energy --> rho_mu_norm = rho_mu / (E/10**18)**0.9
    norm_factor_pmt = (10**(df_pmt_cut["lge"].to_numpy()-18))**0.9
    rho_pmt_norm = df_pmt_cut["rho_shower_plane"] / norm_factor_pmt

    norm_factor_sipm = (10**(df_sipm_cut["lge"].to_numpy()-18))**0.9
    rho_sipm_norm = df_sipm_cut["rho_shower_plane"] / norm_factor_sipm

    # Profiles 
    lge_bin_edges_sipm = list(lge_bin_edges)
    lge_bin_edges_sipm.append(max(lge_bin_edges) + 0.2)
    lge_bin_edges_sipm.append(max(lge_bin_edges) + 0.6)
    print(f"lge_bin_edges_sipm = {lge_bin_edges_sipm}")
    p_pmt = Profile(df_pmt_cut["lge"], rho_pmt_norm).doProfile(lge_bin_edges)
    p_sipm = Profile(df_sipm_cut["lge"], rho_sipm_norm).doProfile(lge_bin_edges_sipm)
    print(f"lge_bin_centers_sipm = {p_sipm.x_center}")
    print(f"n_per_bin_sipm = {p_sipm.n_per_bin}")

    fig, ax = plt.subplots(1, 1, figsize=(16, 11))
    #ax = ax.flatten( )

    color_pmt = "b"
    color_sipm = "g"

    ax.errorbar(p_pmt.x_mean, p_pmt.y_mean, yerr=p_pmt.y_std_of_mean,
                    fmt="s", alpha=0.8, ms=10, label="PMT", mfc="None", color=color_pmt)
    ax.errorbar(p_sipm.x_mean, p_sipm.y_mean, yerr=p_sipm.y_std_of_mean,
                    fmt="o", alpha=0.8, ms=10, label="SiPM", color=color_sipm)
    
    ax.errorbar(df_pmt_cut["lge"], rho_pmt_norm, fmt="s", mfc="None", alpha=0.15, 
                ms=4, color=color_pmt)
    ax.errorbar(df_sipm_cut["lge"], rho_sipm_norm, fmt="o", mfc="None", alpha=0.15, 
                ms=4, color=color_sipm)           

    ax.set_xlim(left=min(lge_bin_edges)-0.1, right=max(lge_bin_edges_sipm)+0.1)
    ax.set_ylim([-0.02, 2.25])
    ax.set_xticks(lge_bin_edges_sipm)
    ax.set_yticks(np.arange(1.2, 2.21, step=0.1))
    ax.set(xlabel=r"$\lg(E / \text{eV})$", ylabel=r"$\langle \rho_{\mu} \, / \, \left( E / 10^{18} \text{eV} \right)^{0.9} \rangle $")
    ax.set_title(f"{rmin:.0f} < r / m < {rmax:.0f}")
    ax.legend()

    fig.suptitle(title_cuts)

    
    for l in lge_bin_edges_sipm:
        ax.axvline(x=l, ls="--", alpha=0.5, color="grey", lw=0.8)
    
    for a in np.array([1, 2, 3]) / 10.45:
        lge_plot = np.linspace(17.3, 18)
        #ax.plot(lge_plot, a*10**(-0.9*(lge_plot-18)), ls="--", color="grey", alpha=0.8)

    return {"fig": fig, "ax": ax}


def plot_relative_efficiency(df_pmt,
                        df_sipm,
                        theta_min,
                        theta_max,
                        lge_min,
                        lge_max,
                        module_size="large",
                        core_in_uc=False,
                        use_energy_icrc2025=False,):
    
    """
    with infinte window, plot relative efficiency of the two detectors
    """

    # --- CUTS ---

    def apply_cuts(df):
        mask = (
            (theta_min < df["theta"])
            & (df["theta"] < theta_max)
            & (lge_min < df["lge"])
            & (df["lge"] < lge_max)
        )
        return df.loc[mask].copy()

    df_pmt_cut = apply_cuts(df_pmt)
    df_sipm_cut = apply_cuts(df_sipm)

    if module_size == "large":
        df_pmt_cut = df_pmt_cut[df_pmt_cut["active_area"] >= 7]
        df_sipm_cut = df_sipm_cut[df_sipm_cut["active_area"] >= 7]
        expected_ratio = 0.87 / 0.98
    elif module_size == "small":
        df_pmt_cut = df_pmt_cut[df_pmt_cut["active_area"] < 7]
        df_sipm_cut = df_sipm_cut[df_sipm_cut["active_area"] < 7]
        expected_ratio = 0.95 / 1
    else:
        expected_ratio = 1

    if core_in_uc:
        xml_path = ("/storage/gpfs_data/pauger/users/jdejesus/auger_software/"
                    "offline/src/Framework/SDetector/SStationList.xml.in")
        df_tanks = get_sd_tank_coordinates(xml_path)
        
        df_indexed = df_tanks.set_index("station_id")

        # Do a square centered in the unitary cell
        # easting limits
        xleft = df_indexed.loc[1574, "easting"]
        xright = df_indexed.loc[1764, "easting"]

        # northing limits
        ybottom = df_indexed.loc[1773, "northing"]
        yup = df_indexed.loc[1570, "northing"]

        mask_pmt_core_uc = ((xleft < df_pmt_cut["xc"]) & (df_pmt_cut["xc"] < xright) & 
                            (ybottom < df_pmt_cut["yc"]) & (df_pmt_cut["yc"] < yup) )
        
        mask_sipm_core_uc = ((xleft < df_sipm_cut["xc"]) & (df_sipm_cut["xc"] < xright) & 
                            (ybottom < df_sipm_cut["yc"]) & (df_sipm_cut["yc"] < yup) )
        
        df_pmt_cut = df_pmt_cut[mask_pmt_core_uc]
        df_sipm_cut = df_sipm_cut[mask_sipm_core_uc]

    # ----- relative eff analysis

    r_bin_edges = np.arange(200, 801, 100)

    title_cuts = (
        f"{theta_min:.0f}° < theta < {theta_max:.0f}°, "
        f"{lge_min:.1f} < lgE < {lge_max:.1f}, "
        f"module_size = {module_size}, core_in_uc = {core_in_uc}, use_energy_icrc2025 = {use_energy_icrc2025}"
    )

    k_pmt = Profile(df_pmt_cut["r"], df_pmt_cut["k_1x1"]).doProfile(r_bin_edges)
    k_sipm = Profile(df_sipm_cut["r"], df_sipm_cut["k"]).doProfile(r_bin_edges)

    # ratio between eff_PMT / eff_SiPM  = ln(1-<k>_PMT / 64) / ln(1-<k>_SiPM / 64)
    # as a function of r
    ns = 64
    num_pmt = np.log( 1 - np.array(k_pmt.y_mean) / ns  )
    denom_sipm = np.log( 1 - np.array(k_sipm.y_mean) / ns  )
    ratio = num_pmt / denom_sipm 

    # error in numerator (assuming k is binomial(ns, p=1-exp(-mu*epsilon/ns)))
    ratio_error = []

    for i, (kpmt, ksipm) in enumerate(zip(k_pmt.y_mean, k_sipm.y_mean)):
        npmt = k_pmt.n_per_bin[i] 
        nsipm = k_sipm.n_per_bin[i]

        p_pmt = kpmt / ns 
        p_sipm = ksipm / ns 

        num_err = np.sqrt( p_pmt / (npmt * ns * (1-p_pmt)) )
        denom_err = np.sqrt( p_sipm / (nsipm * ns * (1-p_sipm)) )

        rerr = ratio[i] * np.sqrt( (num_err / num_pmt[i])**2 + (denom_err / denom_sipm[i])**2 )

        ratio_error.append(rerr)

    wmean = compute_weighted_mean(values=ratio, values_err=ratio_error)

    fig, ax = plt.subplots(1, 2, figsize=(16, 10))

    ax[0].errorbar(k_pmt.x_center, ratio, yerr=ratio_error, fmt="o", color="b")
    ax[0].axhline(y=wmean, label=f"weighted_mean = {wmean:.2f}")

    ax[0].axhline(y=expected_ratio, ls="--", color="grey", alpha=0.9, label="expected")

    ax[0].set_xlabel(r"$r \, / \, \text{m}$")
    ax[0].set_ylabel(r"$\epsilon_\text{PMT} \, / \, \epsilon_\text{SiPM}$")
    ax[0].set_title(r"$\epsilon_\text{PMT} \, / \, \epsilon_\text{SiPM} = \ln ( 1 - \langle k \rangle_\text{PMT} / 64) \, / \, \ln ( 1 - \langle k \rangle_\text{SiPM} / 64)$")

    ax[0].legend()
    
    fig.suptitle(title_cuts)

    return {"fig": fig}
    


    


# -----------------------------
# Event-level residuals for the two datasets
# TODO
def plot_event_level_residuals_ldf(
    df_pmt,
    df_sipm,
    theta_min,
    theta_max,
    lge_bin_edges,
    module_size="all",
    core_in_uc=False
):
    """
    """

    lge_bin_centers = 0.5 * (lge_bin_edges[:-1] + lge_bin_edges[1:])
    nbins = len(lge_bin_centers)

    return 
