# ======== Standard library imports ========
import os
import pickle
import sys
from pathlib import Path
from datetime import datetime, timedelta

# ======== Third-party imports ========
import numpy as np
from scipy.stats import chi2, poisson, norm
from scipy.stats import t as student_t
from scipy.optimize import least_squares
from iminuit import Minuit
import pandas as pd 
import matplotlib.pyplot as plt 

# =========== My imports ================
sys.path.append(os.path.expanduser("/storage/gpfs_data/pauger/users/jdejesus/my_python_modules"))
from myClasses import Profile
from myMplStyles import setStyle
from myUsefulFunctions import generate_colormap

setStyle(False)


def sync_gps_utc_axes(ax, n_ticks=8):
    """
    Force bottom (GPS) and top (UTC) axes to share identical tick positions.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis with GPS seconds on x-axis.
    n_ticks : int
        Number of ticks to enforce.
    """

    gps_epoch = datetime(1980, 1, 6)

    # Get current limits
    gps_min, gps_max = ax.get_xlim()

    # Define common ticks
    gps_ticks = np.linspace(gps_min, gps_max, n_ticks)

    # --- Bottom axis (GPS) ---
    ax.set_xticks(gps_ticks)

    # --- Top axis (UTC) ---
    secax = ax.secondary_xaxis('top')
    secax.set_xticks(gps_ticks)

    # Convert to UTC labels
    utc_dates = [gps_epoch + timedelta(seconds=float(t)) for t in gps_ticks]
    labels = [dt.strftime("%m/%Y") for dt in utc_dates]

    secax.set_xticklabels(labels, fontsize="medium")
    secax.set_xlabel("UTC (mm/yyyy)", fontsize="medium")

    return secax

## TODO Implementar analyze_two_bar_events para estimar corner-clipping ineficientes y corner-clipping eficientes

class Sequences:
    """
    Data-driven modeling of detector sequence (string of consectuive 1s) activity for background
    characterization and single-muon response estimation.

    This class analyzes time-ordered "sequence" objects extracted from
    detector readout data in order to:

    1. Characterize and model the background activity in different time
       regions (early/late noise windows).
    2. Estimate background rates for:
       - Generic sequences (any width)
       - Muon-like patterns (typically width >= 4 bins)
    3. Derive probability distributions of sequence observables,
       in particular the width distribution conditioned on noise.
    4. Provide input for single-muon response modeling, i.e. the
       probability that an individual muon produces:
           - 0 fired strips (inefficiency / non-detection)
           - 1 fired strip (single-hit response)
           - 2 fired strips (multi-strip response / clustering)
    5. Identify and remove unstable data-taking periods based on
       time-dependent variations of sequence observables.

    The methodology is fully data-driven and relies on separating
    physics-like activity from noise using predefined time windows:

    - noise_early / noise_late: regions dominated by background
    - trigger_scope: region associated with physics-triggered activity

    The class operates on a per-module basis and assumes an input
    dataframe containing reconstructed sequence-level information
    (start time, width, event association, GPS time, etc.).

    A key aspect of the analysis is the assumption that sequence widths
    encode physical information about particle interactions in the
    scintillator system, allowing the extraction of:

    - Background rates (Poisson-like behavior in noise windows)
    - Width probability distributions P(w | noise)
    - Scaling relations between sequence rates and muon-induced patterns
    - Constraints on single-muon detection probabilities

    These ingredients are ultimately used to build a probabilistic model
    of detector response to individual muons, including low-multiplicity
    cases (0-2 fired strips), which are critical for efficiency
    corrections and reconstruction bias studies.
    """


    def __init__(self, counter, module, noise_early=None, noise_late=None, 
                trigger_scope=None, trace_length=2048, df=None,):
                 
        """
        Initialize the Sequences analysis container.

        Parameters
        ----------
        counter : int
            Counter id used for output file naming and bookkeeping.
        module : int or str
            Module id for output file naming and bookkeeping.
        noise_early : tuple(float, float), optional
            Time window (t0, t1) defining the early noise region.
        noise_late : tuple(float, float), optional
            Time window (t0, t1) defining the late noise region.
        trigger_scope : tuple(float, float), optional
            Time window defining the trigger region of interest.
        trace_length : int, default=2048
            Length of the time trace in time bins (3.125 ns units).
        df : pandas.DataFrame, optional
            Input sequence-level dataframe containing at least:
            - entryType
            - eventId
            - startTime
            - width
            - gpsSecond
            - totalChannelsOn
            - isT4, isT5 (optional depending on cuts)

        Attributes
        ----------
        df : pandas.DataFrame
            Working copy of the input dataframe (may be cleaned in-place).
        rate_seq_bkg : dict
            Background rate estimates for sequences in early/late windows.
        rate_muon_pattern_bkg : dict
            Background rate estimates for muon-like patterns (width >= 4).
        prob_width_noise : dict or None
            Probability distribution of sequence widths in noise windows.
        bad_periods_found_and_removed : bool or None
            Flag indicating whether unstable GPS periods were removed.
        label_cuts : str
            Human-readable description of applied time window cuts.

        Notes
        -----
        The class assumes that noise windows and trigger scope are pre-defined
        and remain fixed during analysis.
        """

        if noise_early is not None:
            assert len(noise_early) == 2, "Expected (t_noise1, t_noise2) for noise_early"
            assert noise_early[1] > noise_early[0], "t_noise2 must be > t_noise1"

        if noise_late is not None:
            assert len(noise_late) == 2, "Expected (t_noise1, t_noise2) for noise_late"
            assert noise_late[1] > noise_late[0], "t_noise2 must be > t_noise1"

        if trigger_scope is not None:
            assert len(trigger_scope) == 2, "Expected (t1, t2) for trigger_scope"
            assert trigger_scope[1] > trigger_scope[0], "t2 must be > t1"

        self.counter = counter 
        self.module = module 
        self.trace_length = trace_length

        # Cuts for the time windows of the noise early/late and the trigger scop
        self.noise_early = noise_early
        self.noise_late = noise_late
        self.trigger_scope = trigger_scope
        
        self.df = df.copy() if df is not None else None

        self.bad_periods_found_and_removed = None

        # Estimation of background rate 
        # Note that this rate Lambda is the one expected in the whole module -> Lambda = 64*lam_1 = rate_patterns_bkg * 2048
        self.rate_seq_bkg = {"early": {"rate": -1, "rate_err": -1}, "late": {"rate": -1, "rate_err": -1}}
        self.rate_muon_pattern_bkg = {"early": {"rate": -1, "rate_err": -1}, "late": {"rate": -1, "rate_err": -1}}

        # Probability distribution of width of noise sequences
        self.prob_width_noise = None 

        self.label_cuts = f"Noise early: {noise_early[0]} < t < {noise_early[1]}; Noise late: {noise_late[0]} < t < {noise_late[1]}\n"
        self.label_cuts += f"Trigger scope: {trigger_scope[0]} < t < {trigger_scope[1]}"

    def sanity_checks(self):
        """
        Perform basic consistency checks on the internal dataframe.

        Ensures:
        - The dataframe is defined
        - Each module_empty event appears only once per eventId
        - No duplicate eventId entries exist for module_empty rows

        Raises
        ------
        ValueError
            If the dataframe is None.
        AssertionError
            If inconsistencies or duplicate entries are detected.
        """
        if self.df is None:
            raise ValueError("df is None")

        df_module_empty = self.df[self.df["entryType"] == "module_empty"].copy()
        

        assert len(df_module_empty) == len(df_module_empty["eventId"].unique())
        assert not df_module_empty["eventId"].duplicated().any(), \
            "There are duplicates for module_empty entries!"

    def exclude_bad_periods(self, outdir):
        """
        Identify and remove unstable data-taking periods based on time evolution
        of sequence properties.

        This method:
        1. Filters non-finite energy events
        2. Computes time-binned profiles of:
        - Mean sequence width
        - Mean start time
        3. Identifies outlier GPS intervals using robust statistics
        4. Removes affected periods from the dataframe
        5. Recomputes all distributions after cleaning
        6. Generates diagnostic plots before and after cleaning

        Parameters
        ----------
        outdir : str
            Output directory where diagnostic plots are saved.

        Behavior
        --------
        Bad periods are defined as time bins where:
        - Mean width deviates > 3σ from median, or is unphysically large
        - Mean start time deviates > 3σ from median, or is zero

        After removal, the dataframe is updated in-place and all subsequent
        analyses use the cleaned dataset.

        Outputs
        -------
        Produces multiple diagnostic figures:
        - Width vs GPS time (raw and cleaned)
        - Start time vs GPS time (raw and cleaned)
        - Start time distributions with time-window overlays
        - Width distributions per noise region

        Notes
        -----
        - Requires external Profile class for time binning
        - Uses self.noise_early, self.noise_late, and self.trigger_scope
        - Sets self.bad_periods_found_and_removed flag
        """
        
        # Remove events for wich lgE is -inf (occurs when theta is above
        # the theta_max of the SD CIC)
        mask_inf = np.isfinite(self.df["lgEnergy"])

        self.df = self.df[mask_inf]
       
        # For this analysis, only the entry type "sequence" is relevant
        # Howver, the masking is done over the whole dataframe df at the end
       
        df_seq = self.df[self.df["entryType"] == "sequence"].copy()

        assert (df_seq["width"] > 0).all()
        assert (df_seq["startTime"] >= 0).all()

        gps_min = df_seq["gpsSecond"].min()
        gps_max = df_seq["gpsSecond"].max()

        seconds_in_hour = 3600
        seconds_in_day = seconds_in_hour*24 

        step = 15*seconds_in_day

       
        gps_bin_edges = np.arange(gps_min, gps_max+1, step=step)
        gps_bin_centers = gps_bin_edges[:-1] + np.diff(gps_bin_edges)/2

        pwidth = Profile(df_seq["gpsSecond"], df_seq["width"]).doProfile(gps_bin_edges)
        pt = Profile(df_seq["gpsSecond"], df_seq["startTime"]).doProfile(gps_bin_edges)

        median_width = np.nanmedian(pwidth.y_mean)
        std_width =  np.nanquantile(pwidth.y_mean, q=0.84) - np.nanquantile(pwidth.y_mean, q=0.16)
        std_width /= 2

        median_t = np.nanmedian(pt.y_mean)
        std_t =  np.nanquantile(pt.y_mean, q=0.84) - np.nanquantile(pt.y_mean, q=0.16)
        std_t /= 2

        # isUUB flag vs time 
        pUUB = Profile(df_seq["gpsSecond"], df_seq["isUUB"]).doProfile(gps_bin_edges)

        # find the earliest gps bin in which "isUUB" is 1
        uub_true_mask = np.asarray(pUUB.y_mean) > 0.98
        if uub_true_mask.any():
            gps_uub_mark = np.min(np.asarray(pUUB.x_mean)[uub_true_mask])
        else:
            gps_uub_mark = None
            print(f"No UB->UUB transition found for c{str(self.counter).zfill(4)}m{self.module}")
        

        # -----------------------
        # figures 
        ms = 10

        # ----------
        fig, ax = plt.subplots(1, 1, figsize=(14, 7))
        
        ax.errorbar(pUUB.x_mean, pUUB.y_mean, yerr=pUUB.y_std_of_mean, fmt="o", ms=ms)
        if gps_uub_mark is not None:
            ax.axvline(x=gps_uub_mark, ls="-.", lw=2, color="r", label="UUB transition")
        ax.legend(fontsize="medium")
        
        ax.set_xlabel("GPS second")
        ax.set_ylabel(r"$\langle \text{isUUB} \rangle $")

        # secondary axis with UTC date times
        sync_gps_utc_axes(ax=ax)

        figname = f"c{str(self.counter).zfill(4)}m{self.module}_isUUB_vs_gps_raw.png"

        fig.savefig(os.path.join(outdir, figname), bbox_inches="tight")
        plt.close(fig)
        print(f"{figname} saved")


        # --------------------------
        # -- Width vs time
        fig, ax = plt.subplots(1, 1, figsize=(14, 7))
        
        ax.errorbar(pwidth.x_mean, pwidth.y_mean, yerr=pwidth.y_std_of_mean, fmt="o", ms=ms)
        ax.axhline(y=median_width, ls="--", color="k", alpha=0.8, lw=2, label=f"median = {median_width:.1f}, std = {std_width:.1f}")
        ax.axhspan(ymin=median_width - 3*std_width, ymax=median_width + 3*std_width, color="grey", alpha=0.5)
        

        # detecting bad periods as periods with outliers in the mean width
        outlier_w_mask_1 = np.abs((np.asarray(pwidth.y_mean) - median_width)) / (3*std_width) > 1
        outlier_w_mask_2 = (np.asarray(pwidth.y_mean)) > 80

        outlier_w_mask = outlier_w_mask_1 | outlier_w_mask_2


        if outlier_w_mask.any():
            ax.errorbar(np.asarray(pwidth.x_mean)[outlier_w_mask], np.asarray(pwidth.y_mean)[outlier_w_mask], fmt="x", color="r", ms=13)

        ax.set_xlabel("GPS second")
        ax.set_ylabel(r"$\langle \text{width} \, / \, (3.125 \text{ ns})\rangle $")
        
        if gps_uub_mark is not None:
            ax.axvline(x=gps_uub_mark, ls="-.", lw=2, color="r", label="UUB transition")
        
        ax.legend(fontsize="medium")

        # secondary axis with UTC date times
        sync_gps_utc_axes(ax=ax)

        figname = f"c{str(self.counter).zfill(4)}m{self.module}_width_vs_gps_raw.png"

        fig.savefig(os.path.join(outdir, figname), bbox_inches="tight")
        plt.close(fig)
        print(f"{figname} saved")

        # ---------------------------
        # -- Start time vs time
        fig, ax = plt.subplots(1, 1, figsize=(14, 7))
        
        ax.errorbar(pt.x_mean, pt.y_mean, yerr=pt.y_std_of_mean, fmt="o", ms=ms)
        ax.axhline(y=median_t, ls="--", color="k", alpha=0.8, lw=2, label=f"median = {median_t:.1f}, std = {std_t:.1f}")
        ax.axhspan(ymin=median_t - 3*std_t, ymax=median_t + 3*std_t, color="grey", alpha=0.5)
        ax.legend(fontsize="medium")

        # detecting bad periods as periods with outliers in the mean t
        outlier_t_mask1 = np.abs((np.asarray(pt.y_mean) - median_t)) / (3*std_t) > 1
        outlier_t_mask2 = np.asarray(pt.y_mean) == 0

        outlier_t_mask = outlier_t_mask1 | outlier_t_mask2

        if outlier_t_mask.any():
            ax.errorbar(np.asarray(pt.x_mean)[outlier_t_mask], np.asarray(pt.y_mean)[outlier_t_mask], fmt="x", color="r", ms=13)

        ax.set_xlabel("GPS second")
        ax.set_ylabel(r"$\langle \text{t} \, / \, (3.125 \text{ ns})\rangle $")
        
        if gps_uub_mark is not None:
            ax.axvline(x=gps_uub_mark, ls="-.", lw=2, color="r", label="UUB transition")

        # secondary axis with UTC date times
        sync_gps_utc_axes(ax=ax)

        figname = f"c{str(self.counter).zfill(4)}m{self.module}_start_time_vs_gps_raw.png"

        fig.savefig(os.path.join(outdir, figname), bbox_inches="tight")
        plt.close(fig)
        print(f"{figname} saved")

        # -- Distributions beofre the cleaning
        # Just ploting the distribution BEFORE removing bad periods of 
        # (i) Start times
        # (ii) Widths for different time windows (noise early, late and trigger scope)
        
        fig, ax = plt.subplots(1, 2, figsize=(17, 7))
        
        tbinedges = np.arange(0, self.trace_length+2, step=1)-0.5
        ax[0].hist(df_seq["startTime"], bins=tbinedges, lw=2, histtype="step")
        
        ax[0].axvspan(xmin=self.noise_early[0], xmax=self.noise_early[1], alpha=0.1, color="grey")
        ax[0].axvspan(xmin=self.noise_late[0], xmax=self.noise_late[1], alpha=0.1, color="m")
        ax[0].axvspan(xmin=self.trigger_scope[0], xmax=self.trigger_scope[1], alpha=0.1, color="g")
        
        ax[0].set(xlabel="Start time / (3.125 ns)", ylabel="Counts", yscale="log")


        wbinedges = np.arange(0, 20, step=1) - 0.5
        noise_early_mask = (self.noise_early[0] < df_seq["startTime"]) & (df_seq["startTime"] < self.noise_early[1])
        noise_late_mask = (self.noise_late[0] < df_seq["startTime"]) & (df_seq["startTime"] < self.noise_late[1])
        trigger_scope_mask = (self.trigger_scope[0] < df_seq["startTime"]) & (df_seq["startTime"] < self.trigger_scope[1])

        ax[1].hist(df_seq[noise_early_mask]["width"], bins=wbinedges, histtype="step", lw=2, color="grey",
                    label="noise early", density=True)
        ax[1].hist(df_seq[noise_late_mask]["width"], bins=wbinedges, histtype="step", lw=2, color="m",
                    label="noise late", density=True)
        ax[1].hist(df_seq[trigger_scope_mask]["width"], bins=wbinedges, histtype="step", lw=2, color="g",
                    label="trigger scope", density=True)
        
        ax[1].legend()
        ax[1].set(xlabel="Width / (3.125 ns)", ylabel="Probability")

        fig.suptitle(self.label_cuts + " | BEFORE REMOVING BAD PERIODS")

        figname = f"c{str(self.counter).zfill(4)}m{self.module}_distributions_raw.png"

        fig.savefig(os.path.join(outdir, figname), bbox_inches="tight")
        plt.close(fig)
        print(f"{figname} saved")

        assert len(outlier_t_mask) == len(outlier_w_mask)

        # -- Bad periods cleaninig
        self.bad_periods_found_and_removed = False
        # Join the bad periods and mask the dataframe
        if outlier_t_mask.any() or outlier_w_mask.any():
            tbad = []

            assert np.array_equal(pt.x_center, pwidth.x_center), "binning is not equal between width and start times!"

            for i, (tmask, wmask) in enumerate(zip(outlier_t_mask, outlier_w_mask)):
                if tmask or wmask:
                    tbad.append((gps_bin_edges[i], gps_bin_edges[i+1]))

            # Start with all True's
            mask = np.ones(len(self.df), dtype=bool)

            # Note that the masking is on df, NOT on df_seq
            for t0, t1 in tbad:
                mask &= ~((self.df["gpsSecond"] >= t0) & (self.df["gpsSecond"] <= t1))

            self.df = self.df[mask]

            self.bad_periods_found_and_removed = True




        # -------------------------------------------------
        # Repeat the plots for the cleaned distributions:

        df_seq = self.df[self.df["entryType"] == "sequence"].copy()

        pwidth = Profile(df_seq["gpsSecond"], df_seq["width"]).doProfile(gps_bin_edges)
        pt = Profile(df_seq["gpsSecond"], df_seq["startTime"]).doProfile(gps_bin_edges)

        median_width = np.nanmedian(pwidth.y_mean)
        std_width =  np.nanquantile(pwidth.y_mean, q=0.84) - np.nanquantile(pwidth.y_mean, q=0.16)
        std_width /= 2

        median_t = np.nanmedian(pt.y_mean)
        std_t =  np.nanquantile(pt.y_mean, q=0.84) - np.nanquantile(pt.y_mean, q=0.16)
        std_t /= 2

        # -- Width vs time
        fig, ax = plt.subplots(1, 1, figsize=(14, 7))
        
        ax.errorbar(pwidth.x_mean, pwidth.y_mean, yerr=pwidth.y_std_of_mean, fmt="o", ms=ms)
        ax.axhline(y=median_width, ls="--", color="k", alpha=0.8, lw=2, label=f"median = {median_width:.1f}, std = {std_width:.1f}")
        ax.axhspan(ymin=median_width - 3*std_width, ymax=median_width + 3*std_width, color="grey", alpha=0.5)
        

        ax.set_xlabel("GPS second")
        ax.set_ylabel(r"$\langle \text{width} \, / \, (3.125 \text{ ns})\rangle $")

        if gps_uub_mark is not None:
            ax.axvline(x=gps_uub_mark, ls="-.", lw=2, color="r", label="UUB transition")

        ax.legend(fontsize="medium")

        # secondary axis with UTC date times
        sync_gps_utc_axes(ax=ax)

        figname = f"c{str(self.counter).zfill(4)}m{self.module}_width_vs_gps_clean.png"

        fig.savefig(os.path.join(outdir, figname), bbox_inches="tight")
        plt.close(fig)
        print(f"{figname} saved")

         # -- Start time vs time
        fig, ax = plt.subplots(1, 1, figsize=(14, 7))
        
        ax.errorbar(pt.x_mean, pt.y_mean, yerr=pt.y_std_of_mean, fmt="o", ms=ms)
        ax.axhline(y=median_t, ls="--", color="k", alpha=0.8, lw=2, label=f"median = {median_t:.1f}, std = {std_t:.1f}")
        ax.axhspan(ymin=median_t - 3*std_t, ymax=median_t + 3*std_t, color="grey", alpha=0.5)
        

        ax.set_xlabel("GPS second")
        ax.set_ylabel(r"$\langle \text{t} \, / \, (3.125 \text{ ns})\rangle $")

        if gps_uub_mark is not None:
            ax.axvline(x=gps_uub_mark, ls="-.", lw=2, color="r", label="UUB transition")
            
        ax.legend(fontsize="medium")

        # secondary axis with UTC date times
        sync_gps_utc_axes(ax=ax)

        figname = f"c{str(self.counter).zfill(4)}m{self.module}_start_time_vs_gps_clean.png"


        fig.savefig(os.path.join(outdir, figname), bbox_inches="tight")
        plt.close(fig)
        print(f"{figname} saved")

        # -- Distributions after removing bad periods
        # (i) Start times
        # (ii) Widths for different time windows (noise early, late and trigger scope)
        
        fig, ax = plt.subplots(1, 2, figsize=(17, 7))
        
        tbinedges = np.arange(0, self.trace_length+2, step=1) - 0.5
        ax[0].hist(df_seq["startTime"], bins=tbinedges, lw=2, histtype="step")
        
        ax[0].axvspan(xmin=self.noise_early[0], xmax=self.noise_early[1], alpha=0.1, color="grey")
        ax[0].axvspan(xmin=self.noise_late[0], xmax=self.noise_late[1], alpha=0.1, color="m")
        ax[0].axvspan(xmin=self.trigger_scope[0], xmax=self.trigger_scope[1], alpha=0.1, color="g")
        
        ax[0].set(xlabel="Start time / (3.125 ns)", ylabel="Counts", yscale="log")


        wbinedges = np.arange(0, 20, step=1) - 0.5
        noise_early_mask = (self.noise_early[0] < df_seq["startTime"]) & (df_seq["startTime"] < self.noise_early[1])
        noise_late_mask = (self.noise_late[0] < df_seq["startTime"]) & (df_seq["startTime"] < self.noise_late[1])
        trigger_scope_mask = (self.trigger_scope[0] < df_seq["startTime"]) & (df_seq["startTime"] < self.trigger_scope[1])

        ax[1].hist(df_seq[noise_early_mask]["width"], bins=wbinedges, histtype="step", lw=2, color="grey",
                    label="noise early", density=True)
        ax[1].hist(df_seq[noise_late_mask]["width"], bins=wbinedges, histtype="step", lw=2, color="m",
                    label="noise late", density=True)
        ax[1].hist(df_seq[trigger_scope_mask]["width"], bins=wbinedges, histtype="step", lw=2, color="g",
                    label="trigger scope", density=True)
        
        ax[1].legend()
        ax[1].set(xlabel="Width / (3.125 ns)", ylabel="Probability")

        fig.suptitle(self.label_cuts + f" | AFTER REMOVING BAD PERIODS. bad_periods_found_and_removed = {self.bad_periods_found_and_removed}")

        figname = f"c{str(self.counter).zfill(4)}m{self.module}_distributions_clean.png"

        fig.tight_layout(pad=1.1)

        fig.savefig(os.path.join(outdir, figname), bbox_inches="tight")
        plt.close(fig)
        print(f"{figname} saved")

        
       
    def exploratory_plots(self, kmax, outdir):
        """
        Generate exploratory diagnostic plots to study detector response
        and background behavior as a function of geometry, energy, and time.

        This method performs a multi-dimensional exploratory analysis of
        reconstructed sequence objects, focusing on how sequence width
        (proxy for strip multiplicity / detector response) depends on:

        - Zenith angle (secθ)
        - Energy (log10(E/eV))
        - Time within the readout window
        - Trigger category (e.g. WCD trigger type)
        - Event topology (short vs long patterns)

        The analysis is restricted to sequences with:
        totalChannelsOn <= kmax

        and is typically applied after bad-period removal.

        Physics motivation
        ------------------
        The goal is to probe both:

        1. Background stability:
        - Compare early/late noise windows
        - Check stability of width vs geometry and energy
        - Identify possible detector or environmental biases

        2. Single-muon response structure:
        - Separate short patterns (width < 4) from muon-like signals
        - Study how width scales with secθ and energy
        - Investigate trigger-dependent effects on reconstructed sequences

        These observables are essential to constrain:
        - Detector response functions
        - Muon multiplicity interpretation
        - Separation between noise and true muon-induced activity

        Parameters
        ----------
        kmax : int
            Maximum number of activated channels (totalChannelsOn) allowed
            in the selection. Controls event topology and pileup contamination.
        outdir : str
            Directory where all diagnostic plots are saved.

        Raises
        ------
        ValueError
            If bad-period removal has not been performed prior to calling
            this method (exclude_bad_periods must be executed first).

        Notes
        -----
        - Requires precomputed noise_early, noise_late, and trigger_scope windows.
        - Assumes dataframe contains at least:
        width, startTime, theta, lgEnergy, isT4, wcdTrigger.
        - Uses Profile objects to compute binned conditional means.
        - Produces multiple diagnostic figures used for validation of:
        * geometric scaling
        * energy dependence
        * temporal stability
        * trigger-dependent biases
        """

        if self.bad_periods_found_and_removed is None:
            raise ValueError("You need to run exclude_bad_periods() first!")

        df_seq = self.df[self.df["entryType"] == "sequence"].copy()
        df_empty = self.df[self.df["entryType"] == "module_empty"].copy()

        assert len(df_seq) + len(df_empty) == len(self.df)

        # cut in kmax (totalChannelsOn)
        df_seq = df_seq[df_seq["totalChannelsOn"] <= kmax]
        
        fig, ax = plt.subplots(1, 2, figsize=(17, 7))
        ms = 10
        
        df_seq_t4 = df_seq[df_seq["isT4"] == 1]

        noise_early_mask = (self.noise_early[0] < df_seq_t4["startTime"]) & (df_seq_t4["startTime"] < self.noise_early[1])
        noise_late_mask = (self.noise_late[0] < df_seq_t4["startTime"]) & (df_seq_t4["startTime"] < self.noise_late[1])
        trigger_scope_mask = (self.trigger_scope[0] < df_seq_t4["startTime"]) & (df_seq_t4["startTime"] < self.trigger_scope[1])

        df_seq_noise_early = df_seq_t4[noise_early_mask]
        df_seq_noise_late = df_seq_t4[noise_late_mask]
        df_seq_trigger_scope = df_seq_t4[trigger_scope_mask]

        sec_theta_bin_edges = np.arange(1, 2, step=0.1)

        # ------------------------------------------------
        # profile of width vs sec theta for noise windows

        p_early = Profile(1/np.cos(np.deg2rad(df_seq_noise_early["theta"])), df_seq_noise_early["width"]).doProfile(sec_theta_bin_edges)
        p_late = Profile(1/np.cos(np.deg2rad(df_seq_noise_late["theta"])), df_seq_noise_late["width"]).doProfile(sec_theta_bin_edges)

        ax[0].errorbar(p_early.x_mean, p_early.y_mean, yerr=p_early.y_std_of_mean, fmt="o", mfc="None", ms=ms,
                       label="Noise early")
        ax[0].errorbar(p_late.x_mean, p_late.y_mean, yerr=p_late.y_std_of_mean, fmt="^", alpha=0.8, ms=ms,
                       label="Noise late")
        
        ax[0].legend(fontsize="large")
        ax[0].set(xlabel=r"$\sec \theta$", ylabel=r"$\langle \text{Width} \rangle \, / \, \text{3.125 ns}$")
        ax[0].axvline(x=1/np.cos(np.deg2rad(45)), ls="--", color="k")

        # ------------------------------------------------
        # profile of width vs sec theta for trigger scope
        p_trigger = Profile(1/np.cos(np.deg2rad(df_seq_trigger_scope["theta"])), df_seq_trigger_scope["width"]).doProfile(sec_theta_bin_edges)

        ax[1].errorbar(p_trigger.x_mean, p_trigger.y_mean, yerr=p_trigger.y_std_of_mean, fmt="o",  ms=ms,
                       label="Trigger scope")
        ax[1].legend(fontsize="large")
        ax[1].set(xlabel=r"$\sec \theta$", ylabel=r"$\langle \text{Width} \rangle \, / \, \text{3.125 ns}$")
        ax[1].axvline(x=1/np.cos(np.deg2rad(45)), ls="--", color="k")

        fig.suptitle(self.label_cuts + f" . kmax = {kmax}. isT4 = True")

        figname = f"c{str(self.counter).zfill(4)}m{self.module}_width_vs_theta_kmax{kmax}.png"

        fig.tight_layout(pad=1.1)

        fig.savefig(os.path.join(outdir, figname), bbox_inches="tight")
        plt.close(fig)
        print(f"{figname} saved")


        # ------------------------------------------------
        # Width vs lgE for noise windows
        fig, ax = plt.subplots(1, 2, figsize=(17, 7))

        lge_bin_edges = [17.3, 17.5, 17.7, 17.9, 18.1, 18.5]

        theta_max = 40
        mask_theta_early = df_seq_noise_early["theta"] < theta_max
        mask_theta_late = df_seq_noise_late["theta"] < theta_max
        mask_theta_trigger = df_seq_trigger_scope["theta"] < theta_max

        p_early = Profile(df_seq_noise_early[mask_theta_early]["lgEnergy"], df_seq_noise_early[mask_theta_early]["width"]).doProfile(lge_bin_edges)
        p_late = Profile(df_seq_noise_late[mask_theta_late]["lgEnergy"], df_seq_noise_late[mask_theta_late]["width"]).doProfile(lge_bin_edges)

        ax[0].errorbar(p_early.x_mean, p_early.y_mean, yerr=p_early.y_std_of_mean, fmt="o", mfc="None", ms=ms,
                       label="Noise early")
        ax[0].errorbar(p_late.x_mean, p_late.y_mean, yerr=p_late.y_std_of_mean, fmt="^", alpha=0.8, ms=ms,
                       label="Noise late")
        
        ax[0].legend(fontsize="large")
        ax[0].set(xlabel=r"$\lg(E \, / \, \text{eV})$", ylabel=r"$\langle \text{Width} \rangle \, / \, \text{3.125 ns}$")
        

        # ------------------------------------------------
        # profile of width vs sec theta for trigger scope
        p_trigger = Profile(df_seq_trigger_scope[mask_theta_trigger]["lgEnergy"], df_seq_trigger_scope[mask_theta_trigger]["width"]).doProfile(lge_bin_edges)

        ax[1].errorbar(p_trigger.x_mean, p_trigger.y_mean, yerr=p_trigger.y_std_of_mean, fmt="o",  ms=ms,
                       label="Trigger scope")
        ax[1].legend(fontsize="large")
        ax[1].set(xlabel=r"$\lg(E \, / \, \text{eV})$", ylabel=r"$\langle \text{Width} \rangle \, / \, \text{3.125 ns}$")
        
        fig.suptitle(self.label_cuts + f" . kmax = {kmax}. isT4 = True. theta < {theta_max:.0f} º")

        figname = f"c{str(self.counter).zfill(4)}m{self.module}_width_vs_lgE_kmax{kmax}.png"

        fig.tight_layout(pad=1.1)

        fig.savefig(os.path.join(outdir, figname), bbox_inches="tight")
        plt.close(fig)
        print(f"{figname} saved")


        ## Looking at the start Time distribution of short patterns (w < 4)
        # and long (w >= 4) patterns

        fig, ax = plt.subplots(1, 1, figsize=(14, 7))

        df_short = df_seq[df_seq["width"] < 4]
        df_long = df_seq[df_seq["width"] >= 4]

        tbinedges = np.arange(0, self.trace_length+2, step=1) - 0.5

        ax.hist(df_seq["startTime"], lw=2, histtype="step", label="all", bins=tbinedges)
        ax.hist(df_short["startTime"], lw=2, histtype="stepfilled", label="short patterns (width < 4)", color="grey",
                alpha=0.6, bins=tbinedges)
        ax.hist(df_long["startTime"], lw=2, histtype="step", label="long patterns (width >= 4)", color="g", bins=tbinedges)

        ax.set_xlabel("start time / (3.125 ns)")
        ax.legend()
        ax.set_ylabel("Counts")
        ax.set_yscale("log")

        fig.suptitle(self.label_cuts + f" . kmax = {kmax}")

        figname = f"c{str(self.counter).zfill(4)}m{self.module}_time_patterns_kmax{kmax}.png"

        fig.tight_layout(pad=1.1)

        fig.savefig(os.path.join(outdir, figname), bbox_inches="tight")
        plt.close(fig)
        print(f"{figname} saved")

        # --------------------------------------------
        # time distribution of different SD triggers
        fig, ax = plt.subplots(1, 1, figsize=(14, 7))
        
        for trigg, df_trigg in df_seq.groupby("wcdTrigger"):
            lbl = f"{trigg}"

            ax.hist(df_trigg["startTime"], lw=2, histtype="step", label=lbl, bins=tbinedges)
            
        ax.set_xlabel("start time / (3.125 ns)")
        ax.legend()
        ax.set_ylabel("Counts")
        ax.set_yscale("log")

        fig.suptitle(self.label_cuts + f" . kmax = {kmax}")

        figname = f"c{str(self.counter).zfill(4)}m{self.module}_time_wcdTriggers_kmax{kmax}.png"

        fig.tight_layout(pad=1.1)

        fig.savefig(os.path.join(outdir, figname), bbox_inches="tight")
        plt.close(fig)
        print(f"{figname} saved")


    def estimate_background_rate(self, outdir, isT4=None, isT5=None, kmax=64):
        """
        Estimate background rates of sequences and muon-like patterns in early/late noise windows.

        This method computes the rate of occurrence of:
        (i) generic sequences (entryType == "sequence"), and
        (ii) muon-like patterns defined as sequences with width >= 4,

        in two noise time windows ("early" and "late"). The rates are normalized by the
        total number of events and the corresponding window durations.

        In addition, it:
        - Builds multiplicity distributions (number of sequences/patterns per event)
        in both noise windows.
        - Compares observed distributions against Poisson expectations.
        - Includes both cases: with and without "empty module" events.
        - Produces diagnostic plots and saves them to disk.
        - Performs internal consistency checks between rate-based and mean-based estimators.

        Parameters
        ----------
        outdir : str
            Output directory where all generated plots will be saved.

        isT4 : int or None, optional
            If 0 or 1, applies a selection on column "isT4". If None, no cut is applied.

        isT5 : int or None, optional
            If 0 or 1, applies a selection on column "isT5". If None, no cut is applied.

        kmax : int, optional
            Maximum allowed value of "totalChannelsOn". Events with higher occupancy
            are removed before analysis (default is 64).

        Returns
        -------
        None
            Results are stored internally in:
            - self.rate_seq_bkg
            - self.rate_muon_pattern_bkg

            and multiple diagnostic figures are written to `outdir`.

        Notes
        -----
        - The analysis assumes Poisson statistics for sequence multiplicities.
        - Early/late windows are defined by self.noise_early and self.noise_late.
        - Consistency checks ensure equivalence between rate-based and mean-based estimators.
        """

        df = self.df.copy()

        df = df[df["totalChannelsOn"] <= kmax]
       

        if isT4 is not None:
            assert isT4 in {0, 1}, "isT4 must be either 0 or 1"

            df  = df[df["isT4"] == 1]
                   
        if isT5 is not None:
            assert isT5 in {0, 1}, "isT5 must be either 0 or 1"

            df = df[df["isT5"] == 1]
        

        df_seq = df[df["entryType"] == "sequence"].copy()
        df_empty = df[df["entryType"] == "module_empty"].copy()

        dt_early = self.noise_early[1] - self.noise_early[0]
        dt_late = self.noise_late[1] - self.noise_late[0]

        # nr of events in which this module was in the event, but without any positive sample
        # this is equal to len(df_empty["eventId"].unique()), see sanity_checks()
        n_empty = len(df_empty)

        # --- Vectorized counting of sequences per event in each time window ---
        # Instead of looping over events, we:
        # 1) Build a boolean mask over ALL rows indicating whether each sequence
        #    falls inside the time window (True/False).
        # 2) Use groupby("eventId") and sum() to count how many True values each
        #    event has. Since True=1 and False=0, the sum gives the number of
        #    sequences in the window per event.
        # This is equivalent to looping over events and counting manually, but
        # much faster because it is vectorized (runs in optimized NumPy/pandas code).

        df_seq["is_early"] = (
            (self.noise_early[0] < df_seq["startTime"]) &
            (df_seq["startTime"] < self.noise_early[1])
        )

        df_seq["is_late"] = (
            (self.noise_late[0] < df_seq["startTime"]) &
            (df_seq["startTime"] < self.noise_late[1])
        )

        # seq_early[i] -> number of sequences (in the early window) in event i
        seq_early = df_seq.groupby("eventId")["is_early"].sum().to_list()
        
        seq_late  = df_seq.groupby("eventId")["is_late"].sum().to_list()
        

        # histogram of number of sequences in each window 
        fig, ax = plt.subplots(1, 2, figsize=(18, 9))
        ms = 9

        seq_bin_edges = np.arange(0, 12) - 0.5
        seq_bin_centers = seq_bin_edges[:-1] + np.diff(seq_bin_edges)/2

        assert np.allclose(seq_bin_centers, np.round(seq_bin_centers)), \
        "Non-integer bin centers: Poisson PMF would be invalid"


        mean_seq_early = np.mean(seq_early)
        mean_seq_late = np.mean(seq_late)


        # --- rate of sequences in each window 
        
        # total number of sequences in the early window
        n_tot_early = np.sum(seq_early)

        # total number of sequences in the late window
        n_tot_late = np.sum(seq_late)

        # total number of observed events (either empty or with sequence)
        n_events_unique = df["eventId"].nunique()

        rate_seq_early = n_tot_early / (n_events_unique * dt_early)
        rate_seq_early_err = np.sqrt(n_tot_early) / (n_events_unique * dt_early)

        rate_seq_late = n_tot_late / (n_events_unique * dt_late)
        rate_seq_late_err = np.sqrt(n_tot_late) / (n_events_unique * dt_late)

        self.rate_seq_bkg["early"]["rate"] = rate_seq_early 
        self.rate_seq_bkg["early"]["rate_err"] = rate_seq_early_err

        self.rate_seq_bkg["late"]["rate"] = rate_seq_late
        self.rate_seq_bkg["late"]["rate_err"] = rate_seq_late_err

      
        n_early, _ = np.histogram(seq_early, bins=seq_bin_edges)
        n_late, _ = np.histogram(seq_late, bins=seq_bin_edges)

        ax[0].errorbar(seq_bin_centers, n_early, yerr=np.sqrt(n_early), fmt="o", ms=ms, label=f"early (dt = {dt_early}), mean = {mean_seq_early:.3f}")
        ax[0].errorbar(seq_bin_centers, n_late, yerr=np.sqrt(n_late), fmt="s", ms=ms, mfc="None", label=f"late (dt = {dt_late}), mean = {mean_seq_late:.3f}")

        # Add Poisson expectation
        poission_early = len(seq_early) * poisson(mu=mean_seq_early).pmf(seq_bin_centers)
        poission_late = len(seq_late) * poisson(mu=mean_seq_late).pmf(seq_bin_centers)

        ax[0].plot(seq_bin_centers, poission_early, "-o")
        ax[0].plot(seq_bin_centers, poission_late, "--s")
        

        ax[0].set(xlabel="nr of sequences in window", ylabel="Counts", yscale="log")
        ax[0].legend()
        ax[0].set_title("W/o empty modules")
        ax[0].set_ylim(bottom=0.01)

        # Considering the empty module cases 
        # Add empty events
        seq_early_empty = seq_early + [0] * n_empty
        seq_late_empty  = seq_late  + [0] * n_empty     

        mean_seq_early_empty = np.mean(seq_early_empty)
        mean_seq_late_empty = np.mean(seq_late_empty)

        n_early2, _ = np.histogram(seq_early_empty, bins=seq_bin_edges)
        n_late2, _ = np.histogram(seq_late_empty, bins=seq_bin_edges)

        ax[1].errorbar(seq_bin_centers, n_early2, yerr=np.sqrt(n_early2), fmt="o", ms=ms, label=f"early (dt = {dt_early}), mean = {mean_seq_early_empty:.3f}")
        ax[1].errorbar(seq_bin_centers, n_late2, yerr=np.sqrt(n_late2), fmt="s", ms=ms, mfc="None", label=f"late (dt = {dt_late}), mean = {mean_seq_late_empty:.3f}")


        # Add Poisson expectation
        poission_early2 = len(seq_early_empty) * poisson(mu=mean_seq_early_empty).pmf(seq_bin_centers)
        poission_late2 = len(seq_late_empty) * poisson(mu=mean_seq_late_empty).pmf(seq_bin_centers)

        ax[1].plot(seq_bin_centers, poission_early2, "-o")
        ax[1].plot(seq_bin_centers, poission_late2, "--s")

        ax[1].set(xlabel="nr of sequences in window", ylabel="Counts", yscale="log")
        ax[1].legend()
        ax[1].set_title("W/ empty modules")
        ax[1].set_ylim(bottom=0.01)

        fig.suptitle(self.label_cuts + f". rate_seq_early = {rate_seq_early:.5f} (3.125 ns)$^{-1}$. rate_seq_late = {rate_seq_late:.5f} (3.125 ns)$^{-1}$" + "\n"
                    + f"isT4 = {isT4}, isT5 = {isT5}, kmax = {kmax}")

        figname = f"c{str(self.counter).zfill(4)}m{self.module}_nr_sequences.png"

        fig.tight_layout(pad=1.1)

        fig.savefig(os.path.join(outdir, figname), bbox_inches="tight")
        plt.close(fig)
        print(f"{figname} saved")

        assert np.isclose(rate_seq_early, mean_seq_early_empty / dt_early)
        assert np.isclose(rate_seq_late,  mean_seq_late_empty  / dt_late)

        # nr of sequences per event (considering empty modules) normalized to probability
        fig, ax = plt.subplots(1, 1, figsize=(16, 9))

        ax.hist(seq_early_empty, bins=seq_bin_edges, histtype="step", density=True,
                label="Early", color="b")
        ax.plot(seq_bin_centers, poisson(mu=mean_seq_early_empty).pmf(seq_bin_centers),
                "-o", color="b", label="Poisson (expected early)")

        ax.hist(seq_late_empty, bins=seq_bin_edges, histtype="step", density=True,
                label="Late", color="r")
        ax.plot(seq_bin_centers, poisson(mu=mean_seq_late_empty).pmf(seq_bin_centers),
                "--s", color="r", label="Poisson (expected late)")
        
        ax.set(xlabel="nr of sequences in window", ylabel="Probability", yscale="log")
        ax.legend()
        ax.set_ylim(bottom=0.001)

        fig.suptitle(self.label_cuts + f". rate_seq_early = {rate_seq_early:.5f} (3.125 ns)$^{-1}$. rate_seq_late = {rate_seq_late:.5f} (3.125 ns)$^{-1}$" + "\n"
                    + f"isT4 = {isT4}, isT5 = {isT5}, kmax = {kmax}")

        figname = f"c{str(self.counter).zfill(4)}m{self.module}_nr_sequences_norm.png"

        fig.tight_layout(pad=1.1)

        fig.savefig(os.path.join(outdir, figname), bbox_inches="tight")
        plt.close(fig)
        print(f"{figname} saved")

        # ---------------------
        # Number of muon patterns (sequences >=4) in noise/late windows

        df_pattern = df_seq[df_seq["width"] >= 4].copy()

        df_pattern["is_early"] = (
            (self.noise_early[0] < df_pattern["startTime"]) &
            (df_pattern["startTime"] < self.noise_early[1])
        )

        df_pattern["is_late"] = (
            (self.noise_late[0] < df_pattern["startTime"]) &
            (df_pattern["startTime"] < self.noise_late[1])
        )

        # pattern_early[i] -> number of muon patterns (in the early window) in event i
        pattern_early = df_pattern.groupby("eventId")["is_early"].sum().to_list()
        pattern_late  = df_pattern.groupby("eventId")["is_late"].sum().to_list()

        # just printing those events with many late patterns 
        event_ids_unique = df_pattern["eventId"].unique()
        #events_fishy = event_ids_unique[np.asarray(pattern_late) > 5]

        #print(" Events with more than 5 patterns late:")
        #print(events_fishy)

        fig, ax = plt.subplots(1, 2, figsize=(18, 9))

        mean_pattern_early = np.mean(pattern_early)
        mean_pattern_late = np.mean(pattern_late)

        # --- rate of muon patterns in each window 
        
        # total number of muon patterns in the early window
        n_patterns_tot_early = np.sum(pattern_early)

        # total number of sequences in the late window
        n_patterns_tot_late = np.sum(pattern_late)

        # total number of observed events (either empty or with sequence)
        n_events_unique = df["eventId"].nunique()

        # Note that this rate is the one expected in the whole module -> Lambda = 64*lam_1 = rate_patterns_ * 2048
        rate_patterns_early = n_patterns_tot_early / (n_events_unique * dt_early)
        rate_patterns_early_err = np.sqrt(n_patterns_tot_early) / (n_events_unique * dt_early)

        rate_patterns_late = n_patterns_tot_late / (n_events_unique * dt_late)
        rate_patterns_late_err = np.sqrt(n_patterns_tot_late) / (n_events_unique * dt_late)

        self.rate_muon_pattern_bkg["early"]["rate"] = rate_patterns_early
        self.rate_muon_pattern_bkg["early"]["rate_err"] = rate_patterns_early_err

        self.rate_muon_pattern_bkg["late"]["rate"] = rate_patterns_late
        self.rate_muon_pattern_bkg["late"]["rate_err"] = rate_patterns_late_err

      
        n_early, _ = np.histogram(pattern_early, bins=seq_bin_edges)
        n_late, _ = np.histogram(pattern_late, bins=seq_bin_edges)

        ax[0].errorbar(seq_bin_centers, n_early, yerr=np.sqrt(n_early), fmt="o", ms=ms, label=f"early (dt = {dt_early}), mean = {mean_pattern_early:.4f}")
        ax[0].errorbar(seq_bin_centers, n_late, yerr=np.sqrt(n_late), fmt="s", ms=ms, mfc="None", label=f"late (dt = {dt_late}), mean = {mean_pattern_late:.4f}")

        # Add Poisson expectation
        poission_early = len(pattern_early) * poisson(mu=mean_pattern_early).pmf(seq_bin_centers)
        poission_late = len(pattern_late) * poisson(mu=mean_pattern_late).pmf(seq_bin_centers)

        ax[0].plot(seq_bin_centers, poission_early, "-o")
        ax[0].plot(seq_bin_centers, poission_late, "--s")
        

        ax[0].set(xlabel="nr of muon patterns in window", ylabel="Counts", yscale="log")
        ax[0].legend()
        ax[0].set_title("W/o empty modules")
        ax[0].set_ylim(bottom=0.01)

        # Considering the empty module cases 
        # Add empty events
        pattern_early_empty = pattern_early + [0] * n_empty
        pattern_late_empty  = pattern_late  + [0] * n_empty     

        mean_pattern_early_empty = np.mean(pattern_early_empty)
        mean_pattern_late_empty = np.mean(pattern_late_empty)

       
        n_early2, _ = np.histogram(pattern_early_empty, bins=seq_bin_edges)
        n_late2, _ = np.histogram(pattern_late_empty, bins=seq_bin_edges)

        ax[1].errorbar(seq_bin_centers, n_early2, yerr=np.sqrt(n_early2), fmt="o", ms=ms, label=f"early (dt = {dt_early}), mean = {mean_pattern_early_empty:.4f}")
        ax[1].errorbar(seq_bin_centers, n_late2, yerr=np.sqrt(n_late2), fmt="s", ms=ms, mfc="None", label=f"late (dt = {dt_late}), mean = {mean_pattern_late_empty:.4f}")


        # Add Poisson expectation
        poission_early2 = len(pattern_early_empty) * poisson(mu=mean_pattern_early_empty).pmf(seq_bin_centers)
        poission_late2 = len(pattern_late_empty) * poisson(mu=mean_pattern_late_empty).pmf(seq_bin_centers)

        ax[1].plot(seq_bin_centers, poission_early2, "-o")
        ax[1].plot(seq_bin_centers, poission_late2, "--s")

        ax[1].set(xlabel="nr of muon patterns in window", ylabel="Counts", yscale="log")
        ax[1].legend()
        ax[1].set_title("W/ empty modules")
        ax[1].set_ylim(bottom=0.01)

        fig.suptitle((self.label_cuts + ". Number of muon patterns (width >=4) in noise windows.\n" + 
                      f"rate_patt_early = {rate_patterns_early:.5f} (3.125 ns)"r"$^{-1}$. "f"rate_patt_late = {rate_patterns_late:.5f} (3.125 ns)"r"$^{-1}$" + "\n"
                      + f"isT4 = {isT4}, isT5 = {isT5}, kmax = {kmax}")) 

        figname = f"c{str(self.counter).zfill(4)}m{self.module}_nr_muonPatterns.png"

        fig.tight_layout(pad=1.1)

        fig.savefig(os.path.join(outdir, figname), bbox_inches="tight")
        plt.close(fig)
        print(f"{figname} saved")


        # quick sanity check. I need to include events with sequences but not muon pattern
        # in pattern_early
        all_events = df["eventId"].unique()
        pattern_counts = (
            df_pattern.groupby("eventId")["is_early"]
            .sum()
            .reindex(all_events, fill_value=0)
        )
        pattern_early_full = pattern_counts.to_list()
        mean_pattern_early_full = np.mean(pattern_early_full)

        assert np.isclose(rate_patterns_early, mean_pattern_early_full / dt_early)

        # plot of rates 
        fig, ax = plt.subplots(1, 2, figsize=(18, 9))

        ax[0].errorbar([0, 1], [self.rate_seq_bkg["early"]["rate"], self.rate_seq_bkg["late"]["rate"]], 
                yerr=[self.rate_seq_bkg["early"]["rate_err"], self.rate_seq_bkg["late"]["rate_err"]],
                fmt="o", ms=ms)
        ax[0].set_xticks([0, 1])
        ax[0].set_xticklabels(["Early", "Late"])
        ax[0].set_label("Window")
        ax[0].set_ylabel(r"Rate / (3.125 ns)$^{-1}$")
        ax[0].set_title("Rate of sequences (of any width) in noise windows")

        ax[1].errorbar([0, 1], [self.rate_muon_pattern_bkg["early"]["rate"], self.rate_muon_pattern_bkg["late"]["rate"]], 
                yerr=[self.rate_muon_pattern_bkg["early"]["rate_err"], self.rate_muon_pattern_bkg["late"]["rate_err"]],
                fmt="o", ms=ms)
        ax[1].set_xticks([0, 1])
        ax[1].set_xticklabels(["Early", "Late"])
        ax[1].set_label("Window")
        ax[1].set_ylabel(r"Rate / (3.125 ns)$^{-1}$")
        ax[1].set_title("Rate of muon patterns (width >= 4) in noise windows")

        fig.suptitle((self.label_cuts + ". Rates of background signals.\n" + f"isT4 = {isT4}, isT5 = {isT5}, kmax = {kmax}")) 

        figname = f"c{str(self.counter).zfill(4)}m{self.module}_rate_bkg.png"

        fig.tight_layout(pad=1.1)

        fig.savefig(os.path.join(outdir, figname), bbox_inches="tight")
        plt.close(fig)
        print(f"{figname} saved")

    
    def estimate_background_prob_distribution(self, outdir):
        """
        Estimate the probability distribution of sequence widths in noise windows.

        This method computes the empirical probability distribution P(width) of
        sequences (entryType == "sequence") in early and late noise windows, and
        constructs a combined "noise model" distribution.

        The procedure includes:
        - Histogram estimation of width distributions in early, late, and combined noise windows.
        - Validation of normalization of probability distributions.
        - Study of stability of P(width) under different event selections (e.g. T4 cuts, kmax cuts).
        - Comparison of distributions under different selection criteria.
        - Sanity checks relating width distribution, event rates, and expected counts
        of short (w <= 3) and long (w >= 4) sequences.
        - Visualization of start-time distributions split by width categories.

        Parameters
        ----------
        outdir : str
            Directory where all diagnostic plots are saved.

        Returns
        -------
        None
            The method stores:
            - self.prob_width_noise : dict
                Contains the combined noise width distribution:
                {
                    "width": bin centers,
                    "prob": P(width),
                    "width_bin_edges": histogram bin edges
                }

            and writes multiple diagnostic figures to disk.

        Notes
        -----
        - The width distribution is conditioned on the presence of a sequence.
        - Early and late noise windows are combined to define the reference noise model.
        - Several cross-checks validate consistency between rate estimates and P(width).
        """

        
        # note that this would be the distribution conditioned to having a sequence..

        df_seq = self.df[self.df["entryType"] == "sequence"].copy()

        noise_early_mask = (self.noise_early[0] < df_seq["startTime"]) & (df_seq["startTime"] < self.noise_early[1])
        noise_late_mask = (self.noise_late[0] < df_seq["startTime"]) & (df_seq["startTime"] < self.noise_late[1])
        trigger_scope_mask = (self.trigger_scope[0] < df_seq["startTime"]) & (df_seq["startTime"] < self.trigger_scope[1])

        wbinedges = np.arange(0, 20, step=1) - 0.5 
        wbincenters = wbinedges[:-1] + np.diff(wbinedges)/2

        # Probability disitribution for noise windos
        p_early, _ = np.histogram(df_seq[noise_early_mask]["width"], bins=wbinedges, density=True)
        p_late, _ = np.histogram(df_seq[noise_late_mask]["width"], bins=wbinedges, density=True)
        # joining both windows -> This will be the definitive p_noise
        p_noise, _ = np.histogram(df_seq[noise_early_mask | noise_late_mask]["width"], bins=wbinedges, density=True)

        self.prob_width_noise = {"width": wbincenters, "prob": p_noise, "width_bin_edges": wbinedges} 

        # Check normalization
        assert np.isclose(np.sum(p_early), 1)
        assert np.isclose(np.sum(p_late), 1)
        assert np.isclose(np.sum(p_noise), 1)

        fig, ax = plt.subplots(1, 1, figsize=(16, 7))

        ax.errorbar(wbincenters, p_early, fmt="-o", label="Early")
        ax.errorbar(wbincenters, p_late, fmt="--s", mfc="None", label="Late")
        ax.errorbar(wbincenters, p_noise, fmt="-.^", alpha=0.9, label="Early + Late (p noise)")

        ax.set_xlabel("Width / (3.125 ns)")
        ax.set_ylabel("Probability")
        ax.legend()


        fig.suptitle((self.label_cuts + ". P(w) for noise windows (conditioned to have a sequence)")) 

        figname = f"c{str(self.counter).zfill(4)}m{self.module}_prob_width_bkg.png"

     
        fig.savefig(os.path.join(outdir, figname), bbox_inches="tight")
        plt.close(fig)
        print(f"{figname} saved")

        ## plot P(w) for different cuts (e.g. T4, T5, kmax, etc)
        # I want to confirm that the shape P(w) does not depend on those variables

        pnoise_t4, _ = np.histogram(df_seq[(noise_early_mask | noise_late_mask) & (df_seq["isT4"] == 1)]["width"], bins=wbinedges, density=True)

        kmax_values = [50, 30, 10, 5, 1]

        fig, ax = plt.subplots(1, 2, figsize=(18, 9))

        ax[0].errorbar(wbincenters, p_noise, fmt="-.^", alpha=0.9, label="Early + Late (p noise)")
        ax[0].errorbar(wbincenters, pnoise_t4, fmt="-s", mfc="None", alpha=0.9, label="T4")

        ax[1].errorbar(wbincenters, pnoise_t4 - p_noise, fmt="o", mfc="None", alpha=0.9, label="p_T4 - p_noise")

        for kmax in kmax_values:
            p , _ = np.histogram(df_seq[(noise_early_mask | noise_late_mask) & (df_seq["totalChannelsOn"] <= kmax)]["width"], bins=wbinedges, density=True)
            ax[0].errorbar(wbincenters, p, fmt="--o", mfc="None", alpha=0.7, label=f"kmax = {kmax}")

            ax[1].errorbar(wbincenters, p - p_noise, fmt="o", mfc="None", alpha=0.9, label=f"p(kmax = {kmax}) - pnoise")


        ax[0].set_xlabel("Width / (3.125 ns)")
        ax[0].set_ylabel("Probability")
        ax[0].legend(fontsize="medium")

        ax[1].set_xlabel("Width / (3.125 ns)")
        ax[1].axhline(y=0, ls="--", color="grey")
        ax[1].set_ylabel("P - Pnoise")
        ax[1].legend(fontsize="medium")


        fig.suptitle((self.label_cuts + ". P(w) for noise windows (conditioned to have a sequence)")) 

        figname = f"c{str(self.counter).zfill(4)}m{self.module}_prob_width_bkg_different_cuts.png"

        fig.tight_layout(pad=1.1)

        fig.savefig(os.path.join(outdir, figname), bbox_inches="tight")
        plt.close(fig)
        print(f"{figname} saved")


        # ----- Sanity checks
        # Using the rate and the P(w), predict how many sequences with w <= 3 I must have
        # in the events and compare it to the actual ones: histogram of startTimes with w <= 3
        # Do the same for various w<3, etc

        # Rate of sequences  due to noise
        rate = (self.rate_seq_bkg["early"]["rate"] + self.rate_seq_bkg["late"]["rate"]) / 2 

        # Expected number of noise sequences (regardless of width) in one time bin:
        expected_seq = rate * self.df["eventId"].nunique()
        
        # Expected number of noise sequencnes with w <= 3 in one time bin
        short_expected3 = (p_noise[0] + p_noise[1] + p_noise[2] + p_noise[3]) * expected_seq

        # Expected number of noise sequences with w <= 2 in one time bin
        short_expected2 = (p_noise[0] + p_noise[1] + p_noise[2]) * expected_seq

        # Expected number of noise sequences with w >= 4 in one time bin according to expected_seq and p_noise(w)
        long_expected = np.sum(p_noise[4:]) * expected_seq

        # Expected number of noise sequences with w >= 4 according to pattern_rate
        rate_muon_pattern = (self.rate_muon_pattern_bkg["early"]["rate"] +  self.rate_muon_pattern_bkg["late"]["rate"]) / 2
        long_expected_alternative = rate_muon_pattern * self.df["eventId"].nunique()


        ## ---- Plots ------
        
        tbinedges = np.arange(0, self.trace_length+2)

        fig, ax = plt.subplots(1, 1, figsize=(16, 9))

        ax.hist(df_seq["startTime"], lw=2, histtype="step", label="all", bins=tbinedges)

        ax.hist(df_seq[df_seq["width"] <= 3]["startTime"], lw=2, label="width <= 3", histtype="step", 
                bins=tbinedges, color="grey")
        
        ax.hist(df_seq[df_seq["width"] <= 2]["startTime"], lw=2, label="width <= 2", histtype="stepfilled", 
                bins=tbinedges, color="orange", alpha=0.5)
        
        ax.hist(df_seq[df_seq["width"] >= 4]["startTime"], lw=2, label="width >= 4", histtype="step", 
                bins=tbinedges, color="g", alpha=0.5)

        ax.axhline(y=expected_seq, ls="--", color="k", label="expected number of noise sequences")
        ax.axhline(y=short_expected3, ls="-.", color="grey", label="expected number of noise sequences with w <= 3")
        ax.axhline(y=short_expected2, ls="-.", color="r", label="expected number of noise sequences with w <= 2")

        ax.axhline(y=long_expected, ls="-.", color="g", label="expected number of noise sequences with w >= 4")
        ax.axhline(y=long_expected_alternative, ls="--", color="lightgreen", label="expected number of noise sequences with w >= 4 (according to rate_pattern)")

        ax.set_xlabel("Start time / (3.125) ns")
        ax.set_ylabel("Counts")
        ax.set_yscale("log")
        ax.legend(fontsize="medium")

        fig.suptitle((self.label_cuts)) 

        figname = f"c{str(self.counter).zfill(4)}m{self.module}_start_time_checks.png"

        fig.tight_layout(pad=1.1)

        fig.savefig(os.path.join(outdir, figname), bbox_inches="tight")
        plt.close(fig)
        print(f"{figname} saved")

    

    def get_selected_events(self, apply_test=False):
        """
        Select valid events containing one or two scintillator bars with exactly one
        sequence in the trigger time window.

        This function identifies events suitable for analysis by applying a set of
        selection criteria on sequence-level data. The goal is to retain events with
        a clean topology in which each scintillator bar contains at most one sequence
        within the trigger scope, while rejecting ambiguous or multi-hit bars.

        Selection procedure:
        1. Select sequence entries within the trigger time window.
        2. Group by event and scintillator to count sequences per bar.
        3. Reject events containing any bar with more than one sequence in the window.
        4. Keep bars with exactly one sequence and remove those belonging to invalid events.
        5. Count valid bars per event and retain events with 1 or 2 valid bars.
        6. Label events as:
            - "one_seq" for events with 1 valid bar
            - "two_seq" for events with 2 valid bars

        Parameters
        ----------
        apply_test : bool, optional
            If True, runs an internal consistency check comparing the optimized
            implementation against a slower reference version (`get_selected_events_test`).

        Returns
        -------
        valid_event_ids : list
            List of selected event identifiers passing all selection criteria.
        event_type : list of str
            Event classification labels ("one_seq" or "two_seq") aligned with
            `valid_event_ids`.

        Notes
        -----
        - This function enforces strict purity conditions on scintillator activity
        within the trigger window.
        - It is optimized for performance using pandas groupby operations.
        - The optional test mode ensures equivalence with a slower, explicit loop-based
        implementation.
        """

        tmin = self.trigger_scope[0]
        tmax = self.trigger_scope[1]

        df_seq = self.df[self.df["entryType"] == "sequence"].copy()

        # Selecting sequences within the trigger scope window
        df_win = df_seq[
            (df_seq["startTime"] > tmin) &
            (df_seq["startTime"] < tmax)
        ]

        
        counts_per_bar = (
            df_win
            .groupby(["eventId", "scintillatorId"])
            .size()
            .rename("n_seq")
            .reset_index()
        )

        # eventos con barras inválidas (ie, con mas de 1 sequence en el trigger scope)
        invalid_events = counts_per_bar[
            counts_per_bar["n_seq"] > 1
        ]["eventId"].unique()

        
        # barras válidas (solo 1 seq) y eventos limpios
        valid_bars = counts_per_bar[
            (counts_per_bar["n_seq"] == 1) &
            (~counts_per_bar["eventId"].isin(invalid_events))
        ]

        bars_per_event = (
            valid_bars
            .groupby("eventId")["scintillatorId"]
            .nunique()
        )

        # ----

        # keep only valid events (1 or 2 bars with single sequence in trigger scope)
        selected = bars_per_event[
            (bars_per_event >= 1) &
            (bars_per_event <= 2)
        ]

        # event ids
        valid_event_ids = selected.index.tolist()

        # event type labels aligned with event_ids
        event_type = [
            "one_seq" if selected.loc[evt] == 1 else "two_seq"
            for evt in valid_event_ids
        ]

        # just to test, I run an alternative (slower) way to compute valid events
        # It succeded! :)
        if apply_test:
            selected_events_test, event_type_test = self.get_selected_events_test()

            assert np.array_equal(selected_events_test, valid_event_ids), "mismatch!"
            assert np.array_equal(event_type_test, event_type), "mismatch!"

            print("Test succeded!")

            #print(" Valid events for the analysis: ")
            #for evt, tt in  zip(valid_event_ids, event_type):
                #print(evt, tt)


        return valid_event_ids, event_type

    def get_selected_events_test(self):
        """
        Reference implementation for event selection using an explicit loop-based approach.

        This function provides a slower but more transparent version of
        `get_selected_events`. It is intended exclusively for validation purposes
        to ensure correctness of the optimized implementation.

        The algorithm iterates over events and scintillator bars, applying the same
        selection criteria as the main method:

        Selection criteria:
        - Only sequences within the trigger time window are considered.
        - Each scintillator bar must contain at most one sequence in the window.
        - Events are rejected if any bar contains more than one sequence.
        - Valid events must contain between 1 and 2 scintillator bars with exactly
        one sequence in the trigger window.

        Events are classified as:
            - "one_seq" if exactly one valid bar is present
            - "two_seq" if exactly two valid bars are present

        Returns
        -------
        valid_events : list
            List of event identifiers passing the selection.
        event_type : list of str
            Corresponding event labels ("one_seq" or "two_seq").

        Notes
        -----
        - This implementation prioritizes clarity over performance.
        - It is used as a cross-check for the optimized pandas-based version.
        - It should not be used in production analysis due to computational cost.
        """
        
        tmin, tmax = self.trigger_scope

        df_seq = self.df[self.df["entryType"] == "sequence"].copy()
        # Ahora deberia meter algun corte adicional, tipo T4, para poder
        # estudiar dependencia con theta, y phi

        # valid events are those in which there is only one or two bars with sequence WITHIN THE TRIGGER SCOPE
        # in the module
        # In addition, each bar must have only ONE sequence in the trigger scope.
        valid_events = []
        event_type = []

        for event, df_event in df_seq.groupby("eventId"):

            valid_bars = 0
            event_is_valid = True

            for scint, df_strip in df_event.groupby("scintillatorId"):

                # filtrar por ventana SOLO en esta barra
                in_window = df_strip[
                    (df_strip["startTime"] > tmin) &
                    (df_strip["startTime"] < tmax)
                ]

                # numbero de secuencias en trigger scope de esta barra
                n_seq = len(in_window)

                if n_seq > 1:
                    event_is_valid = False
                    break

                if n_seq == 1:
                    valid_bars += 1

            # aplicar condición final
            if event_is_valid and (1 <= valid_bars <= 2):
                valid_events.append(event)
                if valid_bars == 1:
                    event_type.append("one_seq")
                else:
                    event_type.append("two_seq")


        return valid_events, event_type


    def analyze_single_bar_events(self, lge_min, outdir):
        """
        Analyze single-bar events and extract muon inefficiency via a mixture model fit.
        This is relevant to estimate single-muon events that produce a sequence only in 
        one bar. 

        This function selects events with exactly one active sequence ("one_seq")
        within a given trigger time window and applies basic physics cuts. It then
        builds the distribution of signal widths and fits it with a mixture model:

            P_model(w) = f * P_noise(w) + (1 - f) * P_muon(w)

        where:
            - P_noise(w) is a precomputed noise template
            - P_muon(w) is modeled as a discretized Student-t distribution

        From the fitted model, the function estimates:
            - total_muons: total number of muon events
            - ineff_muons: number of muons with width <= 3 (inefficient signals)
            - ratio: ineff_muons / total_muons

        Uncertainties on these quantities are obtained via Monte Carlo sampling
        using the covariance matrix of the fit parameters.

        Additionally, diagnostic plots are produced:
            - Fit to the width distribution (data vs model components)
            - Residuals (pull and relative difference)

        Parameters
        ----------
        lge_min : float
            Minimum log-energy threshold used to select physical events.
        outdir : str
            Output directory where plots are saved.

        Raises
        ------
        ValueError
            If the noise width distribution (`self.prob_width_noise`) has not
            been initialized.

        Notes
        -----
        - Only events passing T4 trigger and energy cuts are considered.
        - The fit is restricted to a predefined width range (excluding w=0).
        - A Student-t model is used for the muon component to better capture
        non-Gaussian tails in the width distribution.
        - If the covariance matrix is invalid or ill-conditioned, uncertainties
        are not reported (flagged via `bad_fit`).
        - The normalization of the muon component is checked for consistency
        during Monte Carlo sampling.

        Outputs
        -------
        Saves two figures to `outdir`:
            - Mixture fit (data, total model, noise, and muon components)
            - Residuals (pull and relative residuals)

        Prints
        ------
        Filenames of the generated figures.
        """

        if self.prob_width_noise is None:
            raise ValueError("You must first fill self.prob_width_noise")

        tmin, tmax = self.trigger_scope
        dt = tmax - tmin

        # ------------------------------------------------------------
        # 1. Select events with 1 or 2 active bars in trigger window
        # ------------------------------------------------------------
        valid_events, event_type = self.get_selected_events(apply_test=False)

        valid_event_df = pd.DataFrame({
            "eventId": valid_events,
            "event_type": event_type
        })

        df_valid = self.df[self.df["eventId"].isin(valid_events)].copy()

        assert np.array_equal(df_valid["eventId"].unique(), valid_event_df["eventId"].unique()), "mismatch!"

        # Attach event classification ("one_seq" or "two_seq")
        df_valid = df_valid.merge(valid_event_df, on="eventId", how="left")

        # Apply physics selection cuts (independent of noise model)
        df_valid = df_valid[
            (df_valid["isT4"] == 1) &
            (df_valid["lgEnergy"] > lge_min)
        ]

        # ------------------------------------------------------------
        # 2. Keep only events with exactly one sequence in trigger window
        # ------------------------------------------------------------
        single_bar_all = df_valid[df_valid["event_type"] == "one_seq"]

        assert np.isfinite(self.df["lgEnergy"]).all(), "strange"
        assert np.isfinite(single_bar_all["lgEnergy"]).all(), "strange"

        # Restrict to sequences inside trigger window
        single_bar_trigger = single_bar_all[
            (tmin < single_bar_all["startTime"]) &
            (single_bar_all["startTime"] < tmax)
        ]


        # Observed histogram of widths (taking the same binning in width as used for the noise)
        counts_data, _ = np.histogram(single_bar_trigger["width"], bins=self.prob_width_noise["width_bin_edges"])
        counts_data_err = np.sqrt(counts_data)
        counts_data_err[counts_data_err == 0] = 1.0

        n_data_tot = np.sum(counts_data)
        w_data = np.asarray(self.prob_width_noise["width"])
        p_noise = np.asarray(self.prob_width_noise["prob"])
        assert len(w_data) == len(self.prob_width_noise["width_bin_edges"])-1
        
        # Restricting fitting range (I dont want to use w=0 bc, by construction, my data has w=0)
        w_min, w_max = 1, 13
        mask_fit = (w_data >= w_min) & (w_data <= w_max)
       
        # ------------------------------------------------------------
        # 3. Fit a model P_model = f * P_noise_template(w) + (1-f)* P_muons(w),
        #  where P_muons(w) is a t-student. NOTE I tried Gaussian, but it fails to
        # fit the tails
        # ------------------------------------------------------------
        def P_muon_student_t(w, mu, sigma, nu):
            """
            Student-t PDF evaluada en bins discretos.
            """
            pdf = student_t.pdf(w, df=nu, loc=mu, scale=sigma)
            return pdf / np.sum(pdf)
            

        # Least square fit of the mixture model
        def residuals_t(params):
            f, mu, sigma, nu = params 
            p_muon = P_muon_student_t(w_data[mask_fit], mu, sigma, nu)
            
            model = f * p_noise[mask_fit] + (1 - f) * p_muon
            
            return (counts_data[mask_fit] - model * n_data_tot) / counts_data_err[mask_fit]
        
         
        p0 = [
            0.2,   # f noise
            7.0,   # mu muon
            2.0,   # sigma
            4.0,  # nu
        ]

        bounds = (
            [0.0, 0.0, 0.1, 1],   # lower
            [1.0, 20.0, 10.0, 60] # upper
        )


        #result = least_squares(residuals, p0, bounds=bounds)
        result = least_squares(
                residuals_t,
                p0,
                bounds=bounds,
                method="trf",
                max_nfev=10000,
                x_scale='jac',
                ftol=1e-10,
                xtol=1e-10,
                gtol=1e-10
            )

        # Best-fit result
        f_fit, mu_fit, sigma_fit, nu_fit = result.x

        # Goodness of fit
        chi2_obs = np.sum(residuals_t(result.x)**2)
        ndof = np.sum(mask_fit) - len(result.x)
        chi2_red = chi2_obs / ndof
        pvalue = chi2(df=ndof).sf(chi2_obs)

        # Covariance matrix and errors        
        J = result.jac
        bad_fit = False
        try:
            cov = np.linalg.inv(J.T @ J)
            # Scale covariance matrix by reduced chi2
            cov *= chi2_red
            errors = np.sqrt(np.diag(cov))

            if not self.is_valid_covariance(cov):
                bad_fit = True

        except np.linalg.LinAlgError:
            errors = np.array([-1, -1, -1, -1])
            bad_fit = True
            cov = None
        
    
        # Unpack errors
        f_fit_err, mu_fit_err, sigma_fit_err, nu_fit_err = errors

        # Estimate inefficient muons, total muons and their ratio from the fit
        counts_model_muons = n_data_tot * (1 - f_fit) * P_muon_student_t(w_data, mu_fit, sigma_fit, nu_fit)
        counts_model_noise = n_data_tot * f_fit * p_noise
        model = counts_model_noise + counts_model_muons

        ineff_muons = np.sum(counts_model_muons[w_data <= 3])
        total_muons = np.sum(counts_model_muons)
        ratio = ineff_muons / total_muons

        # Estimate error on inefficient muons, total muons and ratio with MonteCarlo sampling
        errors_ineff = self.errors_monte_carlo_sampling(best_fit_param=result.x, cov_matrix=cov, n_data_tot=n_data_tot, 
                                P_muon_student_t=P_muon_student_t, bad_fit=bad_fit, n_toys=2000)


        if bad_fit: 
            label_fit = f"SINGULAR COV MATRIX\nf = {f_fit:.2f} +/- {f_fit_err:.2f}\n"
        else:
            label_fit = f"f = {f_fit:.2f} +/- {f_fit_err:.2f}\n"
        
        label_fit += f"mu = {mu_fit:.2f} +/- {mu_fit_err:.2f}\n"
        label_fit += f"sigma = {sigma_fit:.2f} +/- {sigma_fit_err:.2f}\n"
        label_fit += f"nu = {nu_fit:.2f} +/- {nu_fit_err:.2f}\n"
        label_fit += f"chi2 / ndf = {chi2_obs:.2f} / {ndof} (p-value = {pvalue:.3f})"


        # Plot fit 
        fig, ax = plt.subplots(1, 1, figsize=(16, 9))

        ax.errorbar(w_data, counts_data, yerr=np.sqrt(counts_data),
                    fmt="o", label="data (signal + noise)")

        ax.plot(w_data, model, "-o", mfc="None",
                label=f"fit total\n" + label_fit)

        ax.plot(w_data, counts_model_noise, "--s", mfc="None",
                label="noise (fit)")

        ax.plot(w_data, counts_model_muons, ":^", mfc="None",
                label=f"muon (fit).")

        xticks = np.arange(0, 20, step=2)
        ax.set_xticks(xticks)

        ax.set_xlabel("Width / (3.125 ns)")
        ax.set_ylabel("Counts")
        ax.legend(fontsize="medium")

        title = f"fitting range = {w_min} <= width <= {w_max}"

        if not bad_fit:
            title+= (f"\n({ineff_muons:.1f} +/- {errors_ineff['ineff_muons_err']:.1f} ineff muons) /"
                    f" ({total_muons:.1f} +/- {errors_ineff['total_muons_err']:.1f} total muons) = "
                    f"{ratio:.3f} +/- {errors_ineff['ratio_err']:.3f} ")
         
        ax.set_title(title)

        fig.suptitle("Mixture fit: noise + muon (t-student)")

        figname = f"c{str(self.counter).zfill(4)}m{self.module}_mixture_fit_tStudent.png"
        fig.savefig(os.path.join(outdir, figname), bbox_inches="tight")
        plt.close(fig)

        print(f"{figname} saved")

        # Residuals of the fit 
        pull = (counts_data - model) / counts_data_err
        rel = (counts_data - model) / model

        fig, ax = plt.subplots(1, 2, figsize=(22, 12))

        ax[0].errorbar(w_data, pull, fmt="o")
        ax[0].set_xlabel("Width / (3.125 ns)")
        ax[0].set_ylabel("pull")
        ax[0].axvspan(xmin=w_min, xmax=w_max, color="g", alpha=0.1)

        ax[0].axhline(y=0, ls="--", color="grey")

        ax[1].errorbar(w_data, rel, fmt="o")
        ax[1].set_xlabel("Width / (3.125 ns)")
        ax[1].set_ylabel("relative res")

        ax[1].axhline(y=0, ls="--", color="grey")
        ax[1].axvspan(xmin=w_min, xmax=w_max, color="g", alpha=0.1)

        fig.tight_layout(pad=1.1)

        figname = f"c{str(self.counter).zfill(4)}m{self.module}_mixture_fit_tStudent_residuals.png"
        fig.savefig(os.path.join(outdir, figname), bbox_inches="tight")
        plt.close(fig)

        print(f"{figname} saved")


    def is_valid_covariance(self, cov, tol=1e-10):
            """
            Check whether a covariance matrix is numerically valid.

            A valid covariance matrix must satisfy:
            - Be finite (no NaN or inf entries)
            - Be well-conditioned (condition number below a threshold)
            - Be symmetric within numerical tolerance
            - Be positive semi-definite (all eigenvalues >= 0 within tolerance)

            Parameters
            ----------
            cov : array-like
                Candidate covariance matrix to validate.
            tol : float, optional
                Numerical tolerance used for symmetry and eigenvalue checks.
                Eigenvalues smaller than -tol are considered non-physical.

            Returns
            -------
            bool
                True if the matrix passes all validity checks, False otherwise.

            Notes
            -----
            This function is intended to guard downstream procedures (e.g. Monte Carlo
            sampling) against ill-defined covariance matrices resulting from unstable
            fits or numerical issues.

            """
            if cov is None or not np.all(np.isfinite(cov)):
                return False 
            
            if np.linalg.cond(cov) > 1e12:
                return False
            
            # 1. check simetría
            if not np.allclose(cov, cov.T, atol=tol):
                return False
            
            # 2. autovalores
            eigvals = np.linalg.eigvalsh(cov)  # más estable para matrices simétricas
            
            if np.min(eigvals) < -tol:
                return False

            return True


    def errors_monte_carlo_sampling(self, best_fit_param, cov_matrix, n_data_tot, 
            bad_fit, P_muon_student_t, n_toys=2000):
        """
        Estimate uncertainties on derived observables via Monte Carlo sampling.

        This function propagates the parameter uncertainties from a fit (encoded
        in the covariance matrix) to derived quantities related to the muon
        component of the model:
            - ineff_muons: number of muons with width <= 3
            - total_muons: total number of muons
            - ratio: ineff_muons / total_muons

        The propagation is performed by sampling parameter sets from a multivariate
        normal distribution defined by the best-fit parameters and their covariance
        matrix, and recomputing the observables for each sample.

        Parameters
        ----------
        best_fit_param : array-like
            Best-fit parameter values (f, mu, sigma, nu) from least_squares.
        cov_matrix : array-like
            Covariance matrix of the fitted parameters.
        n_data_tot : int
            Total number of entries in the observed histogram.
        bad_fit : bool
            Flag indicating whether the fit or covariance matrix is unreliable.
            If True, no sampling is performed and None errors are returned.
        P_muon_student_t : callable
            Function that evaluates the discretized Student-t PDF for the muon
            component.
        n_toys : int, optional
            Number of Monte Carlo samples used to estimate uncertainties.

        Returns
        -------
        dict
            Dictionary containing standard deviations of the sampled distributions:
            {
                "ineff_muons_err": float or None,
                "total_muons_err": float or None,
                "ratio_err": float or None
            }

        Notes
        -----
        - Unphysical parameter samples (e.g. sigma <= 0, nu <= 1, or f outside [0,1])
        are discarded.
        - A consistency check ensures proper normalization of the muon component.
        - If no valid samples are obtained, or if the fit is flagged as bad,
        the function returns None for all uncertainties.
        - The resulting uncertainties naturally include parameter correlations.
        """
       

        if bad_fit:
            return {"ineff_muons_err": None, 
                "total_muons_err": None,
                "ratio_err": None}
            

        w_data = np.asarray(self.prob_width_noise["width"])

        #f_fit, mu_fit, sigma_fit, nu_fit = best_fit_param

        samples = np.random.multivariate_normal(best_fit_param, cov_matrix, size=n_toys)

        ineff_samples = []
        total_samples = []
        ratio_samples = []

        for f_s, mu_s, sigma_s, nu_s in samples:
        
            # avoid unphysical values
            if sigma_s <= 0 or nu_s <= 1 or not (0 <= f_s <= 1):
                continue
            
            p_muon_s = P_muon_student_t(w_data, mu_s, sigma_s, nu_s)
            
            counts_mu_s = n_data_tot * (1 - f_s) * p_muon_s

                       
            ineff_s = np.sum(counts_mu_s[w_data <= 3])
            total_s = np.sum(counts_mu_s)

            # Sanity cross check
            if not np.isclose(total_s, n_data_tot * (1 - f_s), rtol=1e-6):
                print("[WARNING]: normalization issue in toy")
                continue

            if total_s > 0:
                ineff_samples.append(ineff_s)
                total_samples.append(total_s)
                ratio_samples.append(ineff_s / total_s)
        

        n_valid = len(ratio_samples)

        if n_valid == 0:
            print("[WARNING]: no valid toys -> cannot estimate errors")
            return {"ineff_muons_err": None, 
                    "total_muons_err": None,
                    "ratio_err": None}

        if n_valid < 0.5 * n_toys:
            print("[WARNING]: many toys rejected -> unreliable errors")
        
        return {"ineff_muons_err": np.std(ineff_samples, ddof=1), 
                "total_muons_err": np.std(total_samples, ddof=1),
                "ratio_err": np.std(ratio_samples, ddof=1)}
        





    def analyze_single_bar_events_DEPRECATED(self, lge_min, outdir):
        """
        In this function, I was trying to estimate the number of inefficient muons (ie 
        those that fail to produce a muon pattern) by using a noise-rate argument: I tried
        to estimate how many noise event I was expecting in the trigger scope, and then used that
        to get counts_noise = N_noise_estimated * P_noise(w). Therefore, the number of muons for each
        w was counts_muons(w) = n_data(w) - counts_noise(w).

        This turns out to be complicated: you need to rely too much on the noise rate to 
        be very well estimated. This also lead to some  counts_muons(w) = n_data(w) - counts_noise(w) 
        to be negative.

        Now, I fit a model to the width distribution accounting for noise and muons
        to estimate the inefficient muons
        
             
        """

        if self.prob_width_noise is None:
            raise ValueError("You must first fill self.prob_width_noise")

        tmin, tmax = self.trigger_scope
        dt = tmax - tmin

        # ------------------------------------------------------------
        # 1. Select events with 1 or 2 active bars in trigger window
        # ------------------------------------------------------------
        valid_events, event_type = self.get_selected_events(apply_test=False)

        valid_event_df = pd.DataFrame({
            "eventId": valid_events,
            "event_type": event_type
        })

        df_valid = self.df[self.df["eventId"].isin(valid_events)].copy()

        assert np.array_equal(df_valid["eventId"].unique(), valid_event_df["eventId"].unique()), "mismatch!"

        # Attach event classification ("one_seq" or "two_seq")
        df_valid = df_valid.merge(valid_event_df, on="eventId", how="left")

        # Apply physics selection cuts (independent of noise model)
        df_valid = df_valid[
            (df_valid["isT4"] == 1) &
            (df_valid["lgEnergy"] > lge_min)
        ]

        # ------------------------------------------------------------
        # 2. Keep only events with exactly one sequence in trigger window
        # ------------------------------------------------------------
        single_bar_all = df_valid[df_valid["event_type"] == "one_seq"]

        # before cut in start time in trigger window - look at the
        # distribution of start times
        fig, ax = plt.subplots(1, 1, figsize=(16, 9))

        tbinedges = np.arange(0, self.trace_length + 2, step=1) - 0.5
        ax.hist(single_bar_all["startTime"], histtype="step", lw=2, bins=tbinedges)
        ax.set_xlabel("Start time / (3.125 ns)")
        ax.set_ylabel("Counts")
        ax.set_yscale("log")   

        fig.suptitle((self.label_cuts + f" single bar/sequence events BEFORE restricting to trigger scope only. "
                        f"isT4 = 1, lge_min={lge_min}")) 

        figname = f"c{str(self.counter).zfill(4)}m{self.module}_start_time_single_bar.png"

        fig.savefig(os.path.join(outdir, figname), bbox_inches="tight")
        plt.close(fig)
        print(f"{figname} saved")

        # Obtain rate of noise sequences in the sample of single_bar to avoid biases
        
        # rate = # seq / (#unique events * dt_noise)
        dt_early = self.noise_early[1] - self.noise_early[0]
        dt_late = self.noise_late[1] - self.noise_late[0]
        dt_noise = dt_early + dt_late
        
        mask_early = (
            (self.noise_early[0] < single_bar_all["startTime"]) &
            (single_bar_all["startTime"] < self.noise_early[1])
        )

        mask_late = (
            (self.noise_late[0] < single_bar_all["startTime"]) &
            (single_bar_all["startTime"] < self.noise_late[1])
        )

        single_bar_noise = single_bar_all[mask_early | mask_late]

        assert len(single_bar_noise) == np.sum(mask_early | mask_late)

        n_seq_noise = len(single_bar_noise)
        n_unique_events = single_bar_all["eventId"].nunique() 
        
        rate = n_seq_noise / (n_unique_events*dt_noise)

        # Restrict to sequences inside trigger window
        single_bar_trigger = single_bar_all[
            (tmin < single_bar_all["startTime"]) &
            (single_bar_all["startTime"] < tmax)
        ]

        assert np.isfinite(self.df["lgEnergy"]).all(), "strange"
        assert np.isfinite(single_bar_all["lgEnergy"]).all(), "strange"

        # ------------------------------------------------------------
        # 3. Compute expected number of noise sequences in trigger scope
        # (based on the noise rate previously computed in the noise windows)
        # ------------------------------------------------------------
        # mean expected number of noise sequences per event in trigger window
        mu = rate * dt

        print("mu = ", mu)

        # ------------------------------------------------------------
        # 5. Convert sequence-level expectation → event-level probability
        # ------------------------------------------------------------

        # Total number of events with one sequence in one bar
        n_unique_events_in_trigger_scope = single_bar_trigger["eventId"].nunique()

        # based on rate
        expected_noise_events_1seq = n_unique_events_in_trigger_scope * mu

        # based on linear extrapolation
        expected_noise_events_1seq_linear = n_seq_noise * dt / dt_noise

        # bc of my event selection, these are the same 
        assert np.isclose(expected_noise_events_1seq, expected_noise_events_1seq_linear)

        n_pred_noise_side = rate * n_unique_events * dt_noise
        print("Closure check:", n_pred_noise_side, n_seq_noise)
        print(f"expected_noise_rate = {expected_noise_events_1seq:.1f}, expected_linear = {expected_noise_events_1seq_linear:.1f} ")

        # ------------------------------------------------------------
        # 6. Histograms: for each width, get how many events are noise and how many are muons
        # ------------------------------------------------------------

        # expected noise counts for each width
        counts_noise = expected_noise_events_1seq * self.prob_width_noise["prob"]
        w_noise = np.asarray(self.prob_width_noise["width"])
        
        # actual counts for each width (muons + noise)
        wbinedges = np.arange(0, 20, step=1) - 0.5 
        wbincenters = wbinedges[:-1] + np.diff(wbinedges)/2

        counts_signal_plus_noise, _ = np.histogram(single_bar_trigger["width"], bins=wbinedges)

        inefficient_muons = counts_signal_plus_noise[wbincenters <= 3] - counts_noise[w_noise <= 3]

        print(f"np.sum(counts_noise) = {np.sum(counts_noise)}, expected_noise_events_1seq = {expected_noise_events_1seq:.1f}")
        mask = wbincenters <= 3
        mask2 = w_noise <= 3

        tot_ineff_muons = np.sum(inefficient_muons) 

        fig, ax = plt.subplots(1, 1, figsize=(16, 9))
        ms = 10

        ax.errorbar(w_noise, counts_noise, yerr=np.sqrt(counts_noise), fmt="-s",
                    ms=ms, label=f"expected noise ({np.sum(counts_noise):.1f})")
        
        ax.errorbar(wbincenters, counts_signal_plus_noise, yerr=np.sqrt(counts_signal_plus_noise), fmt="o",
                    ms=ms, label=f"signal + noise ({np.sum(counts_signal_plus_noise)})")
        
        ax.errorbar(wbincenters[wbincenters <= 3], inefficient_muons, yerr=np.sqrt(inefficient_muons), fmt="--^",
                    ms=ms, label=f"inefficient muons ({tot_ineff_muons:.1f})")

        ax.legend()
        ax.set_xlabel("Width / (3.125 ns)")
        ax.set_ylabel("Counts")
        xticks = np.arange(0, 20, step=2)
        ax.set_xticks(xticks)
        #ax.set_yscale("log")

        fig.suptitle((self.label_cuts + f"single bar events. isT4 = 1, lge_min={lge_min}")) 

        figname = f"c{str(self.counter).zfill(4)}m{self.module}_ineff_single_bar.png"

        fig.tight_layout(pad=1.1)

        fig.savefig(os.path.join(outdir, figname), bbox_inches="tight")
        plt.close(fig)
        print(f"{figname} saved")

        ####### ---------------
        #### PRUEBA FIT: noise template + gaussiana
        counts_data, _ = np.histogram(single_bar_trigger["width"], bins=self.prob_width_noise["width_bin_edges"])
        p_data = counts_data / np.sum(counts_data)

        def discrete_gaussian(w, mu, sigma):
            g = np.exp(-0.5 * ((w - mu) / sigma)**2)
            return g / np.sum(g)
        
        w_vals = np.asarray(self.prob_width_noise["width"])
        p_noise = np.asarray(self.prob_width_noise["prob"])

        def mixture_model(params):
            f, mu, sigma = params

            # restricciones físicas simples
            if not (0 <= f <= 1 and sigma > 0):
                return np.inf * np.ones_like(p_data)

            p_muon = discrete_gaussian(w_vals, mu, sigma)

            model = f * p_noise + (1 - f) * p_muon

            return model
        
        def residuals(params):
            model = mixture_model(params)
            return (p_data - model)
        
        

        # valores iniciales razonables
        init = [0.3, 7.0, 2.0]  # f, mu, sigma

        result = least_squares(residuals, init)

        f_fit, mu_fit, sigma_fit = result.x

        print(f"Fit results: f_noise = {f_fit:.3f}, mu_muon = {mu_fit:.2f}, sigma = {sigma_fit:.2f}")

        p_muon_fit = discrete_gaussian(w_vals, mu_fit, sigma_fit)
        p_model = f_fit * p_noise + (1 - f_fit) * p_muon_fit

        # convertir a counts
        counts_model = p_model * np.sum(counts_data)
        
        counts_noise_fit = f_fit * p_noise * np.sum(counts_data)
        
        counts_muons_fit = (1 - f_fit) * p_muon_fit * np.sum(counts_data)

        ineff_muons_gaussian = np.sum(counts_muons_fit[w_vals <= 3])

        fig, ax = plt.subplots(1, 1, figsize=(16, 9))

        ax.errorbar(w_vals, counts_data, yerr=np.sqrt(counts_data),
                    fmt="o", label="data (signal + noise)")

        ax.plot(w_vals, counts_model, "-o", mfc="None",
                label=f"fit total (f={f_fit:.2f})")

        ax.plot(w_vals, counts_noise_fit, "--",
                label="noise (fit)")

        ax.plot(w_vals, counts_muons_fit, ":^", mfc="None",
                label=f"muon (fit). {ineff_muons_gaussian:.2f} ineff muons")

        xticks = np.arange(0, 20, step=2)
        ax.set_xticks(xticks)

        ax.set_xlabel("Width / (3.125 ns)")
        ax.set_ylabel("Counts")
        ax.legend()

        fig.suptitle("Mixture fit: noise + muon")

        figname = f"c{str(self.counter).zfill(4)}m{self.module}_mixture_fit.png"
        fig.savefig(os.path.join(outdir, figname), bbox_inches="tight")
        plt.close(fig)

        print(f"{figname} saved")

        ####### ---------------
        #### PRUEBA FIT: noise template + t student

        def P_muon_student_t(w, mu, sigma, nu):
            """
            Student-t PDF evaluada en bins discretos.
            """
            return student_t.pdf(w, df=nu, loc=mu, scale=sigma)
        
        P_noise_template = self.prob_width_noise["prob"]
        P_noise_template = P_noise_template / np.sum(P_noise_template)

        y = counts_data.astype(float)
        x = w_vals

        # errores Poisson
        sigma_y = np.sqrt(y)
        sigma_y[sigma_y == 0] = 1.0

        def model(params):
            f, mu, sigma, nu = params

            P_mu = P_muon_student_t(x, mu, sigma, nu)
            P_mu = P_mu / np.sum(P_mu)  # normalización discreta

            return f * P_noise_template + (1 - f) * P_mu
        
        def residuals(params):
            m = model(params)
            return (y - m * np.sum(y)) / sigma_y
        

        p0 = [
            0.2,   # f noise
            7.0,   # mu muon
            2.0,   # sigma
            5.0    # nu (cola; típico 3–10)
        ]

        bounds = (
            [0.0, 0.0, 0.1, 1.0],   # lower
            [1.0, 20.0, 10.0, 50.0] # upper
        )

        result = least_squares(residuals, p0, bounds=bounds)

        f_fit, mu_fit, sigma_fit, nu_fit = result.x

        fit_label = f"f = {f_fit:.2f}, mu = {mu_fit:.2f}, sigma = {sigma_fit:.2f}, nu = {nu_fit:.2f}"

        fit_curve = model(result.x) * np.sum(y)
        fit_noise = f_fit*P_noise_template*np.sum(y)
        fit_muons = (1-f_fit)*P_muon_student_t(x, mu_fit, sigma_fit, nu_fit)*np.sum(y)

        ineff_muons = np.sum(fit_muons[x <= 3])

        fig, ax = plt.subplots(1, 1, figsize=(16, 9))

        ax.errorbar(w_vals, counts_data, yerr=np.sqrt(counts_data),
                    fmt="o", label="data (signal + noise)")

        ax.plot(x, fit_curve, "-o", mfc="None", label="fit (noise + Student-t muon).\n" + fit_label)
        ax.plot(x, fit_noise, color="g", ls="--", marker="s", label="noise")
        ax.plot(x, fit_muons, color="r", ls="-.", marker="^", label=f"muons. Ineff muons = {ineff_muons:.3f}")

        ax.legend()

        xticks = np.arange(0, 20, step=2)
        ax.set_xticks(xticks)

        ax.set_xlabel("Width / (3.125 ns)")
        ax.set_ylabel("Counts")
        ax.legend()

        fig.suptitle("Mixture fit: noise + muon")

        figname = f"c{str(self.counter).zfill(4)}m{self.module}_mixture_fit_student.png"
        fig.savefig(os.path.join(outdir, figname), bbox_inches="tight")
        plt.close(fig)

        print(f"{figname} saved")

        
                
        




        
       
