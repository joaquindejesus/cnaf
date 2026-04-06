# ======== Standard library imports ========
import os
import pickle
import sys
from pathlib import Path

# ======== Third-party imports ========
import numpy as np
from scipy.stats import chi2
import pandas as pd 
import matplotlib.pyplot as plt 

# =========== My imports ================
sys.path.append(os.path.expanduser("/storage/gpfs_data/pauger/users/jdejesus/my_python_modules"))
from myClasses import Profile
from myMplStyles import setStyle
from myUsefulFunctions import generate_colormap

setStyle(False)

class Sequences:
    """
    """
    def __init__(self, counter, module, noise_early=None, noise_late=None, 
                trigger_scope=None, trace_length=2048, df=None,):

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
    
        self.noise_early = noise_early
        self.noise_late = noise_late
        self.trigger_scope = trigger_scope
        self.df = df.copy() if df is not None else None

        self.bad_periods_found_and_removed = False

        self.trace_length = trace_length

        self.label_cuts = f"Noise early: {noise_early[0]} < t < {noise_early[1]}; Noise late: {noise_late[0]} < t < {noise_late[1]}\n"
        self.label_cuts += f"Trigger scope: {trigger_scope[0]} < t < {trigger_scope[1]}"

    def sanity_checks(self):
        """
        Basic consistency checks on the dataframe.
        """
        if self.df is None:
            raise ValueError("df is None")

        df_module_empty = self.df[self.df["entryType"] == "module_empty"].copy()

        assert not df_module_empty["eventId"].duplicated().any(), \
            "There are duplicates for module_empty entries!"

    def exclude_bad_periods(self, outdir):
        """
        """
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

        # figures 
        ms = 10

        # -- Width vs time
        fig, ax = plt.subplots(1, 1, figsize=(14, 7))
        
        ax.errorbar(pwidth.x_mean, pwidth.y_mean, yerr=pwidth.y_std_of_mean, fmt="o", ms=ms)
        ax.axhline(y=median_width, ls="--", color="k", alpha=0.8, lw=2, label=f"median = {median_width:.1f}, std = {std_width:.1f}")
        ax.axhspan(ymin=median_width - 3*std_width, ymax=median_width + 3*std_width, color="grey", alpha=0.5)
        ax.legend(fontsize="medium")

        # detecting bad periods as periods with outliers in the mean width
        outlier_w_mask_1 = np.abs((np.asarray(pwidth.y_mean) - median_width)) / (3*std_width) > 1
        outlier_w_mask_2 = (np.asarray(pwidth.y_mean)) > 80

        outlier_w_mask = outlier_w_mask_1 | outlier_w_mask_2


        if outlier_w_mask.any():
            ax.errorbar(np.asarray(pwidth.x_mean)[outlier_w_mask], np.asarray(pwidth.y_mean)[outlier_w_mask], fmt="x", color="r", ms=13)

        ax.set_xlabel("GPS second")
        ax.set_ylabel(r"$\langle \text{width} \, / \, (3.125 \text{ ns})\rangle $")

        figname = f"c{str(self.counter).zfill(4)}m{self.module}_width_vs_gps_raw.png"

        fig.savefig(os.path.join(outdir, figname), bbox_inches="tight")
        plt.close(fig)
        print(f"{figname} saved")

         # -- Start time vs time
        fig, ax = plt.subplots(1, 1, figsize=(14, 7))
        
        ax.errorbar(pt.x_mean, pt.y_mean, yerr=pt.y_std_of_mean, fmt="o", ms=ms)
        ax.axhline(y=median_t, ls="--", color="k", alpha=0.8, lw=2, label=f"median = {median_t:.1f}, std = {std_t:.1f}")
        ax.axhspan(ymin=median_t - 3*std_t, ymax=median_t + 3*std_t, color="grey", alpha=0.5)
        ax.legend(fontsize="medium")

        # detecting bad periods as periods with outliers in the mean t
        outlier_t_mask = np.abs((np.asarray(pt.y_mean) - median_t)) / (3*std_t) > 1
        if outlier_t_mask.any():
            ax.errorbar(np.asarray(pt.x_mean)[outlier_t_mask], np.asarray(pt.y_mean)[outlier_t_mask], fmt="x", color="r", ms=13)

        ax.set_xlabel("GPS second")
        ax.set_ylabel(r"$\langle \text{t} \, / \, (3.125 \text{ ns})\rangle $")

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
                mask &= ~((self.df["gpsSecond"] > t0) & (self.df["gpsSecond"] < t1))

            self.df = self.df[mask]

            self.bad_periods_found_and_removed = True

        
        # Repeat the plots for the cleaned distributions:
        df_seq = self.df[self.df["entryType"] == "sequence"].copy()

        pwidth = Profile(df_seq["gpsSecond"], df_seq["width"]).doProfile(gps_bin_edges)
        pt = Profile(df_seq["gpsSecond"], df_seq["startTime"]).doProfile(gps_bin_edges)

        # -- Width vs time
        fig, ax = plt.subplots(1, 1, figsize=(14, 7))
        
        ax.errorbar(pwidth.x_mean, pwidth.y_mean, yerr=pwidth.y_std_of_mean, fmt="o", ms=ms)
        ax.axhline(y=median_width, ls="--", color="k", alpha=0.8, lw=2, label=f"median = {median_width:.1f}, std = {std_width:.1f}")
        ax.axhspan(ymin=median_width - 3*std_width, ymax=median_width + 3*std_width, color="grey", alpha=0.5)
        ax.legend(fontsize="medium")

        ax.set_xlabel("GPS second")
        ax.set_ylabel(r"$\langle \text{width} \, / \, (3.125 \text{ ns})\rangle $")

        figname = f"c{str(self.counter).zfill(4)}m{self.module}_width_vs_gps_clean.png"

        fig.savefig(os.path.join(outdir, figname), bbox_inches="tight")
        plt.close(fig)
        print(f"{figname} saved")

         # -- Start time vs time
        fig, ax = plt.subplots(1, 1, figsize=(14, 7))
        
        ax.errorbar(pt.x_mean, pt.y_mean, yerr=pt.y_std_of_mean, fmt="o", ms=ms)
        ax.axhline(y=median_t, ls="--", color="k", alpha=0.8, lw=2, label=f"median = {median_t:.1f}, std = {std_t:.1f}")
        ax.axhspan(ymin=median_t - 3*std_t, ymax=median_t + 3*std_t, color="grey", alpha=0.5)
        ax.legend(fontsize="medium")

        ax.set_xlabel("GPS second")
        ax.set_ylabel(r"$\langle \text{t} \, / \, (3.125 \text{ ns})\rangle $")

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

        fig.savefig(os.path.join(outdir, figname), bbox_inches="tight")
        plt.close(fig)
        print(f"{figname} saved")

       
