import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.time import Time


# Add custom module paths
sys.path.append('/storage/gpfs_data/pauger/users/jdejesus/my_python_modules/')
from myClasses import Profile
from myUsefulFunctions import generate_colormap
from myMplStyles import setStyle

fname = "module_level_info_01-2025_09-2026_{1816, 1771, 643, 1815}.csv"

df = pd.read_csv(fname)

setStyle(use_tex=False)

def mask_outliers(p, unbinned_gps, tol=5):
    """
    Arguments:
        p: Profile object of k vs GPS time.
        unbinned_gps: GPS times of the unbinned data.
        tol: tolerance in units of sigma to classify a bad period.

    Returns:
        Boolean array with the same shape as unbinned_gps.
        True indicates that the event belongs to a bad period.

    Example:
        >> bad_periods = mask_outliers(p, unbinned_gps)
        >> bad_gps = unbinned_gps[bad_periods]
        >> good_gps = unbinned_gps[~bad_periods]
    """

    unbinned_gps = np.asarray(unbinned_gps)


    median = np.nanmedian(p.y_mean)

    # robust estimation of std using the 16th and 84th percentiles
    q16 = np.nanquantile(p.y_mean, q=0.16)
    q84 = np.nanquantile(p.y_mean, q=0.84)
    std_iql = (q84 - q16) / 2

    outliers_bool = np.abs(p.y_mean - median) / std_iql > tol

    # One boolean per unbinned event
    bad_periods = np.zeros(unbinned_gps.shape, dtype=bool)

    for i in range(len(p.x_center)):

        if outliers_bool[i]:
            gps1 = p.x_bin_edges[i]
            gps2 = p.x_bin_edges[i + 1]

            bad_periods |= (gps1 <= unbinned_gps) & (unbinned_gps < gps2)

    return {"bad_periods_bool": bad_periods, "outliers_bool": outliers_bool}





def plot_counter(counter_id, rmin=300, theta_max=40, fig_dir=".",
                 date_1="2025-01-01", date_2="2026-09-20"):
    """
    """
    df_c = df[(df["counter_id"] == counter_id) & (df["r"] > rmin)
              & (df["theta"] < theta_max)]

    nmu_c = {101: [], 102: [], 103: []}
    k_c = {101: [], 102: [], 103: []}
    gps_c = {101: [], 102: [], 103: []}

    for event_id in df_c["event_id"].unique():
        df_event = df_c[df_c["event_id"] == event_id]

        for module_id in nmu_c:
            module_rows = df_event.loc[
                df_event["module_id"] == module_id,
                ["nmu", "k", "gps"],
            ]

            if len(module_rows) == 0:
                nmu = np.nan
                k = np.nan
                gps = np.nan
                # the module is not in the event
                #print(f"{counter_id}-{module_id} is not in the event {event_id}. Filling nans")

            elif len(module_rows) == 1:
                nmu = module_rows["nmu"].iloc[0]
                k = module_rows["k"].iloc[0]
                gps = module_rows["gps"].iloc[0]

                if k < 0:
                    print(f"k = {k} < 0 ! for {event_id} and module {counter_id}-{module_id}!")

            else:
                raise ValueError(
                    f"Event {event_id}, module {module_id}: "
                    f"found {len(module_rows)} entries instead of one"
                )

            #print(
            #    f"event={event_id}, module={module_id}, "
            #    f"nmu={nmu}, k={k}"
            #)



            nmu_c[module_id].append(nmu)
            k_c[module_id].append(k)
            gps_c[module_id].append(gps)

    # Plotting all combinations of modules
    possible_pairs =  list(combinations(nmu_c.keys(), 2))

    nmu_bin_edges = np.arange(0, 16, step=1) - 0.5

    # common title for plots
    title = (f" Counter {counter_id}. lg(E/eV) > 17.3, r > {rmin:.0f} m, "
             r"$\theta \, / \, ^\circ $ < " + f"{theta_max:.0f}.\n"
             f"{date_1} - {date_2}")

    # traling string for fig names
    trail = f"{date_1}_{date_2}"
    for pair in possible_pairs:

        module_1, module_2 = pair


        # -------------------------------------------------
        ## Fig 0 - time series of k: looking for unstable periods
        k_all_1 = np.asarray(k_c[module_1], dtype=float)
        k_all_2 = np.asarray(k_c[module_2], dtype=float)

        gps_all_1 = np.asarray(gps_c[module_1], dtype=float)
        gps_all_2 = np.asarray(gps_c[module_2], dtype=float)

        # Note: at this stage k_all has np.nans, so asking assert (k_all_1 <= 0).all() fails

        # taking only events in which the two modules were present and are in the
        # period between date_1 and date_2
        t1 = Time(f"{date_1}T00:00:00", scale="utc")
        t2 = Time(f"{date_2}T00:00:00", scale="utc")

        time_mask = ((t1.gps < gps_all_1) & (gps_all_1 < t2.gps) &
                     (t1.gps < gps_all_2) & (gps_all_2 < t2.gps) )

        common_mask_k = np.isfinite(k_all_1) & np.isfinite(k_all_2) & time_mask

        gps = gps_all_1[common_mask_k]
        k_1 = k_all_1[common_mask_k]
        k_2 = k_all_2[common_mask_k]


        assert (k_1 >= 0).all()
        assert (k_2 >= 0).all()


        fig0, ax0 = plt.subplots(1, 1, figsize=(19, 9))


        assert np.array_equal(gps, gps_all_2[common_mask_k]), "both gps times should be the same!"

        seconds_in_day = 3600 * 24
        gps_bin_width = seconds_in_day * 15
        gps_bin_edges = np.arange(np.min(gps), np.max(gps) + gps_bin_width,
                                  step=gps_bin_width)

        pgps_1 = Profile(x=gps, y=k_1).doProfile(gps_bin_edges)
        median_1 = np.median(pgps_1.y_mean)

        bp_1 = mask_outliers(p=pgps_1, unbinned_gps=gps_all_1)
        outliers_bool_1 = bp_1["outliers_bool"]
        bad_periods_bool_1 = bp_1["bad_periods_bool"]
        good_bool_1 = ~bad_periods_bool_1


        pgps_2 = Profile(x=gps, y=k_2).doProfile(gps_bin_edges)
        median_2 = np.median(pgps_2.y_mean)

        bp_2 = mask_outliers(p=pgps_2, unbinned_gps=gps_all_2)
        outliers_bool_2 = bp_2["outliers_bool"]
        bad_periods_bool_2 = bp_2["bad_periods_bool"]
        good_bool_2 = ~bad_periods_bool_2

        ax0.errorbar(pgps_1.x_mean, pgps_1.y_mean, yerr=pgps_1.y_std_of_mean,
                    color="r", fmt="o", ms=9, alpha=0.7, label=f"Module {module_1} (median = {median_1:.2f})")
        ax0.errorbar(np.array(pgps_1.x_mean)[outliers_bool_1], np.array(pgps_1.y_mean)[outliers_bool_1],
                    color="m", fmt="x", ms=10)
        ax0.axhline(y=median_1, ls="--", color="r")

        ax0.errorbar(pgps_2.x_mean, pgps_2.y_mean, yerr=pgps_2.y_std_of_mean,
                    color="g", fmt="s", mfc="None", ms=9, alpha=0.9, label=f"Module {module_2} (median = {median_2:.2f})")
        ax0.errorbar(np.array(pgps_2.x_mean)[outliers_bool_2], np.array(pgps_2.y_mean)[outliers_bool_2],
                    color="m", fmt="x", ms=10)
        ax0.axhline(y=median_2, ls="-.", color="g")

        ax0.set_xlabel(r"gps second")
        ax0.set_ylabel(r"$\langle k \rangle$")
        ax0.legend()
        ax0.set_title(title)

        # Vertical line in important dates for selected modules
        # dates given by B. Andrada
        if counter_id == 643 and 102 in pair:
            t = Time("2026-08-19T00:00:00", scale="utc")

            ax0.axvline(x=t.gps, ls="--", color="k")

        if counter_id == 1816 and 103 in pair:
            t = Time("2025-09-17T00:00:00", scale="utc")

            ax0.axvline(x=t.gps, ls="--", color="k")


        figname0 = os.path.join(fig_dir, f"c{counter_id}_{module_1}_{module_2}_k_time_series_{trail}.png")
        fig0.savefig(figname0, bbox_inches="tight")

        plt.close(fig0)
        print(f"{figname0} generated")

        # removing outliers for the next stages of the analysis
        # as they are likely bad periods
        common_mask = common_mask_k & (good_bool_1) & (good_bool_2)


        # ---------------------------------
        # Fig. 1 - nmu module_1 vs nmu module_2

        nmu_all_1 = np.asarray(nmu_c[module_1], dtype=float)
        nmu_all_2 = np.asarray(nmu_c[module_2], dtype=float)

        nmu_1 = nmu_all_1[common_mask]
        nmu_2 = nmu_all_2[common_mask]

        print(
            f"Modules {module_1}-{module_2}: "
            f"{len(nmu_1)} common events"
        )


        fig, ax = plt.subplots(1, 1, figsize=(16, 9))

        # scatter plot (too bussy)
        #ax.errorbar(nmu_1, nmu_2, fmt="o",
                    #color="grey", alpha=0.7)

        # hist2d
        h = ax.hist2d(nmu_1, nmu_2, bins=nmu_bin_edges, norm=LogNorm())

        p = Profile(x=nmu_1,
                    y=nmu_2).doProfile(nmu_bin_edges)

        ax.errorbar(p.x_center, p.y_mean, yerr=p.y_std_of_mean,
                    color="r", fmt="o", ms=9, alpha=0.9)

        ax.set_xlabel(rf"$\hat{{N}}_{{\mu}}^{{{module_1}}}$")
        ax.set_ylabel(rf"$\hat{{N}}_{{\mu}}^{{{module_2}}}$ ")

        # identity
        ax.plot([min(nmu_1), max(nmu_1)],
                [min(nmu_1), max(nmu_1)], ls="--",
                color="k")

        ax.set_xlim([-0.5, 15])
        ax.set_ylim([-0.5, 15])

        ax.set_title(title)

        fig.colorbar(h[3], ax=ax, label="Counts")

        figname = os.path.join(fig_dir, f"c{counter_id}_{module_1}_{module_2}_nmu_{trail}.png")
        fig.savefig(figname, bbox_inches="tight")

        plt.close(fig)
        print(f"{figname} generated")

        # --------------------------
        # delta Nmu

        fig2, ax2 = plt.subplots(1, 1, figsize=(16, 9))
        delta = np.array(nmu_2) - np.array(nmu_1)

        pdelta = Profile(x=nmu_1,
                    y=delta).doProfile(nmu_bin_edges)

        ax2.errorbar(pdelta.x_center, pdelta.y_mean, yerr=pdelta.y_std_of_mean,
                    fmt="o", color="m", ms=9)
        ax2.axhline(y=0, ls="--", color="k")

        ax2.set_xlabel(rf"$\hat{{N}}_{{\mu}}^{{{module_1}}}$")
        ax2.set_ylabel(rf"$\hat{{N}}_{{\mu}}^{{{module_2}}}$ - $\hat{{N}}_{{\mu}}^{{{module_1}}}$ ")
        ax2.set_title(title)

        figname2 = os.path.join(fig_dir, f"c{counter_id}_{module_1}_{module_2}_delta_nmu_{trail}.png")
        fig2.savefig(figname2, bbox_inches="tight")

        plt.close(fig2)
        print(f"{figname2} generated")


        # ---------------------------------
        # k module_1 vs k_module 2
        k_1_good = k_all_1[common_mask]
        k_2_good = k_all_2[common_mask]

        p = Profile(x=k_1_good,
                    y=k_2_good).doProfile(nmu_bin_edges)

        fig3, ax3 = plt.subplots(1, 1, figsize=(16, 9))

        ax3.errorbar(p.x_center, p.y_mean, yerr=p.y_std_of_mean,
                    color="r", fmt="o", ms=9, alpha=0.9)

        ax3.set_xlabel(rf"$k_{{{module_1}}}$")
        ax3.set_ylabel(rf"$k_{{{module_2}}}$")

        # identity
        ax3.plot([min(k_1_good), max(k_1_good)],
                [min(k_1_good), max(k_1_good)], ls="--",
                color="k")


        ax3.set_xlim([-0.5, 15])
        ax3.set_ylim([-0.5, 15])

        ax3.set_title(title)

        figname3 = os.path.join(fig_dir, f"c{counter_id}_{module_1}_{module_2}_k_{trail}.png")
        fig3.savefig(figname3, bbox_inches="tight")

        plt.close(fig3)
        print(f"{figname3} generated")

        # -----------------------------------------------------
        # Time series of nmu (already cleaned from bad periods)







rmin = 300

plots_dir = "./plots"

# analysis for counter 1815
plot_counter(1815, fig_dir=plots_dir)

# analysis for counter 1771
plot_counter(1771, fig_dir=plots_dir)

# analysis for counter 643 - full period
plot_counter(643, fig_dir=plots_dir)

# analysis for counter 643 - before and after intervention
plot_counter(643, fig_dir=plots_dir, date_2="2026-08-17")
plot_counter(643, fig_dir=plots_dir, date_1="2026-08-20")


# analysis for counter 1816 - full period
plot_counter(1816, fig_dir=plots_dir)

# analysis for counter 1816 - before and after intervention
plot_counter(1816, fig_dir=plots_dir, date_2="2025-09-16")
plot_counter(1816, fig_dir=plots_dir, date_1="2025-09-19")
