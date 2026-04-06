"""
Functions to handle reading of ADST for the PMT paper reproduction.
Note that the offline environment needs to be set
"""

import math
import os
import sys

import numpy as np
from scipy.stats import chi2, norm, poisson
import ROOT
import pandas as pd

sys.path.append("/storage/gpfs_data/pauger/users/jdejesus/my_python_modules/auger/")
from augerAlgorithms import extract_one_sequences

# ============================================================
# Schema for the output dictionary
# ============================================================

OUTDICT_KEYS = [
    "event_id", "yymmdd", "hhmmss", "gps_sec",
    "theta", "theta_err", "phi", "cos_theta", "cos_theta_err",
    "lgE", "shower_size", "shower_size_err",
    "x_core_site_CS", "y_core_site_CS", "z_core_site_CS",
    "x_core_UTM_CS", "y_core_UTM_CS", "z_core_UTM_CS",
    "x_axis_site_CS", "y_axis_site_CS", "z_axis_site_CS",
    "x_axis_core_CS", "y_axis_core_CS", "z_axis_core_CS",
    "x_SD_err", "y_SD_err", "corr_xy_SD",
    "T4_trigg", "T4_trigg_name", "T5_trigg", "hottest_sd_station", "is6T5",
    "beta", "beta_err", "nmu450", "nmu450_err",
    "rho450", "rho450_err", "rho450_cic", "rho450_cic_err",
    "x_core_site_CS_md", "y_core_site_CS_md", "z_core_site_CS_md",
    "x_core_UTM_CS_md", "y_core_UTM_CS_md", "z_core_UTM_CS_md",
    "rec_level", "chi2_ldf", "ldf_like", "ndof", "deviance",
    "is_beta_fixed", "is_core_fixed", "is_md_event_saturated", "isMD6T5",
    "r_hottest", "hottest_counter",
    "sd_rec_level", "sd_ldf_status"
]


def init_outdict():
    """Initialize output dictionary with empty lists for all expected keys."""
    return {key: [] for key in OUTDICT_KEYS}


# ============================================================
# Helpers for extracting information
# ============================================================

def extract_sd_info(event):
    """Extract reconstructed SD shower info from a ROOT.RecEvent."""
    s_event = event.GetSDEvent()
    s_shower = s_event.GetSdRecShower()

    core_site_CS = list(s_shower.GetCoreSiteCS())
    core_UTM_CS = list(s_shower.GetCoreUTMCS())

    axis_site_CS = list(s_shower.GetAxisSiteCS())
    axis_core_CS = list(s_shower.GetAxisCoreCS())

    return dict(
        gps_sec=s_event.GetGPSSecond(),
        sd_rec_level=s_event.GetRecLevel(),
        lgE=np.log10(s_shower.GetEnergy()),
        theta=np.rad2deg(s_shower.GetZenith()),
        theta_err=np.rad2deg(s_shower.GetZenithError()),
        phi=np.rad2deg(s_shower.GetAzimuth()),
        cos_theta=s_shower.GetCosZenith(),
        cos_theta_err=s_shower.GetCosZenithError(),
        shower_size=s_shower.GetShowerSize(),
        shower_size_err=s_shower.GetShowerSizeError(),
        x_core_site_CS=core_site_CS[0],
        y_core_site_CS=core_site_CS[1],
        z_core_site_CS=core_site_CS[2],
        x_core_UTM_CS=core_UTM_CS[0],
        y_core_UTM_CS=core_UTM_CS[1],
        z_core_UTM_CS=core_UTM_CS[2],
        x_axis_site_CS=axis_site_CS[0],
        y_axis_site_CS=axis_site_CS[1],
        z_axis_site_CS=axis_site_CS[2],
        x_axis_core_CS=axis_core_CS[0],
        y_axis_core_CS=axis_core_CS[1],
        z_axis_core_CS=axis_core_CS[2],
        x_SD_err=s_shower.GetCoreEastingError(),
        y_SD_err=s_shower.GetCoreNorthingError(),
        corr_xy_SD=s_shower.GetCoreNorthingEastingCorrelation(),
        T4_trigg=s_event.GetT4Trigger(),
        T5_trigg=s_event.GetT5Trigger(),
        T4_trigg_name=s_event.ReturnT4Name(),
        hottest_sd_station=s_event.GetT5HottestStation(),
        is6T5=s_event.Is6T5(),
        sd_ldf_status=s_shower.GetLDFStatus(),
    )


def extract_md_info(event):
    """Extract reconstructed MD shower info from a ROOT.RecEvent."""
    m_event = event.GetMDEvent()
    if not m_event.HasRecEvent():
        return None

    md_shower = m_event.GetMdRecShower()
    if not md_shower.IsLdfReconstructed():
        return None

    hottest_counter = m_event.GetHottestCounter()

    core_site_CS_md = list(md_shower.GetCoreSiteCS())
    core_UTM_CS_md = list(md_shower.GetCoreUTMCS())

    # Access to arbitrary LDF values is obtained with
    # md_shower.TabulatedFunction(r)

    return dict(
        beta=md_shower.GetBeta(),
        beta_err=md_shower.GetBetaError(),
        nmu450=md_shower.GetNMuRef(),
        nmu450_err=md_shower.GetNMuRefError(),
        rho450=md_shower.GetMuDensity(),
        rho450_err=md_shower.GetMuDensityError(),
        rho450_cic=md_shower.GetMuDensityCicCorrected(),
        rho450_cic_err=md_shower.GetMuDensityCicCorrectedError(),
        x_core_site_CS_md=core_site_CS_md[0],
        y_core_site_CS_md=core_site_CS_md[1],
        z_core_site_CS_md=core_site_CS_md[2],
        x_core_UTM_CS_md=core_UTM_CS_md[0],
        y_core_UTM_CS_md=core_UTM_CS_md[1],
        z_core_UTM_CS_md=core_UTM_CS_md[2],
        chi2_ldf=md_shower.GetMLDFChi2(),
        ldf_like=md_shower.GetMLDFLikelihood(),
        ndof=md_shower.GetMLDFNdof(),
        deviance=md_shower.GetMLDFDeviance(),
        rec_level=int(m_event.GetRecLevel()),
        hottest_counter=hottest_counter.GetId(),
        r_hottest=hottest_counter.GetSPDistance(),
        is_beta_fixed=md_shower.IsBetaFixed(),
        is_core_fixed=md_shower.IsCoreFixedLdf(),
        is_geometry_reconstructed=md_shower.IsGeometryReconstructed(),
        is_ldf_reconstructed=md_shower.IsLdfReconstructed(),
        isMD6T5=md_shower.GetT5Trigger(),
        is_md_event_saturated=is_md_event_saturated(event),
    )


# ============================================================
# Printing utility
# ============================================================

def print_event_summary(event_id, yymmdd, hhmmss, sd_info, md_info):
    """Print a human-readable summary of one event."""
    print((
        '\n****** MLDF results streamed to ADST ******\n'
        f'{event_id}\n'
        f'yy-mm-dd = {yymmdd}  hh-mm-ss = {hhmmss}  gps_sec = {sd_info["gps_sec"]}\n'
        f'lgE = {sd_info["lgE"]:.2f}. theta = {sd_info["theta"]:.2f}. phi = {sd_info["phi"]:.2f}\n'
        f'hottest_md_counter = {md_info["hottest_counter"]}. hottest_sd_station = {sd_info["hottest_sd_station"]}.\n'
        f'is6T5 = {sd_info["is6T5"]}. isMD6T5 = {md_info["isMD6T5"]}. rec_level = {md_info["rec_level"]}. '
        f'isBetaFixed = {md_info["is_beta_fixed"]}. isCoreFixed = {md_info["is_core_fixed"]}.\n'
        f'isGeomReconstructed = {md_info["is_geometry_reconstructed"]}. is_ldf_reconstructed = {md_info["is_ldf_reconstructed"]}. '
        f'chi2_ldf = {md_info["chi2_ldf"]:.2f}. ndof = {md_info["ndof"]}. ldf_like = {md_info["ldf_like"]:.2f}.\n'
        f'rho450 = {md_info["rho450"]:.3f} / m2. rho450err = {md_info["rho450_err"]:.3f} / m2. '
        f'beta = {md_info["beta"]:.3f}. betaerr = {md_info["beta_err"]:.3f}\n'
        f'nmu450 = {md_info["nmu450"]:.3f}, nmu450err = {md_info["nmu450_err"]:.3f}, '
        f'rho450_cic = {md_info["rho450_cic"]:.3f}, rho450_cic_err = {md_info["rho450_cic_err"]:.3f}'
    ))

# ============================================================
# Main function
# ============================================================

def readADST_data(fname, verbose=True):
    """Read ADST file(s) and extract event-level information.

    This function loads Auger ADST files through the ROOT interface and
    extracts both Surface Detector (SD) and Muon Detector (MD) reconstructed
    event-level quantities. The results are collected into a dictionary that
    can be directly converted into a pandas.DataFrame for further analysis.

    Args:
        fname (str or list of str): Path or list of paths to ADST file(s).
        verbose (bool, optional): If True, print detailed information for each
            event during processing. Defaults to True.

    Returns:
        dict: A dictionary containing event-level quantities. Each key
        corresponds to a physical observable or reconstruction parameter,
        with values stored as lists aligned by event index. Example keys:

            - "event_id" (str): Event identifier.
            - "theta", "theta_err" (float): Shower zenith angle [deg] and error.
            - "phi" (float): Shower azimuth [deg].
            - "cos_theta", "cos_theta_err" (float): Cosine of zenith angle and error.
            - "lgE" (float): Log10 of reconstructed primary energy [eV].
            - "shower_size", "shower_size_err" (float): SD shower size and error.
            - "core_site_CS", "core_UTM_CS" (list[float]): Reconstructed core position (site and UTM coordinates).
            - "beta", "beta_err" (float): MLDF slope parameter and uncertainty.
            - "rho450", "rho450_err" (float): Reconstructed muon density at 450 m and error.
            - "nmu450", "nmu450_err" (float): Reconstructed muon number at 450 m and error.
            - "rho450_cic", "rho450_cic_err" (float): CIC-corrected muon density at 450 m and error.
            - "chi2_ldf", "ldf_like", "ndof", "deviance" (float): MLDF reconstruction quality metrics.
            - "is6T5", "isMD6T5" (bool): SD/MD trigger conditions.
            - "is_beta_fixed", "is_core_fixed" (bool): Fit constraints applied in MD reconstruction.
            - "is_md_event_saturated" (bool): Whether at least one MD module was saturated.
            - "hottest_sd_station", "hottest_counter" (int): IDs of hottest SD station and MD counter.
            - "yymmdd", "hhmmss", "gps_sec" (int): Event time information.

    Notes:
        - Requires Auger Offline environment (`$AUGEROFFLINEROOT`) to be set
          and ADST shared libraries available.
        - If `verbose=True`, summary information is printed per event.
        - Events without MD reconstruction are skipped.
        - Events with failed MLDF fits (e.g., geometry not reconstructed)
          are also skipped.

    Example:
        >>> data = readADST_data("mydata.ADST.root", verbose=False)
        >>> import pandas as pd
        >>> df = pd.DataFrame(data)
        >>> print(df.head())
    """
    # Load Auger Offline environment
    auger_offline_root = os.environ["AUGEROFFLINEROOT"]
    ROOT.gSystem.Load(os.path.join(auger_offline_root, "lib/libRecEventKG.so"))

    # Handle input file(s)
    data_files = fname if isinstance(fname, list) else [fname]
    files = ROOT.std.vector('string')()
    for datfile in data_files:
        files.push_back(datfile)

    # Initialize ROOT classes
    rec_file = ROOT.RecEventFile(files)
    event = ROOT.RecEvent()
    geo = ROOT.DetectorGeometry()
    rec_file.ReadDetectorGeometry(geo)
    rec_file.SetBuffers(event)

    outdict = init_outdict()

    # Loop over events
    while rec_file.ReadNextEvent() == ROOT.RecEventFile.eSuccess:
        event_id = str(event.GetEventId())
        yymmdd = event.GetYYMMDD()
        hhmmss = str(event.GetHHMMSS()).zfill(6)

        sd_info = extract_sd_info(event)
        md_info = extract_md_info(event)

        if md_info is None:
            if verbose:
                print(f"{event_id} has no valid MD reconstruction. Skipping..")
            continue

        if verbose:
            print_event_summary(event_id, yymmdd, hhmmss, sd_info, md_info)

        # sanity check 1: avoid overlapping of keys between dictionaries before
        # merging them into a single dictionary

        overlap = sd_info.keys() & md_info.keys()
        if overlap:
            raise ValueError(f"Overlapping keys found: {overlap}")

        # Merge and append results (** operator unpacks the dictionaries)
        merged = {**sd_info, **md_info,
                  "event_id": event_id, "yymmdd": yymmdd, "hhmmss": hhmmss}
        for key in OUTDICT_KEYS:
            # sanity check 2: assert that the key exists in the merged dictionary
            # (otherwise I would be silently appending a None)
            assert key in merged.keys(), f"key {key} is not present in sd_info or md_info"

            outdict[key].append(merged.get(key, None))

    return outdict

def get_module_list(counter):
    """Retrieve the list of modules present in a given MdRecCounter object.

    Args:
        counter (ROOT.MdRecCounter): Counter object containing module information.

    Returns:
        list: List of module objects corresponding to the available module IDs.
        Only modules for which `counter.HasModule(mod_id)` is True are included.
    """
    # Define the possible module IDs
    # For SiPM data: range(100, 116)
    possible_modules = range(0, 55)  # For PMT data

    # Collect the available modules
    modules = []
    # Note: silent modules are not included since Counter.HasRecData yields False in MD2ADST.cc
    for mod_id in possible_modules:
        if counter.HasModule(mod_id):
            modules.append(counter.GetModule(mod_id))

    return modules


def get_channel_list(module):
    """
    Arguments:
        module (ROOT.MdRecModule): MdRecModule object
    """
    channels = []
    for chId in range(1, 65):
        if module.HasChannel(chId):
            channels.append(module.GetChannel(chId))

    return channels

def is_md_event_saturated(event):
    """Check whether an MD event has at least one saturated module or counter.

    Args:
        event (ROOT.RecEvent): Event object containing SD and MD data.

    Returns:
        bool: True if at least one counter or module is saturated, False otherwise.
    """
    saturation_flag = False


    # Muon detector (MD) event
    m_event = event.GetMDEvent()
    counter_vector = m_event.GetCounterVector()

    for counter in counter_vector:

        # Check if counter itself is saturated
        if counter.IsSaturated():
            saturation_flag = True
            break

        # Check if any module inside the counter is saturated
        modules = get_module_list(counter)
        for mod in modules:
            if mod.IsSaturated():
                saturation_flag = True
                break

        if saturation_flag:
            break

    return saturation_flag

# =============================================
## Validation of binary traces at channel level

def find_1x1_patterns(trace, w):
    """
    TODO note that this function can be easily generalized for any
    type of patterns, not just for 1x1

    Find indices where the pattern 1 x 1 occurs in a binary array.
    After finding a match, skip searching for the next `w` elements.

    Args:
        trace (list[int] or np.ndarray): Array of 0s and 1s.
        w (int): Window size to skip after each found pattern.

    Returns:
        list[int]: Indices of the first '1' in each detected 1x1 pattern.
    """
    trace = np.asarray(trace, dtype=int)
    indices = []
    i = 0
    while i < len(trace) - 2:
        if trace[i] == 1 and trace[i+2] == 1:
            indices.append(i)
            i += w  # skip next w elements
        else:
            i += 1
    return indices

def find_111_patterns(trace, w):
    """

    Find indices where the pattern 1 1 1 occurs in a binary array.
    After finding a match, skip searching for the next `w` elements.

    Args:
        trace (list[int] or np.ndarray): Array of 0s and 1s.
        w (int): Window size to skip after each found pattern.

    Returns:
        list[int]: Indices of the first '1' in each detected 111 pattern.

    Examples:
    >>> find_111_patterns([0, 1, 1, 1, 1, 1, 0], w=3)
    [1]

    >>> find_111_patterns([1, 1, 1, 1, 1, 1], w=7)
    [0]
    """
    trace = np.asarray(trace, dtype=int)
    indices = []
    i = 0
    while i < len(trace) - 2:
        if trace[i] == 1 and trace[i+1] == 1 and trace[i+2] == 1:
            indices.append(i)
            i += w  # skip next w elements
        else:
            i += 1
    return indices

def estimate_number_of_muons(w, start_times_module=None, ns=64):
    """
    Estimate the number of muons in a detector module from the start times
    of detected patterns (defined elsewhere, not in this function),
    using a window-based counting method.

    The trace of length 1024 bins is divided into windows of size `w`.
    For each window, the number of channels with at least one start time
    in that window is counted, and the muon count is estimated using
    the occupancy formula:

        n_muons(window) = -ns * log(1 - k/ns)

    where `k` is the number of channels that fired in the window,
    and `ns` is the segmentation of the detector. The total number of muons
    is obtained by summing over all windows.

    Note:
        Because the pattern definition enforces an inhibition window equal
        to `w`, each channel can contribute at most once per window. This
        ensures no double-counting of multiple start times in the same
        channel within a single window.

    Arguments:
        start_times_module (dict[int, list[int]]): Dictionary mapping channel
            IDs to lists of start times (in bins) where muon patterns were detected.
            Example: start_times_module[3] = [325, 422] means channel 3 had
            two patterns starting at bins 325 and 422.
        w (int): Window size (number of bins per window).
        ns (int): Number of segments.

    Returns:
        tuple:
            n_muons_total (float): Estimated total number of muons in the module.
            channels_on_per_window (numpy.ndarray): Number of channels firing
                in each window.
    """
    TRACE_LENGTH = 1024

    # number of windows
    #nw = math.ceil(TRACE_LENGTH / w)
    nw = int(TRACE_LENGTH / w)

    #print(f"number of windows = {nw}")

    channels_on_per_window = np.zeros(nw)
    #print(f"len(chanels_on_per_window) = {len(channels_on_per_window)}")

    # It stores the start times in each window
    channels_on_per_window_verbose = {ww: [] for ww in range(nw)}

    # Iterate over time windows
    # If the pattern-finding algorithm itself already
    # enforces an inhibition window equal to w,
    # then by construction each channel can contribute at most once per window.
    for window in range(nw):
        idx1 = window * w
        idx2 = idx1 + w - 1
        #print(f"window = {window}, indexes = {idx1}, {idx2}")

        # iterate over channels
        for ch_id, start_times in start_times_module.items():
            for t in start_times:
                if idx1 <= t <= idx2:
                    channels_on_per_window[window] += 1
                    channels_on_per_window_verbose[window].append(t)

    ## Compute number of muons in each window and sum
    n_muons_total = 0
    for window, k in enumerate(channels_on_per_window):
        if k == ns:
            n_muons_total = float("inf")
            break
        else:
            n_muons_total -= ns*np.log(1-k/ns)

    #print(f"channels_on_per_window = {channels_on_per_window}")
    #print(f"channels_on_per_window_verbose = {channels_on_per_window_verbose}")


    return n_muons_total, channels_on_per_window


def validate_111_patterns(fname, w, outfile, verbose=True, print_traces=False):
    """
    Validate the Offline reconstruction of 111 muon patterns against a
    custom Python implementation. At the end of the run, a summary of
    traces with errors (mismatches between what is stored in ADST and my python
    implementation) is shown.

    The validation is performed at two levels:

    1) Channel-level:
       - For each channel, searches for 111 patterns using the Python
         implementation `find_111_patterns()` with inhibition window `w`.
       - Compares the number and start times of detected patterns with
         those stored in the ADST.
       - Reports mismatches and, if requested, prints full traces with
         annotated pattern matches.

    2) Module-level:
       - Combines channel-level results into module-level information.
       - Computes the number of channels with pattern starts in each global
         time window of size `w`.
       - Estimates the number of muons using `estimate_number_of_muons()`.
       - Compares the resulting `channelsOnPerWindow` array and the total
         number of muons with the corresponding quantities stored in the ADST.

    Arguments:
        fname (str or list[str]): Path(s) to the ADST file(s) to be validated.
        w (int): Window size (in bins). Used both as the inhibition window
            for channel-level pattern finding and as the size of the global
            time windows at module level.
        outfile (file object): Output file with prints relevant to the validation
        verbose (bool, optional): If True, print information about skipped
            or invalid events/modules. Default is True.
        print_traces (bool, optional): If True, print the full channel trace
            with pattern matches annotated. Default is False.

    Returns:
        None.
        The function performs validation by printing mismatches, errors,
        and summary information to outfile. It also raises `AssertionError`
        if critical mismatches are found.

    Notes:
        - Assumes each channel trace has length 1024 bins.
    """
     # Load Auger Offline environment
    auger_offline_root = os.environ["AUGEROFFLINEROOT"]
    ROOT.gSystem.Load(os.path.join(auger_offline_root, "lib/libRecEventKG.so"))

    # Handle input file(s)
    data_files = fname if isinstance(fname, list) else [fname]
    files = ROOT.std.vector('string')()
    for datfile in data_files:
        files.push_back(datfile)

    # Initialize ROOT classes
    rec_file = ROOT.RecEventFile(files)
    event = ROOT.RecEvent()
    geo = ROOT.DetectorGeometry()
    rec_file.ReadDetectorGeometry(geo)
    rec_file.SetBuffers(event)

    # number of traces with a mismatch between Offline & my python algorithm
    n_errors = 0
    # total number of analyzed traces
    n_analzed_traces = 0

    #
    errors = {"time_bin_1021":
              {"counter": [], "module": [], "channel": [],
               "event": [], "yymmdd": [], "hhmmss": [],
               "trace": []},
               "other":
               {"counter": [], "module": [], "channel": [],
               "event": [], "yymmdd": [], "hhmmss": [],
               "trace": []}
                }

    # Loop over events
    while rec_file.ReadNextEvent() == ROOT.RecEventFile.eSuccess:

        event_id = str(event.GetEventId())
        yymmdd = event.GetYYMMDD()
        hhmmss = str(event.GetHHMMSS()).zfill(6)

        sd_info = extract_sd_info(event)
        md_info = extract_md_info(event)

        if md_info is None:
            if verbose:
                outfile.write(f"{event_id} has no valid MD reconstruction. Skipping..\n")
            continue

        m_event = event.GetMDEvent()

        # counters
        counter_vector = m_event.GetCounterVector()

         # loop over counters
        for counter in counter_vector:
            counter_id = counter.GetId()

            modules = get_module_list(counter)

            for mod in modules:
                if mod.IsRejected():
                    continue

                # Apparently, there seems to be a bug in Offline in which sequences "110" starting
                # at time bin 1021 are counted as muon patterns (for the 1x1 case). I found this in some traces, although
                # at first glance it does not seem to be too frequent.
                # If that is the case, the following module-level checks, in which I compare the channelsOnPerWindow,
                # and the numberOfMuons stored in the ADST with that computed with my own library will fail.
                # Therefore, I define this boolean flag to catch these cases.
                mismatch_expected = False

                start_time_module = {chId: [] for chId in range(1, 65)}
                mod_id = mod.GetId()
                n_mu_uncorrected = mod.GetNumberOfEstimatedMuonsUncorrected()
                ns = mod.GetSegmentation()
                outfile.write(f" \n--- Counter {counter_id} -  mod {mod_id}, n_mu_uncorr = {n_mu_uncorrected:.3f}, ns = {ns}\n")
                channels = get_channel_list(mod)
                for ch in channels:
                    if ch.IsMasked():
                        continue
                    if not ch.HasTrace():
                        continue

                    trace_ = ch.GetTrace()
                    trace = np.array([int(ord(sample)) for sample in trace_])

                    # if trace is full of zeros, continue
                    if np.sum(trace) == 0:
                        continue

                    n_analzed_traces += 1

                    # obtain start times of muon patterns with my own library
                    start_times = np.array(find_111_patterns(trace, w=w))
                    # storing start times in start_times_module
                    start_time_module[ch.GetId()] += list(start_times)

                    # pattern match start times from the ADST
                    n_matches_adst = ch.GetNumberOfPatternMatchs()
                    pattern_start_times_adst = np.array(ch.GetPatternMatchTimes())

                    if len(start_times) != n_matches_adst or (len(start_times) == n_matches_adst and
                                                              not (start_times == pattern_start_times_adst).all()):

                        # elements that are in _py but not in ADST
                        diff_py_adst = np.setdiff1d(start_times, pattern_start_times_adst)
                        # elements that are in  ADST but not in _py
                        diff_adst_py = np.setdiff1d(pattern_start_times_adst, start_times)
                        # elements that are in one array but not the other (symmetric difference)
                        symmetric_diff = np.setxor1d(start_times, pattern_start_times_adst)

                        outfile.write((f"ERROR in Counter {counter_id} -  mod {mod_id} - ch {ch.GetId()} "
                                f"in event {event_id}, yymmdd = {yymmdd}, hhmmss = {hhmmss}.: len(start_times) != n_matches_adst\n"
                                f"len(start_times) = {len(start_times)}, n_matches_adst = {n_matches_adst}\n"
                                f"start_times_py = {start_times}, start_times_adst = {pattern_start_times_adst}.\n"
                                f"elements that are in one array but not the other = {symmetric_diff}\n"
                                f"elements in _py but not in ADST = {diff_py_adst}, elements in ADST but not in py = {diff_adst_py}.\n"
                                ))

                        outfile.write(" Printing trace:\n")
                        for tbin, sample in enumerate(trace):
                            if (tbin in start_times) and (tbin in pattern_start_times_adst):
                                outfile.write(f"t = {tbin}, {sample} <--- pattern match\n")
                            elif (tbin in start_times) and (tbin not in pattern_start_times_adst):
                                outfile.write(f"t = {tbin}, {sample} <--- pattern match in py but NOT in ADST\n")
                            elif (tbin not in start_times) and (tbin in pattern_start_times_adst):
                                outfile.write(f"t = {tbin}, {sample} <--- pattern match in ADST but NOT in py\n")
                            else:
                                outfile.write(f"t = {tbin}, {sample}\n")

                        # classifying the error
                        # Offline bug -> pattern start time at time bin 1021
                        if ((n_matches_adst == len(start_times) + 1) and
                            (symmetric_diff[0] == 1021) and
                            (diff_adst_py[0] == 1021)):

                            errors["time_bin_1021"]["counter"].append(counter_id)
                            errors["time_bin_1021"]["module"].append(mod_id)
                            errors["time_bin_1021"]["channel"].append(ch.GetId())
                            errors["time_bin_1021"]["event"].append(event_id)
                            errors["time_bin_1021"]["yymmdd"].append(yymmdd)
                            errors["time_bin_1021"]["hhmmss"].append(hhmmss)
                            errors["time_bin_1021"]["trace"].append(trace)

                        else:
                            errors["other"]["counter"].append(counter_id)
                            errors["other"]["module"].append(mod_id)
                            errors["other"]["channel"].append(ch.GetId())
                            errors["other"]["event"].append(event_id)
                            errors["other"]["yymmdd"].append(yymmdd)
                            errors["other"]["hhmmss"].append(hhmmss)
                            errors["other"]["trace"].append(trace)

                        n_errors += 1
                        mismatch_expected = True

                        continue


                    # The following channel-level asserts will only be executed
                    # if len(start_times) == n_matches_adst
                    assert len(start_times) == n_matches_adst

                    assert len(pattern_start_times_adst) == n_matches_adst

                    assert len(start_times) == len(pattern_start_times_adst)

                    assert len(trace) == 1024

                    #print(f"start_times_python = {start_times}, "
                            #f"start_times_offline = {pattern_start_times_adst}")

                    ## Assuring that my start times coincide exactly to those in the ADST

                    assert (start_times == pattern_start_times_adst).all(), (f"start_times_py = {start_times}, "
                                                                             f"start_times_adst = {pattern_start_times_adst}")

                    if print_traces:
                        for tbin, sample in enumerate(trace):
                            if tbin in start_times:
                                outfile.write(f"t = {tbin}, {sample} <--- pattern match\n")
                            else:
                                outfile.write(f"t = {tbin}, {sample}\n")



                #print(f"start_times_module_py: {start_time_module}")
                nmu_uncorrected_pyhton, channels_on_per_window_py = estimate_number_of_muons(start_times_module=start_time_module,
                                                                                             w=w,ns=ns)

                outfile.write(f" --- Counter {counter_id} -  mod {mod_id}, "
                      f"n_mu_uncorr = {n_mu_uncorrected:.3f}, "
                      f"n_mu_uncorr_py = {nmu_uncorrected_pyhton:.3f}, ns = {ns}\n")


                channels_on_adst = np.array(mod.GetChannelsOn())

                # check that the channels on per window calculated with my
                # own library coincides with that stored in the ADST (if
                # no mismatch is expected)
                if not (channels_on_adst == channels_on_per_window_py).all() and not mismatch_expected:
                    outfile.write("MISMATCH IN CHANNELSOnPerWindow, but no mismatch is expected. ")
                    outfile.write(f"channelsOnPerWindowADST = {channels_on_adst}")
                    outfile.write(f"channelsOnPerWindowPython = {channels_on_per_window_py}\n")
                    raise Exception("MISMATCH IN CHANNELSOnPerWindow, but no mismatch is expected")

                if not np.isclose(nmu_uncorrected_pyhton, n_mu_uncorrected) and not mismatch_expected:
                    outfile.write(f"MISMATCH IN NUMBER OF MUONS, but no mismatch is expected, "
                          f"len(channels_on_adst) = {len(channels_on_adst)}, "
                          f"n_mu_uncorr = {n_mu_uncorrected:.3f}, "
                          f"n_mu_uncorr_py = {nmu_uncorrected_pyhton:.3f}\n")
                    raise Exception("MISMATCH IN NUMBER OF MUONS, but no mismatch is expected")


    outfile.write("\n --- SUMMARY OF TRACES ERRORS ----\n")
    outfile.write(f"n_analyzed_traces = {n_analzed_traces}\n"
          f"n_errors = {n_errors}\n")
    outfile.write("\n --- VERBOSE INFO OF TRACES ERRORS ----\n")

    if n_errors:
        for error_type in errors.keys():
            outfile.write(f" *) ERROR TYPE = {error_type}\n")
            for i, _ in enumerate(errors[error_type]["counter"]):
                outfile.write(f"c = {errors[error_type]['counter'][i]}, "
                    f"mod = {errors[error_type]['module'][i]}, "
                    f"ch = {errors[error_type]['channel'][i]}, "
                    f"evt = {errors[error_type]['event'][i]}, "
                    f"yymmdd = {errors[error_type]['yymmdd'][i]}, "
                    f"hhmmss = {errors[error_type]['hhmmss'][i]}\n")

def validate_1x1_patterns(fname, w, outfile, verbose=True, print_traces=False):
    """
    Validate the Offline reconstruction of 1x1 muon patterns against a
    custom Python implementation. At the end of the run, a summary of
    traces with errors (mismatches between what is stored in ADST and my python
    implementation) is shown.

    The validation is performed at two levels:

    1) Channel-level:
       - For each channel, searches for 1x1 patterns using the Python
         implementation `find_1x1_patterns()` with inhibition window `w`.
       - Compares the number and start times of detected patterns with
         those stored in the ADST.
       - Reports mismatches and, if requested, prints full traces with
         annotated pattern matches.

    2) Module-level:
       - Combines channel-level results into module-level information.
       - Computes the number of channels with pattern starts in each global
         time window of size `w`.
       - Estimates the number of muons using `estimate_number_of_muons()`.
       - Compares the resulting `channelsOnPerWindow` array and the total
         number of muons with the corresponding quantities stored in the ADST.

    Arguments:
        fname (str or list[str]): Path(s) to the ADST file(s) to be validated.
        w (int): Window size (in bins). Used both as the inhibition window
            for channel-level pattern finding and as the size of the global
            time windows at module level.
        outfile (file object): output file
        verbose (bool, optional): If True, print information about skipped
            or invalid events/modules. Default is True.
        print_traces (bool, optional): If True, print the full channel trace
            with pattern matches annotated. Default is False.

    Returns:
        None.
        The function performs validation by printing mismatches, errors,
        and summary information to the outfile. It also raises `AssertionError`
        if critical mismatches are found.

    Notes:
        - There appears to be a bug in Offline in which sequences "110"
          starting at time bin 1021 are counted as muon patterns. When this
          occurs (not so frequent), the module-level comparisons (channelsOnPerWindow and
          numberOfMuons) are expected to fail, and such cases are flagged
          internally with `mismatch_expected = True`.
        - There is also one trace in the PMT dataset in which a pattern match
        is missed by the Offline algorithm.
        - Assumes each channel trace has length 1024 bins.
    """
     # Load Auger Offline environment
    auger_offline_root = os.environ["AUGEROFFLINEROOT"]
    ROOT.gSystem.Load(os.path.join(auger_offline_root, "lib/libRecEventKG.so"))

    # Handle input file(s)
    data_files = fname if isinstance(fname, list) else [fname]
    files = ROOT.std.vector('string')()
    for datfile in data_files:
        files.push_back(datfile)

    # Initialize ROOT classes
    rec_file = ROOT.RecEventFile(files)
    event = ROOT.RecEvent()
    geo = ROOT.DetectorGeometry()
    rec_file.ReadDetectorGeometry(geo)
    rec_file.SetBuffers(event)

    # number of traces with a mismatch between Offline & my python algorithm
    n_errors = 0
    # total number of analyzed traces
    n_analzed_traces = 0

    # error classification. For the 1x1 pattern, there is a known
    errors = {"time_bin_1021":
              {"counter": [], "module": [], "channel": [],
               "event": [], "yymmdd": [], "hhmmss": [],
               "trace": []},
               "other":
               {"counter": [], "module": [], "channel": [],
               "event": [], "yymmdd": [], "hhmmss": [],
               "trace": []}
                }

    # Loop over events
    while rec_file.ReadNextEvent() == ROOT.RecEventFile.eSuccess:

        event_id = str(event.GetEventId())
        yymmdd = event.GetYYMMDD()
        hhmmss = str(event.GetHHMMSS()).zfill(6)

        sd_info = extract_sd_info(event)
        md_info = extract_md_info(event)

        if md_info is None:
            if verbose:
                outfile.write(f"{event_id} has no valid MD reconstruction. Skipping..\n")
            continue

        m_event = event.GetMDEvent()

        # counters
        counter_vector = m_event.GetCounterVector()

         # loop over counters
        for counter in counter_vector:
            counter_id = counter.GetId()

            modules = get_module_list(counter)

            for mod in modules:
                if mod.IsRejected():
                    continue

                # Apparently, there seems to be a bug in Offline in which sequences "110" starting
                # at time bin 1021 are counted as muon patterns. I found this in some traces, although
                # at first glance it does not seem to be too frequent.
                # If that is the case, the following module-level checks, in which I compare the channelsOnPerWindow,
                # and the numberOfMuons stored in the ADST with that computed with my own library will fail.
                # Therefore, I define this boolean flag to catch these cases.
                mismatch_expected = False

                start_time_module = {chId: [] for chId in range(1, 65)}
                mod_id = mod.GetId()
                n_mu_uncorrected = mod.GetNumberOfEstimatedMuonsUncorrected()
                ns = mod.GetSegmentation()
                outfile.write(f" \n--- Counter {counter_id} -  mod {mod_id}, n_mu_uncorr = {n_mu_uncorrected:.3f}, ns = {ns}\n")
                channels = get_channel_list(mod)
                for ch in channels:
                    if ch.IsMasked():
                        continue
                    if not ch.HasTrace():
                        continue

                    trace_ = ch.GetTrace()
                    trace = np.array([int(ord(sample)) for sample in trace_])

                    # if trace is full of zeros, continue
                    if np.sum(trace) == 0:
                        continue

                    n_analzed_traces += 1

                    # obtain start times of muon patterns with my own library
                    start_times = np.array(find_1x1_patterns(trace, w=w))
                    # storing start times in start_times_module
                    start_time_module[ch.GetId()] += list(start_times)

                    # pattern match start times from the ADST
                    n_matches_adst = ch.GetNumberOfPatternMatchs()
                    pattern_start_times_adst = np.array(ch.GetPatternMatchTimes())

                    if len(start_times) != n_matches_adst:

                        # elements that are in _py but not in ADST
                        diff_py_adst = np.setdiff1d(start_times, pattern_start_times_adst)
                        # elements that are in  ADST but not in _py
                        diff_adst_py = np.setdiff1d(pattern_start_times_adst, start_times)
                        # elements that are in one array but not the other (symmetric difference)
                        symmetric_diff = np.setxor1d(start_times, pattern_start_times_adst)

                        outfile.write((f"ERROR in Counter {counter_id} -  mod {mod_id} - ch {ch.GetId()} "
                                f"in event {event_id}, yymmdd = {yymmdd}, hhmmss = {hhmmss}.: len(start_times) != n_matches_adst\n"
                                f"len(start_times) = {len(start_times)}, n_matches_adst = {n_matches_adst}\n"
                                f"start_times_py = {start_times}, start_times_adst = {pattern_start_times_adst}.\n"
                                f"elements that are in one array but not the other = {symmetric_diff}\n"
                                f"elements in _py but not in ADST = {diff_py_adst}, elements in ADST but not in py = {diff_adst_py}.\n"
                                ))

                        outfile.write(" Printing trace:\n")
                        for tbin, sample in enumerate(trace):
                            if (tbin in start_times) and (tbin in pattern_start_times_adst):
                                outfile.write(f"t = {tbin}, {sample} <--- pattern match\n")
                            elif (tbin in start_times) and (tbin not in pattern_start_times_adst):
                                outfile.write(f"t = {tbin}, {sample} <--- pattern match in py but NOT in ADST\n")
                            elif (tbin not in start_times) and (tbin in pattern_start_times_adst):
                                outfile.write(f"t = {tbin}, {sample} <--- pattern match in ADST but NOT in py\n")
                            else:
                                outfile.write(f"t = {tbin}, {sample}\n")

                        # classifying the error
                        # Offline bug -> pattern start time at time bin 1021
                        if ((n_matches_adst == len(start_times) + 1) and
                            (symmetric_diff[0] == 1021) and
                            (diff_adst_py[0] == 1021)):

                            errors["time_bin_1021"]["counter"].append(counter_id)
                            errors["time_bin_1021"]["module"].append(mod_id)
                            errors["time_bin_1021"]["channel"].append(ch.GetId())
                            errors["time_bin_1021"]["event"].append(event_id)
                            errors["time_bin_1021"]["yymmdd"].append(yymmdd)
                            errors["time_bin_1021"]["hhmmss"].append(hhmmss)
                            errors["time_bin_1021"]["trace"].append(trace)

                        else:
                            errors["other"]["counter"].append(counter_id)
                            errors["other"]["module"].append(mod_id)
                            errors["other"]["channel"].append(ch.GetId())
                            errors["other"]["event"].append(event_id)
                            errors["other"]["yymmdd"].append(yymmdd)
                            errors["other"]["hhmmss"].append(hhmmss)
                            errors["other"]["trace"].append(trace)

                        n_errors += 1
                        mismatch_expected = True

                        continue


                    # The following channel-level asserts will only be executed
                    # if len(start_times) == n_matches_adst
                    assert len(start_times) == n_matches_adst

                    assert len(pattern_start_times_adst) == n_matches_adst

                    assert len(start_times) == len(pattern_start_times_adst)

                    assert len(trace) == 1024

                    #print(f"start_times_python = {start_times}, "
                            #f"start_times_offline = {pattern_start_times_adst}")

                    ## Assuring that my start times coincide exactly to those in the ADST
                    assert (start_times == pattern_start_times_adst).all()

                    if print_traces:
                        for tbin, sample in enumerate(trace):
                            if tbin in start_times:
                                outfile.write(f"t = {tbin}, {sample} <--- pattern match\n")
                            else:
                                outfile.write(f"t = {tbin}, {sample}\n")



                #print(f"start_times_module_py: {start_time_module}")
                nmu_uncorrected_pyhton, channels_on_per_window_py = estimate_number_of_muons(start_times_module=start_time_module, w=w,
                                                                                             ns=ns)

                outfile.write(f" --- Counter {counter_id} -  mod {mod_id}, "
                      f"n_mu_uncorr = {n_mu_uncorrected:.3f}, "
                      f"n_mu_uncorr_py = {nmu_uncorrected_pyhton:.3f}, ns = {ns}\n")


                channels_on_adst = np.array(mod.GetChannelsOn())

                # check that the channels on per window calculated with my
                # own library coincides with that stored in the ADST (if
                # no mismatch is expected)
                if not (channels_on_adst == channels_on_per_window_py).all() and not mismatch_expected:
                    outfile.write("MISMATCH IN CHANNELSOnPerWindow, but no mismatch is expected\n")
                    outfile.write(f"channelsOnPerWindowADST = {channels_on_adst}\n")
                    outfile.write(f"channelsOnPerWindowPython = {channels_on_per_window_py}\n")
                    raise Exception("MISMATCH IN CHANNELSOnPerWindow, but no mismatch is expected")

                if not np.isclose(nmu_uncorrected_pyhton, n_mu_uncorrected) and not mismatch_expected:
                    outfile.write(f"MISMATCH IN NUMBER OF MUONS, but no mismatch is expected, "
                          f"len(channels_on_adst) = {len(channels_on_adst)}, "
                          f"n_mu_uncorr = {n_mu_uncorrected:.3f}, "
                          f"n_mu_uncorr_py = {nmu_uncorrected_pyhton:.3f}\n")
                    raise Exception("MISMATCH IN NUMBER OF MUONS, but no mismatch is expected")


    outfile.write("\n --- SUMMARY OF TRACES ERRORS ----\n")
    outfile.write(f"n_analyzed_traces = {n_analzed_traces}\n"
          f"n_errors = {n_errors}\n")
    outfile.write("\n --- VERBOSE INFO OF TRACES ERRORS ----\n")

    if n_errors:
        for error_type in errors.keys():
            outfile.write(f" *) ERROR TYPE = {error_type}\n")
            for i, _ in enumerate(errors[error_type]["counter"]):
                outfile.write(f"c = {errors[error_type]['counter'][i]}, "
                    f"mod = {errors[error_type]['module'][i]}, "
                    f"ch = {errors[error_type]['channel'][i]}, "
                    f"evt = {errors[error_type]['event'][i]}, "
                    f"yymmdd = {errors[error_type]['yymmdd'][i]}, "
                    f"hhmmss = {errors[error_type]['hhmmss'][i]}\n")
