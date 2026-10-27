import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ROOT

# Add custom module paths
sys.path.append('/storage/gpfs_data/pauger/users/jdejesus/my_python_modules/')
from myClasses import Profile
from myUsefulFunctions import generate_colormap

sys.path.append('/storage/gpfs_data/pauger/users/jdejesus/my_python_modules/auger')
from augerADSTgeneral import extract_sd_info, extract_md_info
import augerDetector as ad

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

def get_total_number_of_channels_on(module):
    """
    Count the number of active, usable channels in an MD module.

    A channel is counted if it:

      - contains a trace,
      - is not masked, and
      - contains at least one pattern match.

    Each channel contributes at most one unit to the returned count,
    independently of the number of pattern matches found in that channel.

    Parameters
    ----------
    module : ROOT MD module object
        Reconstructed MD module providing access to its channels through
        ``GetChannelsVector()``.

    Returns
    -------
    int
        Number of unmasked channels with a trace and at least one pattern
        match.
    """

    channels = get_channel_list(module)

    k = 0

    for ch in channels:
        if (ch.HasTrace() and not ch.IsMasked() and ch.GetNumberOfPatternMatchs()):
            k += 1
    return k

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
    possible_modules_sipm = list(range(100, 117))
    possible_modules = possible_modules_sipm

    # Collect the available modules
    modules = []
    # Note: silent modules are not included since Counter.HasRecData yields False in MD2ADST.cc
    for mod_id in possible_modules:
        if counter.HasModule(mod_id):
            modules.append(counter.GetModule(mod_id))

    return modules



def get_event_relevant_module_level_info(event, counters_to_analyze={1815, 1771, 643, 1816}):
    """
    Extract selected event-, counter-, and module-level observables.

    The function processes one reconstructed event and collects information
    for every non-rejected MD module belonging to one of the selected
    counters. Event- and shower-level quantities are repeated for every
    accepted module, so that the returned dictionary can be converted
    directly into a row-oriented pandas DataFrame.

    An MD counter is included only if:

      - its identifier belongs to ``counters_to_analyze``, and
      - the counter is not rejected.

    Within each selected counter, rejected modules are skipped. For every
    accepted module, the function stores its geometry, segmentation,
    reconstructed muon content, corner-clipping probability, active area,
    number of active channels, and information from the associated SD
    station.

    Parameters
    ----------
    event : ROOT event object
        ADST event containing the reconstructed SD and MD information.

    counters_to_analyze : collections.abc.Collection of int, optional
        Identifiers of the MD counters to process.

    Returns
    -------
    dict[str, list] or None
        Dictionary of equally sized lists, with one entry per accepted MD
        module. It contains the following fields:

        Event and shower information:

        - ``event_id``: event identifier.
        - ``yymmdd``: event date.
        - ``gps``: event GPS time.
        - ``lge``: logarithm of the reconstructed shower energy.
        - ``theta``: reconstructed shower zenith angle.
        - ``phi``: reconstructed shower azimuth.
        - ``sd_shower_size``: reconstructed SD shower-size observable.
        - ``sd_shower_size_err``: uncertainty in the SD shower size.
        - ``sd_shower_size_label``: name of the shower-size observable.
        - ``sd_shower_size_att``: attenuation-corrected shower size.
        - ``sd_shower_size_att_err``: uncertainty in the corrected size.
        - ``is6T5``: 6T5 quality-trigger flag.
        - ``xc`` and ``yc``: reconstructed core coordinates in the UTM
          coordinate system.

        Counter and module information:

        - ``counter_id``: MD counter identifier.
        - ``module_id``: MD module identifier.
        - ``r``: module distance from the shower axis.
        - ``phi_sp``: module azimuth in shower-plane coordinates.
        - ``n_segments``: number of segments in the module.
        - ``active_area``: active area of the module.
        - ``pcc``: module corner-clipping probability.
        - ``phi0``: module orientation.
        - ``dphi``: folded angular difference between the shower azimuth
          and the module orientation.

        Muon observables:

        - ``k``: number of unmasked channels containing a trace and at
          least one pattern match.
        - ``nmu``: corrected reconstructed number of muons.
        - ``nmu_uncorrected``: reconstructed number of muons before the
          module-level corrections.

        Associated SD-station information:

        - ``sd_signal``: total signal measured by the associated SD
          station.
        - ``sd_signal_err``: uncertainty in the SD signal.
        - ``r_station``: distance of the associated SD station from the
          shower axis.

        If no associated SD station is found, ``sd_signal``,
        ``sd_signal_err``, and ``r_station`` are set to ``-1``.

        The function returns ``None`` if the event has no reconstructed MD
        event or if the MD lateral-distribution reconstruction is
        unavailable. If the event is valid but no module passes the
        selection, a dictionary containing empty lists is returned.

    Notes
    -----
    The function relies on ``extract_sd_info()`` for the event-level SD.

    The quantities ``kind`` and ``photodetector``, used to obtain the module
    orientation, must currently exist in the surrounding scope. Preferably,
    they should be added explicitly as function parameters.
    """


    m_event = event.GetMDEvent()
    if not m_event.HasRecEvent():
        return None

    md_rec_shower = m_event.GetMdRecShower()
    if not md_rec_shower.IsLdfReconstructed():
        return None

    sd_info = extract_sd_info(event)
    s_event = event.GetSDEvent()

    event_id = event.GetEventId()
    yymmdd = event.GetYYMMDD()

    counter_vector = m_event.GetCounterVector()
    station_vector = s_event.GetStationVector()

    outdict = {
        "event_id": [],
        "yymmdd": [],
        "gps": [],
        "lge": [],
        "theta": [],
        "phi": [],
        "phi0": [],
        "dphi": [],
        "sd_shower_size": [],
        "sd_shower_size_err": [],
        "sd_shower_size_label": [],
        "sd_shower_size_att": [],
        "sd_shower_size_att_err": [],
        "is6T5": [],
        "counter_id": [],
        "module_id": [],
        "r": [],
        "k": [],
        "n_segments": [],
        "pcc": [],
        "active_area": [],
        "phi_sp": [],
        "nmu": [],
        "nmu_uncorrected": [],
        "sd_signal": [],
        "sd_signal_err": [],
        "r_station": [],
        "xc": [],
        "yc": []
    }

    for counter in counter_vector:
        counter_id = counter.GetId()
        if counter_id not in counters_to_analyze:
            continue

        if counter.IsRejected():
            continue

        sd_id = counter.GetSdPartnerId()

        # Using station = s_event.GetStation(sd_id) yields a null pointer, idk why.
        # So to access to the corresponding SD station, I need to do this
        # ugly loop over station_vector
        sd_signal = -1
        sd_signal_err = -1
        r_station = -1
        #print(f" counter_id = {counter_id}, sd_id = {sd_id}")
        for station in station_vector:
            if station.GetId() == sd_id:

                sd_signal = station.GetTotalSignal()
                sd_signal_err = station.GetTotalSignalError()
                r_station = station.GetSPDistance()
                break


        module_vector = get_module_list(counter)

        for mod in module_vector:
            if mod.IsRejected():
                continue

            mod_id = mod.GetId()
            ns = mod.GetSegmentation()
            area = mod.GetActiveArea()


            k = get_total_number_of_channels_on(mod)


            # From ADST
            nmu_uncorrected = mod.GetNumberOfEstimatedMuonsUncorrected()
            nmu = mod.GetNumberOfEstimatedMuons()

            nmu_adst_err_low = mod.GetNumberOfMuonsErrorLow()
            nmu_adst_err_high = mod.GetNumberOfMuonsErrorHigh()

            pcc = mod.GetCornerClippingProbability()


            r = mod.GetSPDistance()
            phi_sp = mod.GetSPAzimuth()

            # module orientation
            phi0 = ad.get_module_orientation(counter=counter_id,
                                        module=mod_id, kind="data",
                                        photodetector="sipm")
            dphi_raw = sd_info["phi"] - phi0
            sindphi = abs(np.sin(np.deg2rad(dphi_raw)))
            dphi = np.rad2deg(np.arcsin(sindphi))

            # Fill output
            outdict["event_id"].append(event_id)
            outdict["yymmdd"].append(yymmdd)
            outdict["gps"].append(sd_info["gps_sec"])
            outdict["lge"].append(sd_info["lgE"])
            outdict["theta"].append(sd_info["theta"])
            outdict["phi"].append(sd_info["phi"])
            outdict["phi0"].append(phi0)
            outdict["dphi"].append(dphi)
            outdict["sd_shower_size"].append(sd_info["shower_size"])
            outdict["sd_shower_size_err"].append(sd_info["shower_size_err"])
            outdict["sd_shower_size_label"].append(sd_info["shower_size_label"])
            outdict["sd_shower_size_att"].append(sd_info["shower_size_att"])
            outdict["sd_shower_size_att_err"].append(sd_info["shower_size_att_err"])
            outdict["is6T5"].append(sd_info["is6T5"])
            outdict["counter_id"].append(counter_id)
            outdict["module_id"].append(mod_id)
            outdict["r"].append(r)
            outdict["k"].append(k)
            outdict["n_segments"].append(ns)
            outdict["pcc"].append(pcc)
            outdict["active_area"].append(area)
            outdict["phi_sp"].append(phi_sp)
            outdict["nmu"].append(nmu)
            outdict["nmu_uncorrected"].append(nmu_uncorrected)
            outdict["sd_signal"].append(sd_signal)
            outdict["sd_signal_err"].append(sd_signal_err)
            outdict["r_station"].append(r_station)
            outdict["xc"].append(sd_info["x_core_UTM_CS"])
            outdict["yc"].append(sd_info["y_core_UTM_CS"])

    return outdict

def get_module_level_info(fname, counters_to_analyze={1815, 1771}, verbose=False):
    """
    Aggregate module-level information from one or more ADST files.

    The function opens the supplied ADST files, initializes the required ROOT
    event and detector-geometry objects, and iterates sequentially over all
    events. For each event with valid MD information, it calls
    ``get_event_module_level_info()`` to extract observables for the selected
    MD counters.

    The module-level results from all accepted events are concatenated into a
    single dictionary of lists. Each position in these lists corresponds to
    one accepted MD module, rather than to one event. Consequently,
    event-level quantities are repeated for every accepted module in the
    corresponding event.

    Parameters
    ----------
    fname : str, os.PathLike, or sequence of str or os.PathLike
        Path to a single ADST file, or a sequence containing the paths to
        multiple ADST files. The files are processed in the supplied order.

    counters_to_analyze : collections.abc.Collection of int, optional
        Identifiers of the MD counters to include in the analysis. By default,
        counters 1815 and 1771 are selected.

    verbose : bool, optional
        If ``True``, print a summary for each event and diagnostic messages
        when events without valid MD information are skipped. The default is
        ``False``. A progress message is printed every 1000 events
        independently of this option.

    Returns
    -------
    dict[str, list]
        Dictionary containing the module-level observables accumulated over
        all processed events. Each key corresponds to one observable and maps
        to a list with one entry per accepted MD module.

        The keys and their meanings are defined by
        ``get_event_module_level_info()``.

        An empty dictionary is returned if none of the input events produces
        valid module-level information.

    Raises
    ------
    ValueError
        If any of the specified input files does not exist.

    KeyError
        If the ``AUGEROFFLINEROOT`` environment variable is not defined.

    Notes
    -----
    Before processing the events, the function loads ``libRecEventKG.so``
    from the Offline installation specified by ``AUGEROFFLINEROOT``.

    Events for which ``extract_md_info()`` returns ``None`` are skipped.
    Additional counter- and module-level selections are applied by
    ``get_event_module_level_info()``.
    """

    if not os.path.exists(fname):
        raise ValueError(f"{fname} does not exists!")

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

    outdict = {}


    n = 0

    # Loop over events
    while rec_file.ReadNextEvent() == ROOT.RecEventFile.eSuccess:
        n += 1
        #if n == 10:
            #break
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

        if n % 1000 == 0:
            print(f" Processing event number {n}")


        module_info = get_event_relevant_module_level_info(event=event,
                                                 counters_to_analyze=counters_to_analyze)

        if module_info is None:
            continue

        # Defining keys the first time
        if not outdict:
            for key in module_info.keys():
                outdict[key] = []

        # Filling outdict
        for key in module_info.keys():
            outdict[key] += module_info[key]

    return outdict

adst_path = "/storage/gpfs_data/pauger/users/jdejesus/analysis/reco_impact_module_low_1s/adst_cuts/ADST_cuts_phase2_01-2025_09-2026.root"

counters_to_analyze = {1815, 1771, 643, 1816}

print(f"counters_to_analyze = {counters_to_analyze}")
print("")

module_level_info = get_module_level_info(adst_path, counters_to_analyze, verbose=False)

df = pd.DataFrame.from_dict(module_level_info)

outfilename = f"module_level_info_01-2025_09-2026_{counters_to_analyze}.csv"
df.to_csv(outfilename, index=False)

print(f"{outfilename} generated")
print(df.head())
