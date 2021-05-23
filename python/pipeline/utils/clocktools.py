import warnings
import numpy as np
import datajoint as dj
import matplotlib.pyplot as plt
from scipy import interpolate
from itertools import groupby
from pipeline import treadmill, fuse, shared, odor
from pipeline.exceptions import PipelineException

stimulus = dj.create_virtual_module("stimulus", "pipeline_stimulus")
pupil = dj.create_virtual_module("pipeline_eye", "pipeline_eye")


def find_idx_boundaries(indices, drop_single_idx=False):
    """
    Given a flatten list/array of indices, break list into a list of lists of indices incrementing by 1

        Example:
            >>>find_idx_boundaries([1,2,3,4,501,502,503,504])
               return value: [[1,2,3,4],[501,502,503,504]]

        Parameters:
            indices: Flattened list or numpy array of indices to break apart into sublists
            drop_single_idx: Boolean which sets if single indices not part of any sublist
                             should be dropped or raise an error upon detection.

        Returns:
            events: List of lists of indices that are incrementing by 1

    """

    events = []

    ## Basic idea: If you have a list [1,2,3,20,21,22], subtracting the index of that value from it
    ## will lead to assigning different numbers to different clusters of values incrementing by one.
    ## For instance [1-1, 2-2, 3-3, 20-4, 21-5, 22-6] = [0, 0, 0, 16, 16, 16]. Using groupby we
    ## split these values into group 1 (everything assigned 0) and group 2 (everything assigned 16).
    for k, g in groupby(enumerate(indices), lambda x: x[0] - x[1]):

        event = np.array([e[1] for e in g])

        if len(event) == 1:
            if not drop_single_idx:
                raise PipelineException(f"Disconnected index found: {event[0]}")
        else:
            events.append(event)

    return events


def find_time_boundaries(indices, times, drop_single_idx=False):
    """
    Given a flatten list/array of indices and timings, return the start & stop times for all idx clusters
    incrementing by 1. Start/stop times are taken to be non-NaN min/max times during the idx fragment.

        Example:
            >>>find_time_boundaries(indices=[1,2,3,501,502,503],times=[0.1,0.2,0.3,0.4,0.5,0.6])
               return value: [[0.1,0.3], [0.4,0.6]]

        Parameters:
            indices: Flattened list or numpy array of indices to break apart into incrementing fragments
            times: Flattened list or numpy array of times of recording. Must be the full list of times all
                   recordings took place on, not just the times which correspond to the list indices.
            drop_single_idx: Boolean which sets if single indices not part of any sublist
                             should be dropped or raise an error upon detection.

        Returns:
            events: List of lists of start & stop times for each incrementing by 1 idx fragments

    """

    idx_events = find_idx_boundaries(indices, drop_single_idx)

    time_boundaries = []
    for idx_event in idx_events:

        start = np.nanmin(times[idx_event])
        end = np.nanmax(times[idx_event])
        time_boundaries.append(np.array([start, end]))

    return time_boundaries


def fetch_timing_data(
    scan_key: dict,
    source_type: str,
    target_type: str,
    debug: bool = True,
):
    """
    Fetches timing data for source and target recordings. Adjusts both timings based on any calculable delays. Returns two
    arrays. Converts target recording times on target clock into target recording times on source clock if the two are different.

        Parameters:

                scan_key: A dictionary specifying a single scan and/or field. A single field must be defined if requesting
                          a source or target from ScanImage. If key specifies a single unit, unit delay will be added to
                          all timepoints recorded. Single units can be specified via unique mask_id + field or via unit_id.
                          If only field is specified, average field delay will be added.

                source_type: A string specifying what recording times to fetch for source_times. Both target and source times
                             will be returned on whatever clock is used for source_type. Fluorescence and deconvolution have
                             a dash followed by "behavior" or "stimulus" to refer to which clock you are using.
                             Supported options:
                                 'fluorescence-stimulus', 'deconvolution-stimulus', ,'fluorescence-behavior',
                                 'deconvolution-behavior', 'pupil', 'treadmill', 'respiration'

                target_type: A string specifying what recording times to fetch for target_times. Both target and source times
                             will be returned on whatever clock is used for source_type. Fluorescence and deconvolution have
                             a dash followed by "behavior" or "stimulus" to refer to which clock you are using.
                             Supported options:
                                 'fluorescence-stimulus', 'deconvolution-stimulus', ,'fluorescence-behavior',
                                 'deconvolution-behavior', 'pupil', 'treadmill', 'respiration'

                debug: Set function to print helpful debug text while running


        Returns:

                source_times: Numpy array of times for source recording on source clock

                target_times: Numpy array of times for target recording on source clock
    """
    
    ## Make settings strings lowercase
    source_type = source_type.lower()
    target_type = target_type.lower()
    
    ##
    ## Set pipe, error check scan_key, and fetch field offset
    ##

    ## Define the pipe (meso/reso) to use
    if len(fuse.MotionCorrection & scan_key) == 0:
        msg = f"scan_key {scan_key} not found in fuse.MotionCorrection."
        raise PipelineException(msg)
    pipe = (fuse.MotionCorrection & scan_key).module

    ## Make strings lowercase and process indices
    source_type = source_type.lower()
    target_type = target_type.lower()

    ## Set default values for later processing
    field_offset = 0
    slice_num = 1
    ms_delay = 0

    ## Determine if source or target type requires extra scan info
    scan_types = (
        "fluorescence-stimulus",
        "fluorescence-behavior",
        "deconvolution-stimulus",
        "deconvolution-behavior",
    )
    if source_type in scan_types or target_type in scan_types:

        ## Check scan_key defines a unique scan
        if len(pipe.ScanInfo & scan_key) != 1:
            msg = (
                f"scan_key {scan_key} does not define a unique scan. "
                f"Matching scans found: {len(fuse.MotionCorrection & scan_key)}"
            )
            raise PipelineException(msg)

        ## Check a single field is defined by scan_key
        if len(pipe.ScanInfo.Field & scan_key) != 1:
            msg = (
                f"scan_key {scan_key} must specify a single field when source or target type is set "
                f"to 'scan'. Matching fields found: {len(pipe.ScanInfo.Field & scan_key)}"
            )
            raise PipelineException(msg)

        ## Determine field offset to slice times later on and set ms_delay to field average
        scan_restriction = (pipe.ScanInfo & scan_key).fetch("KEY")
        all_z = np.unique(
            (pipe.ScanInfo.Field & scan_restriction).fetch("z", order_by="field ASC")
        )
        slice_num = len(all_z)
        field_z = (pipe.ScanInfo.Field & scan_key).fetch1("z")
        field_offset = np.where(all_z == field_z)[0][0]
        if debug:
            print(f"Field offset found as {field_offset} for depths 0-{len(all_z)}")

        field_delay_im = (pipe.ScanInfo.Field & scan_key).fetch1("delay_image")
        average_field_delay = np.mean(field_delay_im)
        ms_delay = average_field_delay
        if debug:
            print(
                f"Average field delay found to be {round(ms_delay,4)}ms. This will be used unless a unit is specified in the key."
            )

        ## If included, add unit offset
        if "unit_id" in scan_key or "mask_id" in scan_key:
            if len(pipe.ScanSet.Unit & scan_key) > 0:
                unit_key = (pipe.ScanSet.Unit & scan_key).fetch1()
                ms_delay = (pipe.ScanSet.UnitInfo & unit_key).fetch1("ms_delay")
                if debug:
                    print(
                        f"Unit found with delay of {round(ms_delay,4)}ms. Delay added to relevant times."
                    )
            else:
                if debug:
                    print(
                        f"Warning: ScanSet.Unit is not populated for the given key! Using field offset minimum instead."
                    )

    ##
    ## Fetch source and target sync data
    ##

    ## Define a lookup for data sources. Key values are in (data_table, column_name) tuples.
    data_source_lookup = {
        "fluorescence-stimulus": (stimulus.Sync, "frame_times"),
        "deconvolution-stimulus": (stimulus.Sync, "frame_times"),
        "fluorescence-behavior": (stimulus.BehaviorSync, "frame_times"),
        "deconvolution-behavior": (stimulus.BehaviorSync, "frame_times"),
        "treadmill": (treadmill.Treadmill, "treadmill_time"),
        "pupil": (pupil.Eye, "eye_time"),
        "respiration": (odor.Respiration * odor.MesoMatch, "times"),
    }

    ## Error check inputs
    if source_type not in data_source_lookup or target_type not in data_source_lookup:
        msg = (
            f"Source and target type combination '{source_type}' and '{target_type}' not supported. "
            f"Valid values are 'fluorescence-behavior', 'fluorescence-stimulus', 'deconvolution-behavior', "
            f"'deconvolution-stimulus', treadmill', 'respiration' or 'pupil'."
        )
        raise PipelineException(msg)

    ## Fetch source and target times using lookup dictionary
    source_table, source_column = data_source_lookup[source_type]
    source_times = (source_table & scan_key).fetch1(source_column).squeeze()

    target_table, target_column = data_source_lookup[target_type]
    target_times = (target_table & scan_key).fetch1(target_column).squeeze()

    ##
    ## Timing corrections
    ##

    ## Slice times if on ScanImage clock and add delay (scan_types defined near top)
    if source_type in scan_types:
        source_times = source_times[field_offset::slice_num] + ms_delay
    if target_type in scan_types:
        target_times = target_times[field_offset::slice_num] + ms_delay

    ##
    ## Interpolate into different clock if necessary
    ##

    clock_type_lookup = {
        "fluorescence-stimulus": "stimulus",
        "deconvolution-stimulus": "stimulus",
        "fluorescence-behavior": "behavior",
        "deconvolution-behavior": "behavior",
        "pupil": "behavior",
        "processed-pupil": "behavior",
        "treadmill": "behavior",
        "processed-treadmill": "behavior",
        "respiration": "odor",
    }

    sync_conversion_lookup = {
        "stimulus": stimulus.Sync,
        "behavior": stimulus.BehaviorSync,
        "odor": odor.OdorSync * odor.MesoMatch,
    }

    source_clock_type = clock_type_lookup[source_type]
    target_clock_type = clock_type_lookup[target_type]

    if source_clock_type != target_clock_type:

        interp_source_table = sync_conversion_lookup[source_clock_type]
        interp_target_table = sync_conversion_lookup[target_clock_type]

        interp_source = (interp_source_table & scan_key).fetch1("frame_times").squeeze()
        interp_target = (interp_target_table & scan_key).fetch1("frame_times").squeeze()

        target2source_interp = interpolate.interp1d(
            interp_target, interp_source, fill_value="extrapolate"
        )
        target_times = target2source_interp(target_times)

    return source_times, target_times


def interpolate_signal_data(
    scan_key: dict,
    source_type: str,
    target_type: str,
    source_times,
    target_times,
    debug: bool = True,
):
    """
    Interpolates target_type recording onto source_times. If target FPS is higher than source FPS, run lowpass hamming
    filter at source Hz over target_type recording before interpolating. Automatically slices ScanImage times and runs
    error checking for length mismatches.

        Parameters:

                scan_key: A dictionary specifying a single scan and/or field. A single field must be defined if requesting
                          a source or target from ScanImage. If key specifies a single unit, unit delay will be added to
                          all timepoints recorded. Single units can be specified via unique mask_id + field or via unit_id.
                          If only field is specified, average field delay will be added.

                source_type: A string specifying what indices you want to convert from. Fluorescence and deconvolution
                             have a dash followed by "behavior" or "stimulus" to refer to which clock you are using.
                             Supported options:
                                 'fluorescence-stimulus', 'deconvolution-stimulus', ,'fluorescence-behavior',
                                 'deconvolution-behavior', 'pupil', 'treadmill', 'respiration'

                target_type: A string specifying what indices to convert into. Fluorescence and deconvolution have a
                             dash followed by "behavior" or "stimulus" to refer to which clock you are using.
                             Supported options:
                                 'fluorescence-stimulus', 'deconvolution-stimulus', ,'fluorescence-behavior',
                                 'deconvolution-behavior', 'pupil', 'treadmill', 'respiration'

                source_times: Numpy array of times for source recording on source clock. Assumed to be corrected for
                              delays such as average field delay or specific unit delay.

                target_times: Numpy array of times for target recording on source clock. Assumed to be corrected for
                              delays such as average field delay or specific unit delay.

                debug: Set function to print helpful debug text while running


        Returns:

                interpolate_signal: Numpy array of target_type signal interpolated to recording times of source_type
    """

    ## Make settings strings lowercase
    source_type = source_type.lower()
    target_type = target_type.lower()

    ## Define the pipe (meso/reso) to use
    if len(fuse.MotionCorrection & scan_key) == 0:
        msg = f"scan_key {scan_key} not found in fuse.MotionCorrection."
        raise PipelineException(msg)
    pipe = (fuse.MotionCorrection & scan_key).module

    ## Run helpful error checking
    if source_type == "pupil":
        tracking_method_num = len(
            dj.U("tracking_method") & (pupil.FittedPupil & scan_key)
        )
        if tracking_method_num > 1:
            msg = (
                "More than one pupil tracking method found for entered scan. "
                "Specify tracking_method in scan key (tracking_method=2 for DeepLabCut)."
            )
            raise PipelineException(msg)

    ## Fetch required signal
    ## Note: Pupil requires .fetch() while other signals require .fetch1().
    ##       It is easier to make an if-elif-else structure than a lookup dictionary in this case.
    if target_type in ("fluorescence-stimulus", "fluorescence-behavior"):
        target_signal = (pipe.Fluorescence.Trace & scan_key).fetch1("trace")
    elif target_type in ("deconvolution-stimulus", "deconvolution-behavior"):
        unit_key = (pipe.ScanSet.Unit & scan_key).fetch1()
        target_signal = (pipe.Activity.Trace & unit_key).fetch1("trace")
    elif target_type == "pupil":
        target_signal = (pupil.FittedPupil.Circle & scan_key).fetch("radius")
    elif target_type == "treadmill":
        target_signal = (treadmill.Treadmill & scan_key).fetch1("treadmill_vel")
    elif target_type == "respiration":
        target_signal = ((odor.Respiration * odor.MesoMatch) & scan_key).fetch1("trace")
    else:
        msg = f"Error, target type {target_type} is not supported. Cannot fetch signal data."
        raise PipelineException(msg)

    ## Calculate FPS to determine if lowpass filtering is needed
    source_fps = 1 / np.nanmedian(np.diff(source_times))
    target_fps = 1 / np.nanmedian(np.diff(target_times))

    ## Fill NaNs to prevent interpolation errors, but store NaNs for later to add back in after interpolating
    source_replace_nans = None  # Use this as a switch to refill things later
    if sum(np.isnan(target_signal)) > 0:
        target_nan_indices = np.isnan(target_signal)
        time_nan_indices = np.isnan(target_times)
        target_replace_nans = np.logical_and(target_nan_indices, ~time_nan_indices)
        if sum(target_replace_nans) > 0:
            source_replace_nans = convert_clocks(
                scan_key,
                np.where(target_replace_nans)[0],
                "indices",
                target_type,
                "indices",
                source_type,
                debug=False,
            )
        nan_filler_func = (
            shared.FilterMethod & {"filter_method": "NaN Filler"}
        ).run_filter
        target_signal = nan_filler_func(target_signal)
        if debug:
            biggest_time_gap = np.nanmax(
                np.diff(target_times[np.where(~target_replace_nans)[0]])
            )
            msg = (
                f"Found NaNs in {sum(target_nan_indices)} locations, which corresponds to "
                f"{round(100*sum(target_nan_indices)/len(target_signal),2)}% of total signal. "
                f"Largest NaN gap found: {round(biggest_time_gap, 2)} seconds."
            )
            print(msg)

    ## Lowpass signal if needed
    if source_fps < target_fps:
        if debug:
            msg = (
                f"Target FPS of {round(target_fps,2)} is greater than source FPS {round(source_fps,2)}. "
                f"Hamming lowpass filtering target signal before interpolation"
            )
            print(msg)
        target_signal = shared.FilterMethod._lowpass_hamming(
            signal=target_signal, signal_freq=target_fps, lowpass_freq=source_fps
        )

    ## Timing and recording array lengths can differ slightly if recording was stopped mid-scan. Timings for
    ## the next X depths would be recorded, but fluorescence values would be dropped if all depths were not
    ## recorded. This would mean timings difference shouldn't be more than the number of depths of the scan.
    if len(target_times) < len(target_signal):
        msg = (
            f"More recording values than target time values exist! This should not be possible.\n"
            f"Target time length: {len(target_times)}\n"
            f"Target signal length: {len(target_signal)}"
        )
        raise PipelineException(msg)

    elif len(target_times) > len(target_signal):

        scan_res = pipe.ScanInfo.proj() & scan_key  ## To make sure we select all fields
        z_plane_num = len(dj.U("z") & (pipe.ScanInfo.Field & scan_res))
        if (len(target_times) - len(target_signal)) > z_plane_num:
            msg = (
                f"Extra timing values exceeds reasonable error bounds. "
                f"Error length of {len(target_times) - len(target_signal)} with only {z_plane_num} z-planes."
            )
            raise PipelineException(msg)

        else:

            shorter_length = np.min((len(target_times), len(target_signal)))
            source_times = target_times[:shorter_length]
            source_signal = target_signal[:shorter_length]
            if debug:
                length_diff = np.abs(len(target_times) - len(target_signal))
                msg = (
                    f"Target times and target signal show length mismatch within acceptable error."
                    f"Difference of {length_diff} within acceptable bounds of {z_plane_num} z-planes."
                )
                print(msg)

    ## Interpolating target signal into source timings
    signal_interp = interpolate.interp1d(
        target_times, target_signal, bounds_error=False
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        interpolated_signal = signal_interp(source_times)
    if source_replace_nans is not None:
        for source_nan_idx in source_replace_nans:
            interpolated_signal[source_nan_idx] = np.nan

    return interpolated_signal


def convert_clocks(
    scan_key: dict,
    input_list,
    source_format: str,
    source_type: str,
    target_format: str,
    target_type: str,
    drop_single_idx: bool = True,
    debug: bool = True,
):
    """
    Converts indices or times of interest on source clock to indices, times, or signals on target clock. Can convert
    a collection of event-triggered fragments of indices/times or a single flat list. Can also be used as an automated
    times/signal fetching function by setting input_list to None and source_type equal to target_type.

        Parameters:

                scan_key: A dictionary specifying a single scan and/or field. A single field must be defined if
                          requesting a source or target from ScanImage. If key specifies a single unit, unit delay
                          will be added to all timepoints recorded. Single units can be specified via unique
                          mask_id + field or via unit_id. If only field is specified, average field delay will be
                          added.


                input_list: List/array/None. Depending on the source_format, there are many possible structures:

                             - source_type='indices'
                                 1) List/array of indices to convert or a boolean array with True values at indices
                                    of interest. Indices can be discontinuous fragments (something like 20 indices
                                    around a several spikes) and the function will return of list of lists, each
                                    containings the corresponding target_idx fragments.
                                 2) None. Set input_list to None to use all indices where source time is not NaN.

                             - source_type='times'
                                 1) List of lists containing [start,stop] boundaries for times of note on source clock.
                                    Start/Stop times are included (>=,<=). These boundaries are converted to all indices
                                    equal to or between recording times on target recording.
                                 2) None. Set input_list to None to use all times where source time is not NaN.


                source_format: A string specifying what the input_list variable represents and what structure to expect.
                               See details for input_list variable to learn more.
                               Supported options:
                                   'indices', 'times'


                source_type: A string specifying what indices/times you want to convert from. Fluorescence and
                             deconvolution have a dash followed by "behavior" or "stimulus" to refer to which clock
                             you are using.
                             Supported options:
                                 'fluorescence-stimulus', 'deconvolution-stimulus', ,'fluorescence-behavior',
                                 'deconvolution-behavior', 'pupil', 'treadmill', 'respiration'


                target_format: A string specifying what values to return. "Times" has a dash followed by "source"
                               or "target" to specify if the returning times should be on the source clock or on
                               the target clock. If set to "signal", returns interpolated target signal on the
                               corresponding source_type recording times specified by input_list.
                               Supported options:
                                   'indices', 'times-source', 'times-target', 'signal'


                target_type: A string specifying what indices to convert into. Fluorescence and deconvolution have a
                             dash followed by "behavior" or "stimulus" to refer to which clock you are using.
                             Supported options:
                                 'fluorescence-stimulus', 'deconvolution-stimulus', ,'fluorescence-behavior',
                                 'deconvolution-behavior', 'pupil', 'treadmill', 'respiration'


                drop_single_idx: Boolean with the following behavior
                                     - True: Drop any signal fragments which would create a signal of length 1.
                                     - False: Raise an error and stop if any list of indices leads to a signal
                                              of length 1.
                                    ex. Source IDX [1,2,...,300] on a 500HZ recording will only correspond to
                                        target IDX [1] if target is recorded at 1Hz.


                debug: Set function to print helpful debug text while running


        Returns:

                requested_array: Numpy array of corresponding indices, times, or interpolated target signal. If
                multiple continuous fragments or time boundaries are in input_list, return value is a list of arrays.


        Warnings:

                * NaN refilling for signal interpolation will only refill values if NaNs stretch for multiple indices
                  on target clock

                * Recording points where the time value is NaN are dropped from analysis/processing


        Examples:

                Fetch fluorescence signal for one unit:

                    >>>key = dict(animal_id=17797, session=4, scan_idx=7, field=1, segmentation_method=6, mask_id=1, tracking_method=2)
                    >>>settings = dict(scan_key=key, input_list=None, source_format='indices', source_type='fluorescence-behavior',
                                       target_format='signal', target_type='fluorescence-behavior', drop_single_idx=True, debug=False)
                    >>>fluorescence_signal = convert_clocks(settings)


                Fetch recording times (on behavior clock) for one unit:

                    >>>key = dict(animal_id=17797, session=4, scan_idx=7, field=1, segmentation_method=6, mask_id=1, tracking_method=2)
                    >>>settings = dict(scan_key=key, input_list=None, source_format='indices', source_type='fluorescence-behavior',
                                       target_format='times-source', target_type='fluorescence-behavior', drop_single_idx=True, debug=False)
                    >>>fluorescence_times = convert_clocks(settings)


                Interpolate entire treadmill trace to fluorescence recording times:

                    >>>key = dict(animal_id=17797, session=4, scan_idx=7, field=1, segmentation_method=6, mask_id=1, tracking_method=2)
                    >>>settings = dict(scan_key=key, input_list=None, source_format='indices', source_type='fluorescence-behavior',
                                       target_format='signal', target_type='treadmill', drop_single_idx=True, debug=False)
                    >>>interpolated_treadmill = convert_clocks(settings)


                Convert discontinuous pupil IDX fragments to treadmill times (on behavior clock):

                    >>>key = dict(animal_id=17797, session=4, scan_idx=7, field=1, segmentation_method=6, mask_id=1, tracking_method=2)
                    >>>input_indices = np.concatenate(((np.arange(1000)), np.arange(1005, 2000)))
                    >>>settings = dict(scan_key=key, input_list=input_indices, source_format='indices', source_type='pupil',
                                       target_format='times-source', target_type='treadmill', drop_single_idx=True, debug=False)
                    >>>treadmill_time_fragments = convert_clocks(settings)


                Convert fluorescence time boundaries on behavior clock to fluorescence times on stimulus clock:

                    >>>key = dict(animal_id=17797, session=4, scan_idx=7, field=1, segmentation_method=6, mask_id=1, tracking_method=2)
                    >>>time_boundaries = [[400, 500], [501, 601]]
                    >>>settings = dict(scan_key=key, input_list=time_boundaries, source_format='times', source_type='fluorescence-behavior',
                                       target_format='times-target', target_type='fluorescence-stimulus', drop_single_idx=True, debug=False)
                    >>>fluorescence_stimulus_times_in_bounds = convert_clocks(settings)
    """

    ##
    ## Make settings strings lowercase
    ##

    source_format = source_format.lower()
    source_type = source_type.lower()
    target_format = target_format.lower()
    target_type = target_type.lower()

    ##
    ## Fetch source and target times, along with converting between Stimulus or Behavior clock if needed
    ##

    source_times_source_clock, target_times_source_clock = fetch_timing_data(
        scan_key, source_type, target_type, debug
    )
    target_times_target_clock, source_times_target_clock = fetch_timing_data(
        scan_key, target_type, source_type, debug
    )

    ##
    ## Convert indices to a list of numbers if argument equals None or a Boolean mask
    ##

    if source_format == "indices":
        if input_list is None:
            input_list = np.arange(len(source_times_source_clock))
        elif type(input_list[0]) == bool:
            input_list = np.where(input_list)[0]
        elif type(input_list[0]) == list or type(input_list[0]) == np.ndarray:
            input_list = [
                item for sublist in input_list for item in sublist
            ]  ## Flatten array if list of lists
        else:
            ## Check for duplicates if manually entered
            if len(np.unique(input_list)) != len(input_list):
                msg = (
                    f"Duplicate entries found for provided indice array! "
                    f"Try to fix the error or use np.unique() on indices array."
                )
                raise PipelineException(msg)

    ## Convert behavior to indices to make None input work smoothly
    if "times" in source_format and input_list is None:
        input_list = np.arange(len(source_times_source_clock))
        source_format = "indices"

    ##
    ## Convert source indices to time boundaries, then convert time boundaries into target indices
    ##

    ## Convert indices into start/end times for each continuous fragment (incrementing by 1)
    if source_format == "indices":
        time_boundaries = find_time_boundaries(
            input_list, source_times_source_clock, drop_single_idx
        )
    elif "times" in source_format:
        time_boundaries = input_list
    else:
        msg = (
            f"Source format {source_format} not supported. "
            f"Valid options are 'indices' and 'times'."
        )
        raise PipelineException(msg)

    target_indices = []
    single_idx_count = 0

    ## Loop through start & end times and create list of indices corresponding to that block of time
    with np.errstate(invalid="ignore"):
        for [start, end] in time_boundaries:
            target_idx = np.where(
                np.logical_and(
                    target_times_source_clock >= start, target_times_source_clock <= end
                )
            )[0]
            if len(target_idx) < 2:
                if drop_single_idx:
                    single_idx_count += 1
                else:
                    msg = (
                        f"Event of length {len(target_idx)} found. "
                        f"Set drop_single_idx to True to suppress these errors."
                    )
                    raise PipelineException(msg)
            else:
                target_indices.append(target_idx)

    if debug:
        print(f"Indices converted. {single_idx_count} events of length 0 or 1 dropped.")

    ##
    ## Interpolate related signal if requested, else just return the target_indices found.
    ##

    if target_format == "signal":

        ## Define source_indices if they're not already defined
        if source_format == "indices":
            source_indices = find_idx_boundaries(input_list, drop_single_idx)
        elif "times" in source_format:
            source_indices = convert_clocks(
                scan_key,
                input_list,
                source_format,
                source_type,
                "indices",
                target_type,
                drop_single_idx,
                False,
            )
        else:
            msg = (
                f"Source format {source_format} not supported. "
                f"Valid options are 'indices' and 'times'."
            )
            raise PipelineException(msg)

        ## Create full interpolated signal
        interpolated_signal = interpolate_signal_data(
            scan_key,
            source_type,
            target_type,
            source_times_source_clock,
            target_times_source_clock,
            debug=debug,
        )

        ## Split indices given into fragments based on which ones are continuous (incrementing by 1)
        target_signal_fragments = []
        for idx_fragment in source_indices:
            idx_fragment_mask = ~np.isnan(source_times_source_clock[idx_fragment])
            masked_idx_fragment = idx_fragment[idx_fragment_mask]
            target_signal_fragments.append(interpolated_signal[masked_idx_fragment])

        ## If full signal is converted, remove wrapping list
        if len(target_signal_fragments) == 1:
            target_signal_fragments = target_signal_fragments[0]

        converted_values = target_signal_fragments

    elif "times" in target_format:

        ## Set type of times to use
        if target_format == "times-source":
            target_times = target_times_source_clock
        elif target_format == "times-target":
            target_times = target_times_target_clock
        else:
            msg = f"'Times' target format must be 'times-source' or 'times-target'. Value was {target_format}."
            raise PipelineException(msg)

        ## Convert indices to times and return
        source_idx_to_target_times = []

        for target_idx_list in target_indices:
            source_idx_to_target_times.append(target_times[target_idx_list])

        if len(source_idx_to_target_times) == 1:
            source_idx_to_target_times = source_idx_to_target_times[0]

        converted_values = source_idx_to_target_times

    elif target_format == "indices":

        if len(target_indices) == 1:
            target_indices = target_indices[0]

        converted_values = target_indices

    else:

        msg = (
            f"Target format {target_format} is not supported. "
            f"Valid options are 'indices', 'times-source', 'times-target', 'signal'."
        )
        raise PipelineException(msg)

    return converted_values
