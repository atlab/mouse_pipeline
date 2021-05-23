import h5py
import os

import datajoint as dj
import numpy as np

from pipeline import mice
from pipeline import meso
from pipeline.utils import h5
from commons import lab
from .exceptions import PipelineException


#dj.config['external-odor'] = {'protocol': 'file',
#                              'location': '/mnt/dj-stor01/pipeline-externals'}

schema = dj.schema('pipeline_odor', locals(), create_tables=False)


@schema
class Odorant(dj.Lookup):
    definition = """ # Odorants used in solutions.
    odorant                     : varchar(32)          # name of odorant
    ---
    molar_mass                  : float                # odorant molar mass (g/mole)
    density                     : float                # odorant density (g/mL)
    vapor_pressure              : float                # vapor pressure at 25C (mmHg)
    odorant_description = ''    : varchar(2048)
    """
    contents = [
        ['Empty', '0', '0', '0', '']
    ]


@schema
class OdorSolution(dj.Manual):
    definition = """ # Solutions made for olfactory experiments.
    -> Odorant
    concentration               : decimal(2,2)         # Dilution fraction
    solution_date               : date                 # date solution was created (YYYY-MM-DD)
    ---
    solution_notes = ''         : varchar(2048)
    """


@schema
class OdorSession(dj.Manual):
    definition = """ # Single session of olfactory experiment.
    -> mice.Mice
    odor_session                : smallint unsigned    # session index for the mouse
    ---
    odor_path                   : varchar(2048)        # folder where h5 files for olfaction are stored
    odor_session_date           : date                 # date session was recorded (YYYY-MM-DD)
    time_loaded = null          : time
    """


@schema
class OdorConfig(dj.Manual):
    definition = """ # Configuration of solutions for each olfactory experiment. One entry for each channel.
    -> OdorSession
    channel                     : tinyint unsigned     # channel number of assigned solution
    ---
    -> OdorSolution
    """


@schema
class OdorRecording(dj.Manual):
    definition = """ # Single scan/recording within an olfactory session.
    -> OdorSession
    recording_idx               : smallint unsigned    # recording index for session
    ---
    filename                    : varchar(255)
    """


@schema
class OdorTrials(dj.Imported):
    definition = """ # Single presentation of olfactory stimulus. One entry for each ON channel.
    -> OdorRecording
    trial_idx                   : smallint unsigned    # trial index for recording
    channel                     : tinyint unsigned     # channel number used
    ---
    trial_start_time            : float                # start of trial on olfactory clock (seconds)
    trial_end_time              : float                # end of trial on olfactory clock (seconds)
    """

    def convert_valves(num):
        """ Converts decimal number encoding valve states into boolean array
        :param num: int. Decimal number encoding valve state in binary
        :returns: numpy array of booleans. Array encodes valve state (True=OPEN) starting with the smallest valve number
            ex. decimal(10) -> [False, True, False, True] -> 2nd and 4th valve open.
        """

        # Format integer into binary, then convert each 0/1 into boolean
        # Finally, reverse list to make first index encode first valve
        valve_states = [bool(int(d)) for d in format(int(num), 'b')][::-1]
        return np.array(valve_states)

    def make(self, key):

        print(f'Populating trials for {key}')

        # Get olfactory h5 path and filename
        olfactory_path = (OdorSession & key).fetch1('odor_path')
        local_path = lab.Paths().get_local_path(olfactory_path)
        filename_base = (OdorRecording & key).fetch1('filename')
        digital_filename = os.path.join(local_path, filename_base + '_D_%d.h5')

        # Load olfactory data
        digital_data = h5.read_digital_olfaction_file(digital_filename)

        # Check valve data ends with all valves closed
        if digital_data['valves'][-1] != 0:
            msg = f'Error: Final valve state is open! Ending time cannot be calculated for {key}.'
            raise PipelineException(msg)

        valve_open_idx = np.where(digital_data['valves'] > 0)[0]
        trial_valve_states = digital_data['valves'][valve_open_idx]
        trial_start_times = h5.ts2sec(digital_data['ts'][valve_open_idx])

        # Shift start indices by one to get end indices
        trial_end_times = h5.ts2sec(digital_data['ts'][valve_open_idx + 1])

        # All keys are appended to a list and inserted at the end to prevent errors from halting mid-calculation
        all_trial_keys = []

        # Find all trials and insert a key for each channel open during each trial
        for trial_num, (state, start, stop) in enumerate(zip(trial_valve_states, trial_start_times, trial_end_times)):

            valve_array = OdorTrials.convert_valves(state)

            for valve_num in np.where(valve_array)[0]:  # Valve array is already a boolean, look for all true values

                # We start counting valves at 1, not 0 like python indices
                valve_num = valve_num + 1
                trial_key = [key['animal_id'], key['odor_session'], key['recording_idx'],
                             trial_num, valve_num, start, stop]
                all_trial_keys.append(trial_key)

        self.insert(all_trial_keys)

        print(f'{valve_open_idx.shape[0]} odor trials found and inserted for {key}.\n')


@schema
class OdorSync(dj.Imported):
    definition = """ # ScanImage frame times on Olfactory Master Clock.
    -> OdorRecording
    ---
    signal_start_time           : float                # start of analog signal recording on olfactory clock (seconds)
    signal_duration             : float                # duration of analog signal (seconds)
    frame_times                 : longblob             # start of each scanimage frame on olfactory clock (seconds)
    sync_ts=CURRENT_TIMESTAMP   : timestamp
    """

    def make(self, key):

        print(f'Populating Sync for {key}')

        # Get olfactory h5 path and filename
        olfactory_path = (OdorSession & key).fetch1('odor_path')
        local_path = lab.Paths().get_local_path(olfactory_path)
        filename_base = (OdorRecording & key).fetch1('filename')
        analog_filename = os.path.join(local_path, filename_base + '_%d.h5')

        # Load olfactory data
        analog_data = h5.read_analog_olfaction_file(analog_filename)

        scan_times = h5.ts2sec(analog_data['ts'], is_packeted=True)
        binarized_signal = analog_data['scanImage'] > 2.7  # TTL voltage low/high threshold
        rising_edges = np.where(np.diff(binarized_signal.astype(int)) > 0)[0]
        frame_times = scan_times[rising_edges]

        # Correct NaN gaps in timestamps (mistimed or dropped packets during recording)
        if np.any(np.isnan(frame_times)):
            # Raise exception if first or last frame pulse was recorded in mistimed packet
            if np.isnan(frame_times[0]) or np.isnan(frame_times[-1]):
                msg = ('First or last frame happened during misstamped packets. Pulses '
                       'could have been missed: start/end of scanning is unknown.')
                raise PipelineException(msg)

            # Fill each gap of nan values with correct number of timepoints
            frame_period = np.nanmedian(np.diff(frame_times))  # approx
            nan_limits = np.where(np.diff(np.isnan(frame_times)))[0]
            nan_limits[1::2] += 1  # limits are indices of the last valid point before the nan gap and first after it
            correct_fts = []
            for i, (start, stop) in enumerate(zip(nan_limits[::2], nan_limits[1::2])):
                correct_fts.extend(frame_times[0 if i == 0 else nan_limits[2 * i - 1]: start + 1])
                num_missing_points = int(round((frame_times[stop] - frame_times[start]) /
                                               frame_period - 1))
                correct_fts.extend(np.linspace(frame_times[start], frame_times[stop],
                                               num_missing_points + 2)[1:-1])
            correct_fts.extend(frame_times[nan_limits[-1]:])
            frame_times = np.array(correct_fts)

        # Check that frame times occur at the same period
        frame_intervals = np.diff(frame_times)
        frame_period = np.median(frame_intervals)
        if np.any(abs(frame_intervals - frame_period) > 0.15 * frame_period):
            print('Frame time period is irregular, attempting to guess and extract correct times.')
            
            # Attempt to fix by finding the largest fragment of times without an outlier and seeing if it matches requested frame number
            n_frames_r = (meso.ScanInfo & (MesoMatch & key)).fetch1('nframes_requested')
            error_interval_idx = np.where((frame_intervals - frame_period) > frame_period*0.15)[0]
            bounds_idx = np.insert(error_interval_idx, [0,len(error_interval_idx)], [0,len(frame_intervals)])
            longest_fragment = np.max(np.diff(bounds_idx))
            
            # Guess correct start and end time
            s_idx = np.argmax(np.diff(bounds_idx))
            start = bounds_idx[s_idx]
            end = bounds_idx[s_idx+1]
            
            # Depending on how 
            if end - start == n_frames_r-1:
                print(f'Longest fragment is {end-start} frames long, but there are {n_frames_r} frames requested. Adding 1 to the end idx.')
                end += 1
            elif end-start == n_frames_r:
                print(f'Longest fragment length equals nFrames. Shifting start & end by one forward to avoid including error at start.')
                start += 1
                end += 1
            else:
                raise PipelineException(f'Unsupported length mismatch of longest fragment length {len(start-end)} vs nFrames {n_frames_rrames_r}.')
            
            # Rerun error checking with new frame_times fragment
            frame_times = frame_times[start:end]
            frame_intervals = np.diff(frame_times)
            frame_period = np.median(frame_intervals)
            
            if np.any(abs(frame_intervals - frame_period) > 0.15 * frame_period) or len(frame_times) != n_frames_r:
                raise PipelineException('Frame time period is irregular and cannot be corrected.')

        self.insert1({**key, 'signal_start_time': frame_times[0],
                      'signal_duration': frame_times[-1] - frame_times[0],
                      'frame_times': frame_times})

        print(f'ScanImage sync added for animal {key["animal_id"]}, '
              f'olfactory session {key["odor_session"]}, '
              f'recording {key["recording_idx"]}\n')


@schema
class Respiration(dj.Imported):
    definition = """ # Analog recording of mouse respiration
    -> OdorRecording
    ---
    trace                       : longblob             # mouse respiration (arbitrary units)
    times                       : longblob             # trace times on olfactory clock (seconds)
    """

    def make(self, key):

        print(f'Populating Respiration for {key}')

        # Get olfactory h5 path and filename
        olfactory_path = (OdorSession & key).fetch1('odor_path')
        local_path = lab.Paths().get_local_path(olfactory_path)
        filename_base = (OdorRecording & key).fetch1('filename')
        analog_filename = os.path.join(local_path, filename_base + '_%d.h5')

        # Load olfactory data
        analog_data = h5.read_analog_olfaction_file(analog_filename)
        breath_times = h5.ts2sec(analog_data['ts'], is_packeted=True)
        breath_trace = analog_data['breath']

        # Correct NaN gaps in timestamps (mistimed or dropped packets during recording)
        if np.any(np.isnan(breath_times)):
            # Raise exception if first or last frame pulse was recorded in mistimed packet
            if np.isnan(breath_times[0]) or np.isnan(breath_times[-1]):
                msg = ('First or last breath happened during misstamped packets. Pulses '
                       'could have been missed: start/end of collection is unknown.')
                raise PipelineException(msg)

            # Linear interpolate between nans
            nans_idx = np.where(np.isnan(breath_times))[0]
            non_nans_idx = np.where(~np.isnan(breath_times))[0]
            breath_times[nans_idx] = np.interp(nans_idx, non_nans_idx, breath_times[non_nans_idx])
            print(f'Largest NaN gap found: {np.max(np.abs(np.diff(breath_times[non_nans_idx])))} seconds')

        # Check that frame times occur at the same period
        breath_intervals = np.diff(breath_times)
        breath_period = np.median(breath_intervals)
        if np.any(abs(breath_intervals - breath_period) > 0.15 * breath_period):
            raise PipelineException('Breath time period is irregular')

        # Error check tracing and timing match
        if breath_trace.shape[0] != breath_times.shape[0]:
            raise PipelineException('Breath timing and trace mismatch!')

        breath_key = {**key, 'trace': breath_trace, 'times': breath_times}

        self.insert1(breath_key)
        print(f'Respiration data for {key} successfully inserted.\n')


@schema
class MesoMatch(dj.Manual):
    definition = """ # Match between Odor Recording and Scan Session
    -> OdorRecording
    ---
    -> meso.ScanInfo
    """
