import datajoint as dj
from datajoint.jobs import key_hash
import numpy as np
import matplotlib.pyplot as plt
from commons import lab
from pipeline import shared
import os

from . import experiment, notify
from .utils import h5, clocktools
from .exceptions import PipelineException


schema = dj.schema('pipeline_treadmill', locals())


@schema
class Treadmill(dj.Computed):
    definition = """ # treadmill velocity synchronized to behavior clock

    -> experiment.Scan
    ---
    treadmill_raw                       :longblob       # raw treadmill counts
    treadmill_time                      :longblob       # (secs) velocity timestamps in behavior clock
    treadmill_vel                       :longblob       # (cm/sec) wheel velocity
    treadmill_ts=CURRENT_TIMESTAMP      :timestamp
    """
    @property
    def key_source(self):
        return experiment.Scan() & experiment.Scan.BehaviorFile().proj()

    def _make_tuples(self, key):
        # Get behavior filename
        behavior_path = (experiment.Session() & key).fetch1('behavior_path')
        local_path = lab.Paths().get_local_path(behavior_path)
        filename = (experiment.Scan.BehaviorFile() & key).fetch1('filename')
        full_filename = os.path.join(local_path, filename)

        # Read file
        data = h5.read_behavior_file(full_filename)

        # Get counter timestamps and convert to seconds
        timestamps_in_secs = h5.ts2sec(data['wheel'][1])
        ts = h5.ts2sec(data['ts'], is_packeted=True)
        # edge case when ts and wheel ts start in different sides of the master clock max value 2 **32
        if abs(ts[0] - timestamps_in_secs[0]) > 2 ** 31:
            timestamps_in_secs += (2 ** 32 if ts[0] > timestamps_in_secs[0] else -2 ** 32)

        # Read wheel position counter and fix wrap around at 2 ** 32
        wheel_position = data['wheel'][0]
        wheel_diffs = np.diff(wheel_position)
        for wrap_idx in np.where(abs(wheel_diffs) > 2 ** 31)[0]:
            wheel_position[wrap_idx + 1:] += (2 ** 32 if wheel_diffs[wrap_idx] < 0 else -2 ** 32)
        wheel_position -= wheel_position[0] # start counts at zero

        # Compute wheel velocity
        num_samples = int(round((timestamps_in_secs[-1] - timestamps_in_secs[0]) * 10)) # every 100 msecs
        sample_times = np.linspace(timestamps_in_secs[0], timestamps_in_secs[-1], num_samples)
        sample_position = np.interp(sample_times, timestamps_in_secs, wheel_position)
        counter_velocity = np.gradient(sample_position) * 10 # counts / sec

        # Transform velocity from counts/sec to cm/sec
        wheel_specs = experiment.TreadmillSpecs() * experiment.Session() & key
        diameter, counts_per_rev = wheel_specs.fetch1('diameter', 'counts_per_revolution')
        wheel_perimeter = np.pi * diameter # 1 rev = xx cms
        velocity = (counter_velocity / counts_per_rev) * wheel_perimeter # cm /sec

        # Resample at initial timestamps
        velocity = np.interp(timestamps_in_secs, sample_times, velocity)

        # Fill with NaNs for out-of-range data or mistimed packets
        velocity[timestamps_in_secs < ts[0]] = float('nan')
        velocity[timestamps_in_secs > ts[-1]] = float('nan')
        nan_limits = np.where(np.diff([0, *np.isnan(ts), 0]))[0]
        for start, stop in zip(nan_limits[::2], nan_limits[1::2]):
            lower_ts = float('-inf') if start == 0 else ts[start - 1]
            upper_ts = float('inf') if stop == len(ts) else ts[stop]
            velocity[np.logical_and(timestamps_in_secs > lower_ts,
                                    timestamps_in_secs < upper_ts)] = float('nan')
        timestamps_in_secs[np.isnan(velocity)] = float('nan')

        # Insert
        self.insert1({**key, 'treadmill_time': timestamps_in_secs,
                      'treadmill_raw': data['wheel'][0], 'treadmill_vel': velocity})
        self.notify(key)

    @notify.ignore_exceptions
    def notify(self, key):
        time, velocity = (self & key).fetch1('treadmill_time', 'treadmill_vel')
        fig = plt.figure()
        plt.plot(time, velocity)
        plt.ylabel('Treadmill velocity (cm/sec)')
        plt.xlabel('Seconds')
        img_filename = '/tmp/' + key_hash(key) + '.png'
        fig.savefig(img_filename)
        plt.close(fig)

        msg = 'treadmill velocity for {animal_id}-{session}-{scan_idx}'.format(**key)
        slack_user = notify.SlackUser() & (experiment.Session() & key)
        slack_user.notify(file=img_filename, file_title=msg)
        
        
@schema
class RunningMethod(dj.Lookup):
    definition = """ # Method to extract periods of running
    running_method_id             : tinyint unsigned    # index for running classification method
    ---
    run_method_description        : varchar(256)        # description of running classification method
    """
    contents = [[1, 'Threshold based method requiring run times > 1sec and concatenating run times < 3sec apart'],
                [2, 'Threshold based method with 1sec padding. Run times must be > 1sec long and are concatenated if < 3sec apart.'],
                [3, 'Threshold based method with 10sec padding. Run times must be > 1sec long and are concatenated if < 21sec apart.'],]
 
    
@schema
class Running(dj.Computed):
    definition = """ # Scan level statistics about running periods
    -> Treadmill
    -> shared.FilterMethod
    -> RunningMethod
    ---
    processed_treadmill           : longblob            # filtered version of treadmill speed
    total_run_duration            : float               # (sec) total amount of time spent running
    percent_running               : float               # (percent) percent of time spent running
    mean_run_velocity             : float               # (cm/sec) mean velocity for entire treadmill trace
    mean_run_speed                : float               # (cm/sec) mean speed for entire treadmill trace
    max_run_speed                 : float               # (cm/sec) maximum speed during recording
    num_run_periods               : smallint unsigned   # number of running periods
    """
   
    class Period(dj.Part):
        definition = """ # Single running event detected 
        -> Running
        run_idx                   : smallint unsigned   # running event id
        ---
        run_onset                 : float               # (sec) onset time on behavior clock
        run_offset                : float               # (sec) offset time on behavior clock
        run_duration              : float               # (sec) duration of running period
        mean_velocity             : float               # (cm/sec) 
        mean_speed                : float               # (cm/sec) 
        max_speed                 : float               # (cm/sec)
        """

    def make(self, key):
        
        print(f'Populating Running for {key}')
        
        ## Get treadmill info
        treadmill_times, treadmill_velocity = (Treadmill & key).fetch1('treadmill_time', 'treadmill_vel')
        treadmill_speed = np.abs(treadmill_velocity)
        treadmill_fps = 1/np.nanmedian(np.diff(treadmill_times))
        
        ## Filter treadmill velocity and speed
        ## NOTE: Treadmill speed is NOT absolute value of filtered velocity. Filtering is applied after np.abs.
        my_filter = (shared.FilterMethod & key).run_filter_with_renan
        filt_treadmill_velocity = my_filter(treadmill_velocity, treadmill_fps)
        filt_treadmill_speed = my_filter(treadmill_speed, treadmill_fps)
        
        ## Run running detection method
        if key['running_method_id'] == 1:

            ## CONSTANTS
            run_speed_threshold = 1 ## cm/sec
            maximum_time_gap = 3    ## sec
            minimum_run_length = 1  ## sec
            padding_time = 0 ## sec
            
            running_indices = np.where(filt_treadmill_speed > run_speed_threshold)[0]
            combined_running_periods = self._combine_running_periods(running_indices, treadmill_times, maximum_time_gap)
            finalized_running_periods = self._remove_short_running_periods(combined_running_periods, treadmill_times, minimum_run_length)

            
        elif key['running_method_id'] == 2:
        
            ## CONSTANTS
            run_speed_threshold = 0.5 ## cm/sec
            maximum_time_gap = 3    ## sec
            minimum_run_length = 1  ## sec
            padding_time = 1 ## sec (MUST BE LESS THAN HALF MAXIMUM TIME GAP)

            running_indices = np.where(filt_treadmill_speed > run_speed_threshold)[0]
            combined_running_periods = self._combine_running_periods(running_indices, treadmill_times, maximum_time_gap)
            finalized_running_periods = self._remove_short_running_periods(combined_running_periods, treadmill_times, minimum_run_length)
        
        elif key['running_method_id'] == 3:
        
            ## CONSTANTS
            run_speed_threshold = 0.5 ## cm/sec
            maximum_time_gap = 21    ## sec
            minimum_run_length = 1  ## sec
            padding_time = 10 ## sec (MUST BE LESS THAN HALF MAXIMUM TIME GAP)

            running_indices = np.where(filt_treadmill_speed > run_speed_threshold)[0]
            combined_running_periods = self._combine_running_periods(running_indices, treadmill_times, maximum_time_gap)
            finalized_running_periods = self._remove_short_running_periods(combined_running_periods, treadmill_times, minimum_run_length)

        else:
            
            msg = f"Running method id {key['running_method_id']} not supported."
            raise PipelineException(msg)

        ## Calculate statistics
        mean_run_velocity = np.nanmean(filt_treadmill_speed)
        mean_run_speed = np.nanmean(treadmill_speed)
        max_run_speed = np.nanmax(treadmill_speed)
        num_run_periods = len(finalized_running_periods)
        
        ## Loop through running periods
        first_tread_time = np.nanmin(treadmill_times)
        last_tread_time = np.nanmax(treadmill_times)
        running_period_keys = []
        total_run_duration = 0
        for n,running_period in enumerate(finalized_running_periods):
            run_period_key = key.copy()
            run_period_key['run_idx'] = n+1
            run_period_key['run_onset'] = np.max((treadmill_times[running_period[0]] - padding_time, first_tread_time))
            run_period_key['run_offset'] = np.min((treadmill_times[running_period[-1]] + padding_time, last_tread_time))
            run_period_key['run_duration'] = run_period_key['run_offset'] - run_period_key['run_onset']
            run_period_key['mean_velocity'] = np.nanmean(filt_treadmill_velocity[running_period])
            run_period_key['mean_speed'] = np.nanmean(filt_treadmill_speed[running_period])
            run_period_key['max_speed'] = np.nanmax(filt_treadmill_speed[running_period])
            total_run_duration += run_period_key['run_duration']
            running_period_keys.append(run_period_key)
        
        ## Calculate total running statistics
        treadmill_duration = np.nanmax(treadmill_times) - np.nanmin(treadmill_times)
        percent_running = total_run_duration / treadmill_duration * 100
        
        ## Insert key values 
        running_key = {**key, 'processed_treadmill': filt_treadmill_speed, 'total_run_duration': total_run_duration, 
                       'percent_running': percent_running, 'mean_run_velocity': mean_run_velocity,
                       'mean_run_speed': mean_run_speed, 'max_run_speed': max_run_speed, 
                       'num_run_periods': num_run_periods}
        self.insert1(running_key)
        self.Period.insert(running_period_keys)

        
    def _combine_running_periods(self, running_indices, tread_times, maximum_time_gap):

        """
        Given a flattened 1d array/list of indices, combines defined indices and indices between them if their gap
        is less than maximum_time_gap.

            Parameters:
                running_indices: Flattened list or numpy array of indices corresponding to detected running times
                tread_times: List or numpy array of times each recording on the treadmill was taken
                maximum_time_gap: Numeric value of maximum gap before running periods should not be combined. 
                                  Value is written in seconds.

            Returns:
                combined_running_fragments: List of lists of continuous incrementing indices corresponding to
                                            single periods of running.
        """

        ## Splits list of indices into a list of lists. Each sublist contains indices which continuously increase by 1.
        ## This function also drops sublists which only contain a single idx.
        running_fragments = clocktools.find_idx_boundaries(running_indices, drop_single_idx=True)

        ## If we have two or more fragments, test if any should be combined
        if len(running_fragments) > 1:

            ## Explanation: You can think of the proceeding code like having two cursors, one for start_idx and another
            ## for end_idx. Start by setting start and end cursors on the start and end of the first fragment. When 
            ## looking at the second fragment, one of two things happens:
            ## 1) The gap between the first and second fragment is below threshold. In that case, set the end cursor to 
            ##    the end of the second fragment and keep looping through.
            ## 2) The gap between the first and second fragment is greater than the threshold. In that case store an 
            ##    np.arange from the start cursor to the end cursor before setting those cursors to the start and end of
            ##    the second fragment, then continue looping through fragments.

            combined_running_fragments = []
            temp_start_idx = running_fragments[0][0]

            ## Fragment1 is the current fragment, while Fragment2 is the fragment directly afterwards. Loop stops one 
            ## fragment before the end of the list running_fragments.
            for fragment1, fragment2 in zip(running_fragments[:-1], running_fragments[1:]):

                fragment1_end_time = tread_times[fragment1[-1]]
                fragment2_start_time = tread_times[fragment2[0]]
                gap_duration = fragment2_start_time - fragment1_end_time

                if gap_duration > maximum_time_gap:
                    temp_end_idx = fragment1[-1]
                    combined_fragment = np.arange(temp_start_idx, temp_end_idx+1) ## +1 so we include final idx
                    combined_running_fragments.append(combined_fragment)
                    temp_start_idx = fragment2[0]

            ## Run check for last entry in running_fragments
            final_gap_start = tread_times[running_fragments[-2][-1]]
            final_gap_end = tread_times[running_fragments[-1][0]]
            final_gap_duration = final_gap_end - final_gap_start

            ## If the last entry should be combined, rewrite last array in combined_running_fragments
            if final_gap_duration <= maximum_time_gap:
                temp_end_idx = running_fragments[-1][-1]
                final_combined_fragment = np.arange(temp_start_idx, temp_end_idx+1)  ## +1 so we include final idx
                combined_running_fragments.append(final_combined_fragment)

            ## Append last running period if it doens't need to be combined
            else:
                combined_running_fragments.append(running_fragments[-1])

        else:

            ## If we only have 0 to 1 entries, nothing to combine
            combined_running_fragments = running_fragments

        return combined_running_fragments


    def _remove_short_running_periods(self, running_fragments, tread_times, minimum_run_length):
        """
        Given a list of lists of indices for running periods, removes any running period which is less than the
        defined minimum run length.

            Parameters:
                running_fragments: List/np.array of lists/np.arrays. Each sublist must correspond to the indices for
                                   a single period of running.
                tread_times: List or np.array of times each recording on the treadmill was taken
                minimum_run_legnth: Numeric value. The duration of a running period must be equal to or greater than
                                    this value to be returned. Units are seconds.

            Returns:
                long_running_fragments: List of lists of running fragments with a duration equal or above
                                        minimum_run_length. Each sublist contains indices correspond to said
                                        running period.

        """

        long_running_fragments = []
        for running_fragment in running_fragments:

            run_start = tread_times[running_fragment[0]]
            run_end = tread_times[running_fragment[-1]]
            run_duration = run_end - run_start

            if run_duration >= minimum_run_length:
                long_running_fragments.append(running_fragment)

        return long_running_fragments
    
    
    def get_nonrunning_idx(self, key):
        
        onsets, offsets = (Running.Period & key).fetch('run_onset', 'run_offset')
        frame_times = clocktools.fetch_timing_data(key, source_type='fluorescence-behavior', target_type='fluorescence-behavior')[0]

        non_running_indices = np.ones_like(frame_times).astype(bool)
        for onset,offset in zip(onsets, offsets):
            non_running_fragment = ~np.all(np.vstack((onset < frame_times, frame_times < offset)),axis=0)
            non_running_indices = np.all(np.vstack((non_running_fragment, non_running_indices)),axis=0)
        
        return non_running_indices
    
    
    def get_running_idx(self, key):
        
        onsets, offsets = (Running.Period & key).fetch('run_onset', 'run_offset')
        frame_times = clocktools.fetch_timing_data(key, source_type='fluorescence-behavior', target_type='fluorescence-behavior')[0]

        running_indices = np.zeros_like(frame_times).astype(bool)
        for onset,offset in zip(onsets, offsets):
            running_fragment = np.all(np.vstack((onset < frame_times, frame_times < offset)),axis=0)
            running_indices = np.any(np.vstack((running_fragment, running_indices)),axis=0)
        
        return running_indices


    def plot_running_periods(self, threshold=None, use_raw=False, x_stepsize=None, figsize=(100,10)):

        key = self.fetch1('KEY')
        
        ## Fetch data
        onsets, offsets = (Running.Period & key).fetch('run_onset', 'run_offset')
        tread_times = (Treadmill & key).fetch1('treadmill_time')
        if use_raw:
            tread_vel = (Treadmill & key).fetch1('treadmill_vel')
            tread_speed = np.abs(tread_vel)
        else:
            tread_speed = (Running & key).fetch1('processed_treadmill')

        ## Create figure
        with plt.style.context('fivethirtyeight'):
            fig,ax = plt.subplots(1,1,figsize=figsize)

            ## Plot treadmill speed
            ax.plot(tread_times, tread_speed, color='black', alpha=0.7, linewidth=2)

            ## Plot threshold line if given
            if threshold is not None:
                ax.axhline(threshold, color='C0', alpha=0.7, linewidth=2)

            ## Plot detected events
            for onset,offset in zip(onsets, offsets):
                ax.axvspan(onset, offset, color='C1', alpha=0.3)

            ## Clean Plot
            title = f"ID {key['animal_id']} Session {key['session']} Scan {key['scan_idx']}"
            ax.set_title(title, fontsize=50)
            ax.set_ylabel('Treadmill Speed (cm/sec)', fontsize=20)
            ax.set_xlabel('Scan Time (sec)', fontsize=20)
            if x_stepsize is not None:
                x_start, x_end = ax.get_xlim()
                ax.xaxis.set_ticks(np.arange(round(x_start,0), round(x_end,0), x_stepsize))

        return fig
