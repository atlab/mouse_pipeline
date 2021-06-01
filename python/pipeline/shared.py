""" Lookup tables shared among multi-photon pipelines. """

import datajoint as dj
import numpy as np
from scipy.signal import hamming, convolve, medfilt
from .exceptions import PipelineException

schema = dj.schema('pipeline_shared', locals(), create_tables=False)

@schema
class Field(dj.Lookup):
    definition = """ # fields in mesoscope scans
    field       : tinyint
    """
    contents = [[i] for i in range(1, 25)]

@schema
class Channel(dj.Lookup):
    definition = """  # recording channel, directly related to experiment.PMTFilterSet.Channel
    channel     : tinyint
    """
    contents = [[i] for i in range(1, 5)]

@schema
class PipelineVersion(dj.Lookup):
    definition = """ # versions for the reso pipeline

    pipe_version                    : smallint
    """
    contents = [[i] for i in range(3)]

@schema
class SegmentationMethod(dj.Lookup):
    definition = """ # methods for mask extraction for multi-photon scans
    segmentation_method         : tinyint
    ---
    name                        : varchar(16)
    details                     : varchar(255)
    language                    : enum('matlab', 'python')  # implementation language
    """
    contents = [
        [1, 'manual', '', 'matlab'],
        [2, 'nmf', 'constrained non-negative matrix factorization from Pnevmatikakis et al. (2016)',
         'python'],
        [3, 'nmf-patches', 'same as nmf but initialized in small image patches', 'python'],
        [4, 'nmf-boutons', 'nmf for axonal terminals', 'python'],
        [5, '3d-conv', 'masks from the segmentation of the stack', 'python'],
        [6, 'nmf-new', 'same as method 3 (nmf-patches) but with some better tuned params', 'python']
    ]

@schema
class StackSegmMethod(dj.Lookup):
    definition = """ # methods for 3-d stack segmentations
    stacksegm_method            : tinyint
    ---
    name                        : varchar(16)
    details                     : varchar(255)
    language                    : enum('matlab', 'python')  # implementation language
    """
    contents = [
        [1, '3dconv', '3-d convolutional network plus watershed', 'python'],
        [2, '3dconv-ensemble', 'an ensemble of 3-d convolutional networks plus watershed', 'python']
    ]

@schema
class ClassificationMethod(dj.Lookup):
    definition = """ # methods to classify extracted masks
    classification_method         : tinyint
    ---
    name                        : varchar(16)
    details                     : varchar(255)
    language                    : enum('matlab', 'python')  # implementation language
    """

    contents = [
    [1, 'manual', 'masks classified by visual inspection', 'python'],
    [2, 'cnn-caiman', 'classification made by a trained convolutional network', 'python']
    ]

@schema
class MaskType(dj.Lookup):
    definition = """ # possible classifications for a segmented mask
    type        : varchar(16)
    """
    contents = [
        ['soma'],
        ['axon'],
        ['dendrite'],
        ['neuropil'],
        ['artifact'],
        ['unknown']
    ]

@schema
class SpikeMethod(dj.Lookup):
    definition = """
    spike_method        : tinyint                   # spike inference method
    ---
    name                : varchar(16)               # short name to identify the spike inference method
    details             : varchar(255)              # more details
    language            : enum('matlab', 'python')  # implementation language
    """

    contents = [
        [2, 'foopsi', 'nonnegative sparse deconvolution from Vogelstein (2010)', 'python'],
        [3, 'stm', 'spike triggered mixture model from Theis et al. (2016)', 'python'],
        [5, 'nmf', 'noise constrained deconvolution from Pnevmatikakis et al. (2016)', 'python'],
        [6, 'dnmf', 'noise constrained deconvolution from Pnevmatikakis et al. (2016) on detrended fluorescence', 'python']
    ]

@schema
class RegistrationMethod(dj.Lookup):
    definition = """
    registration_method : tinyint                   # method used for registration
    ---
    name                : varchar(16)               # short name to identify the registration method
    details             : varchar(255)              # more details
    language            : enum('matlab', 'python')  # implementation language
    """

    contents = [
        [1, 'rigid', '3-d cross-correlation (40 microns above and below estimated z)', 'python'],
        [2, 'rigid2', '3-d cross-correlation (100 microns above and below estimated z)', 'python'],
        [3, 'affine', ('exhaustive search of 3-d rotations + cross-correlation (40 microns'
                       'above and below estimated z)'), 'python'],
        [4, 'affine2', ('exhaustive search of 3-d rotations + cross-correlation (100 microns'
                        'above and below estimated z)'), 'python'],
        [5, 'non-rigid', 'affine plus deformation field learnt via gradient ascent on correlation', 'python']
    ]

@schema
class CurationMethod(dj.Lookup):
    definition = """
    curation_method     : tinyint                   # method to curate the initial registration estimates
    ---
    name                : varchar(16)               # short name to identify the curation method
    details             : varchar(255)              # more details
    language            : enum('matlab', 'python')  # implementation language
    """

    contents = [
        [1, 'none', 'estimates are left unchanged', 'python'],
        [2, 'manual', 'manually inspect each field estimate', 'matlab'],
    ]

@schema
class AreaMaskMethod(dj.Lookup):
    definition = """
    # method for assigning cortex to visual areas
    mask_method                 : tinyint           # method to assign membership to visual areas
    ---
    name                        : varchar(16)
    details                     : varchar(255)
    language                    : enum('matlab', 'python')  # implementation language
    """

    contents = [
        [1, 'manual', '', 'matlab'],
    ]

@schema
class TrackingMethod(dj.Lookup):
    definition = """
    tracking_method : tinyint                       # method used for pupil tracking
    ---
    name                : varchar(16)               # short name to identify the tracking method
    details             : varchar(255)              # more details
    language            : enum('matlab', 'python')  # implementation language
    """

    contents = [
        [1, 'manual', 'manually tracking', 'python'],
        [2, 'deeplabcut', 'automatically tracking using deeplabcut package', 'python'],
    ]

@schema
class SurfaceMethod(dj.Lookup):
    definition = """ # Methods used to compute surface of the brain

    surface_method_id   : tinyint unsigned   # Unique ID given to each surface calculation method
    ---
    method_title        : varchar(32)        # Title of surface calculation method
    method_description  : varchar(256)       # Details on surface calculation
    """

    contents = [
        [1, 'Paraboloid Fit', 'Fit ax^2 + by^2 + cx + dy + f to surface after finding max of sobel']
    ]

@schema
class ExpressionConstruct(dj.Lookup):
    definition = """ # Construct expressed within certain fields
    construct_label               : varchar(64)                        # name of construct expressed/injected within field
    ---
    construct_notes               : varchar(256)
    """

@schema
class FilterMethod(dj.Lookup):
    definition = """ # Established methods to filter time series signals
    filter_method                 : varchar(32)         # name of filter method
    ---
    arguments                     : varchar(256)        # required values to pass to the function
    filter_description            : varchar(256)        # description of filter method
    """
    contents = [['0.5Hz Hamming Lowpass', 'signal, signal_freq (Hz)', '0.5Hz lowpass filter using hamming window'],
                ['1Hz Hamming Lowpass', 'signal, signal_freq (Hz)', '1Hz lowpass filter, zero-phase (without added delay) using Hamming window'],
                ['2Hz Hamming Lowpass', 'signal, signal_freq (Hz)', '2Hz lowpass filter, zero-phase (without added delay) using Hamming window'],
                ['0.1 - 1Hz Hamming Bandpass', 'signal, signal_freq (Hz)', '0.1 - 1Hz bandpass filter, zero-phase (without added delay) using Hamming window'],
                ['0.5sec Median Filter', 'signal, signal_freq (Hz)', 'Filter which returns median value over sliding 0.5sec window'],
                ['NaN Filler', 'signal', 'Linearly interpolates over all NaNs in a signal with NaNs outside of bounds filled via nearest neighbor interpolation'],
               ]
    
    
    def _make_hamming_window(signal_freq, lowpass_freq, *args, **kwargs):
        """
        Create an array representing a hamming function which lowpass filters at lowpass_fs frequency on
        a signal of signal_fs frequency.
        
        Parameters:
            signal_freq: Float/Int representing frequency of signal to be filtered in Hz (FPS)
            lowpass_freq: Float/Int representing frequency to lowpass filter at in Hz (FPS)
            
        Returns:
            hamming_filter: Numpy array of requested hamming window
        """
        
        hamming_length = int(2*round(signal_freq/lowpass_freq,0)+1)
        hamming_filter = hamming(hamming_length, sym=True)
        hamming_filter = hamming_filter/np.sum(hamming_filter)
        
        return hamming_filter


    def _lowpass_hamming(signal, signal_freq, lowpass_freq, *args, **kwargs):
        """
        Lowpass filters a given signal using a hamming window. Reflects the signal at start/end to avoid
        any edge artifacts. Filtering is zero-phase, meaning the signal will not be delayed in time after
        filtering.
        
        Parameters:
            signal: Numpy array of the signal to be filtered
            signal_freq: Float/Int sampling frequency of the input_signal in Hz (FPS)
            lowpass_freq: Float/Int frequency to lowpass the signal at in Hz
            
        Returns:
            filtered_signal: Numpy array of the lowpass filtered input signal
        """
        
        hamming_filter = FilterMethod._make_hamming_window(signal_freq, lowpass_freq)

        pad_length = len(hamming_filter)
        padded_signal = np.pad(signal, pad_length, mode='reflect')

        filtered_signal = convolve(padded_signal, hamming_filter, mode='same')
        filtered_signal = filtered_signal[pad_length:-pad_length]

        return filtered_signal


    def _bandpass_hamming(signal, signal_freq, lower_freq, upper_freq, *args, **kwargs):
        """
        Bandpass filters a given signal using a hamming window. Reflects the signal at the start/end to
        avoid any edge artifacts. Filtering is zero-phase, meaning the signal will not be delayed in time
        after filtering.
        
        Parameters:
            signal: Numpy array of the signal to be filtered
            signal_freq: Float/Int sampling frequency of the input_signal in Hz (FPS)
            lower_freq: Float/Int lower frequency of bandpass in Hz
            upper_freq: Float/Int upper frequency of bandpass in Hz
            
        Returns:
            filtered_signal: Numpy array of the bandpass filtered input signal
        """
        
        upper_hamming_filter = FilterMethod._make_hamming_window(signal_freq, upper_freq)
        lower_hamming_filter = FilterMethod._make_hamming_window(signal_freq, lower_freq)

        pad_length = np.max((len(upper_hamming_filter), len(lower_hamming_filter)))
        padded_signal = np.pad(signal, pad_length, mode='reflect')

        highpass_signal = padded_signal - convolve(padded_signal, lower_hamming_filter, mode='same')
        filtered_signal = convolve(highpass_signal, upper_hamming_filter, mode='same')
        filtered_signal = filtered_signal[pad_length:-pad_length]

        return filtered_signal
    
    
    def _median_filter(signal, signal_freq, window_sec, *args, **kwargs):
        """
        Filters the signal by taking the median value within predefined window centered around every point
        in the signal. Reflects signal at the start/end to avoid any edge artifacts.
        
        Parameters:
            signal: Numpy array of the signal to be filtered
            signal_freq: Float/Int sampling frequency of the input_signal in Hz (FPS)
            window_sec: Float/Int length of filter window to look for median in seconds
            
        Returns:
            filtered_signal: Numpy array of the median filtered input signal
        """
        
        window_idx_length = round(signal_freq * window_sec)
        
        ## Window length must be odd
        if window_idx_length % 2 == 0:
            window_idx_length = window_idx_length + 1
        
        pad_length = window_idx_length
        padded_signal = np.pad(signal, pad_length, mode='reflect')
        
        filtered_signal = medfilt(padded_signal, window_idx_length)
        filtered_signal = filtered_signal[pad_length:-pad_length]
        
        return filtered_signal
    
    
    def _nan_filler(signal, *args, **kwargs):
        """
        Linearly interpolates over all NaNs in a signal. Values outside of non-NaN signal bounds
        are filled with nearest neighbor.
        
        ex. [np.nan, np.nan, 1, 2, 3, np.nan, 5] -> [1, 1, 1, 2, 3, 4, 5]
        
        Parameters:
            signal: Numpy array of the signal to be filtered
            
        Returns:
            filtered_signal: Numpy array of the interpolate signal with no NaNs
        """
        
        
        
        nan_bool_mask = np.isnan(signal)
        nan_idx = nan_bool_mask.nonzero()[0]
        non_nan_bool_mask = ~nan_bool_mask
        non_nan_idx = (non_nan_bool_mask).nonzero()[0]
        
        filtered_signal = np.copy(signal)
        filled_values = np.interp(nan_idx, non_nan_idx, filtered_signal[non_nan_idx])
        filtered_signal[nan_idx] = filled_values
        
        return filtered_signal
        
    
    def run_filter(self, *args, **kwargs):
            
        filter_method = self.fetch1('filter_method')
        filters_allowing_nans = ('NaN Filler',)
        
        ## Check if there are NaNs
        if filter_method not in filters_allowing_nans:
            if 'signal' in kwargs:
                signal_to_test = kwargs['signal']
            else:
                signal_to_test = args[0]
            if np.any(np.isnan(signal_to_test)):
                raise PipelineException('Given signal contains NaNs. Try using run_filter_with_renan() method instead.')
        
        ## Run requested filter
        if filter_method == "0.5Hz Hamming Lowpass":
            kwargs['lowpass_freq'] = 0.5
            filtered_signal = FilterMethod._lowpass_hamming(*args, **kwargs)
        
        elif filter_method == "1Hz Hamming Lowpass":
            kwargs['lowpass_freq'] = 1
            filtered_signal = FilterMethod._lowpass_hamming(*args, **kwargs)
        
        elif filter_method == "2Hz Hamming Lowpass":
            kwargs['lowpass_freq'] = 2
            filtered_signal = FilterMethod._lowpass_hamming(*args, **kwargs)
        
        elif filter_method == "0.1 - 1Hz Hamming Bandpass":
            kwargs['lower_freq'] = 0.1
            kwargs['upper_freq'] = 1
            filtered_signal = FilterMethod._bandpass_hamming(*args, **kwargs)
        
        elif filter_method == '0.5sec Median Filter':
            kwargs['window_sec'] = 0.5
            filtered_signal = FilterMethod._median_filter(*args, **kwargs)
        
        elif filter_method == 'NaN Filler':
            filtered_signal = FilterMethod._nan_filler(*args, **kwargs)
        
        else:
            msg = f'Error: Filter method {filter_method} is not defined.'
            raise Exception(msg)
        
        return filtered_signal
    
    
    def run_filter_with_renan(self, *args, **kwargs):
        """
        Runs selected filter method by first linearly interpolating over all NaNs before running
        filter. Resulting signal then has NaNs re-added into the filtered signal at the same
        locations. See arguments table for specific inputs and outputs.
        """
        
        ## Linearly interpolate over NaNs
        nan_filter_func = (FilterMethod & {'filter_method': 'NaN Filler'}).run_filter
        if 'signal' in kwargs:
            nan_indices = np.where(np.isnan(kwargs['signal']))[0]
            kwargs['signal'] = nan_filter_func(kwargs['signal'])
        else:
            nan_indices = np.where(np.isnan(args[0]))[0]
            args = list(args)
            args[0] = nan_filter_func(args[0])
        
        ## Run filter
        filtered_signal = self.run_filter(*args, **kwargs)
        
        ## ReNaN
        filtered_signal[nan_indices] = np.nan
        
        return filtered_signal
