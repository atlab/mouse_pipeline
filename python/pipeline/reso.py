""" Schemas for resonant scanners."""
import datajoint as dj
from datajoint.jobs import key_hash
import matplotlib.pyplot as plt
import numpy as np
import scanreader
import gc

from . import experiment, notify, shared
from .utils import galvo_corrections
from .utils import signal
from .utils import quality
from .exceptions import PipelineException


schema = dj.schema('pipeline_reso', locals())


def erd():
    """a shortcut for convenience"""
    dj.ERD(schema).draw(prefix=False)


@schema
class ScanInfo(dj.Imported):
    definition = """ # master table with general data about the scans

    -> experiment.Scan
    ---
    nslices                 : tinyint           # number of slices
    nchannels               : tinyint           # number of recorded channels
    nframes                 : int               # number of recorded frames
    nframes_requested       : int               # number of frames (from header)
    px_height               : smallint          # lines per frame
    px_width                : smallint          # pixels per line
    um_height               : float             # height in microns
    um_width                : float             # width in microns
    x                       : float             # (um) center of scan in the motor coordinate system
    y                       : float             # (um) center of scan in the motor coordinate system
    fps                     : float             # (Hz) frames per second
    zoom                    : decimal(5,2)      # zoom factor
    bidirectional           : boolean           # true = bidirectional scanning
    usecs_per_line          : float             # microseconds per scan line
    fill_fraction           : float             # raster scan temporal fill fraction (see scanimage)
    """

    @property
    def key_source(self):
        rigs = [{'rig': '2P2'}, {'rig': '2P3'}, {'rig': '2P5'}]
        return (experiment.Scan() - experiment.ScanIgnored()) & (experiment.Session() & rigs)

    class Slice(dj.Part):
        definition = """ # slice-specific scan information

        -> ScanInfo
        -> shared.Slice
        ---
        z           : float             # (um) absolute depth with respect to the surface of the cortex
        """

    class QuantalSize(dj.Part):
        definition = """ # quantal size in images

        -> ScanInfo
        -> shared.Slice
        -> shared.Channel
        ---
        min_intensity               : int           # min value in movie
        max_intensity               : int           # max value in movie
        intensities                 : longblob      # intensities for fitting variances
        variances                   : longblob      # variances for each intensity
        quantal_size                : float         # variance slope, corresponds to quantal size
        zero_level                  : int           # level corresponding to zero (computed from variance dependence)
        quantal_frame               : longblob      # average frame expressed in quanta
        median_quantum_rate         : float         # median value in frame
        percentile95_quantum_rate   : float         # 95th percentile in frame
        """

        def _make_tuples(self, key, scan, slice_id, channel):
            # Create results tuple
            tuple_ = key.copy()
            tuple_['slice'] = slice_id + 1
            tuple_['channel'] = channel + 1

            # Compute quantal size
            middle_frame =  int(np.floor(scan.num_frames / 2))
            frames = slice(max(middle_frame - 2000, 0), middle_frame + 2000)
            mini_scan = scan[slice_id, :, :, channel, frames]
            results = quality.compute_quantal_size(mini_scan)

            # Add results to tuple
            tuple_['min_intensity'] = results[0]
            tuple_['max_intensity'] = results[1]
            tuple_['intensities'] = results[2]
            tuple_['variances'] = results[3]
            tuple_['quantal_size'] = results[4]
            tuple_['zero_level'] = results[5]

            # Compute average frame rescaled with the quantal size
            mean_frame = np.mean(mini_scan, axis=-1)
            average_frame = (mean_frame - tuple_['zero_level']) / tuple_['quantal_size']
            tuple_['quantal_frame'] = average_frame
            tuple_['median_quantum_rate'] = np.median(average_frame)
            tuple_['percentile95_quantum_rate'] = np.percentile(average_frame, 95)

            # Insert
            self.insert1(tuple_)

    def _make_tuples(self, key):
        """ Read some scan parameters, compute FOV in microns and quantal size."""
        from decimal import Decimal

        # Read the scan
        print('Reading header...')
        scan_filename = (experiment.Scan() & key).local_filenames_as_wildcard
        scan = scanreader.read_scan(scan_filename, dtype=np.float32)

        # Get attributes
        tuple_ = key.copy()  # in case key is reused somewhere else
        tuple_['nslices'] = scan.num_fields
        tuple_['nchannels'] = scan.num_channels
        tuple_['nframes'] = scan.num_frames
        tuple_['nframes_requested'] = scan.num_requested_frames
        tuple_['px_height'] = scan.image_height
        tuple_['px_width'] = scan.image_width
        tuple_['x'] = scan.motor_position_at_zero[0]
        tuple_['y'] = scan.motor_position_at_zero[1]
        tuple_['fps'] = scan.fps
        tuple_['zoom'] = Decimal(str(scan.zoom))
        tuple_['bidirectional'] = scan.is_bidirectional
        tuple_['usecs_per_line'] = scan.seconds_per_line * 1e6
        tuple_['fill_fraction'] = scan.temporal_fill_fraction

        # Estimate height and width in microns using measured FOVs for similar setups
        fov_rel = (experiment.FOV() * experiment.Session() * experiment.Scan() & key
                   & 'session_date>=fov_ts')
        zooms = fov_rel.fetch['mag'].astype(np.float32)  # measured zooms in setup
        closest_zoom = zooms[np.argmin(np.abs(np.log(zooms / scan.zoom)))]

        interval = 'ABS(mag - {}) < 1e-4'.format(closest_zoom)
        um_height, um_width = (fov_rel & interval).fetch1['height', 'width']
        tuple_['um_height'] = float(um_height) * (closest_zoom / scan.zoom) * scan._y_angle_scale_factor
        tuple_['um_width'] = float(um_width) * (closest_zoom / scan.zoom) * scan._x_angle_scale_factor

        # Insert in ScanInfo
        self.insert1(tuple_)

        # Insert slice information
        slice_depths = [z * scan.zstep_in_microns for z in scan.field_depths]
        depth_zero = (experiment.Scan() & key).fetch1['depth'] # true z at ScanImage's 0
        for slice_id, slice_depth in enumerate(slice_depths):
            ScanInfo.Slice().insert1({**key, 'slice': slice_id + 1, 'z': depth_zero + slice_depth})

        # Compute quantal size for all slice/channel combinations
        print('Computing quantal size...')
        for slice_id in range(scan.num_fields):
            for channel in range(scan.num_channels):
                ScanInfo.QuantalSize()._make_tuples(key, scan, slice_id, channel)

        self.notify(key)

    def notify(self, key):
        # --  notification
        d = {k: v for k, v in key.items() if k in ['animal_id', 'session', 'scan_idx']}
        msg = 'ScanInfo for `{}` has been populated.'.format(d)
        (notify.SlackUser() & (experiment.Session() & key)).notify(msg)

    @ property
    def microns_per_pixel(self):
        """ Returns an array with microns per pixel in height and width. """
        um_height, px_height, um_width, px_width = self.fetch1['um_height', 'px_height',
                                                               'um_width', 'px_width']
        return np.array([um_height/px_height, um_width/px_width])

    def save_video(self, filename='galvo_corrections.mp4', slice_id=1, channel=1,
                   start_index=0, seconds=30, dpi=250):
        """ Creates an animation video showing the original vs corrected scan.

        :param string filename: Output filename (path + filename)
        :param int slice_id: Slice to use for plotting. Starts at 1
        :param int channel: What channel from the scan to use. Starts at 1
        :param int start_index: Where in the scan to start the video.
        :param int seconds: How long in seconds should the animation run.
        :param int dpi: Dots per inch, controls the quality of the video.

        :returns Figure. You can call show() on it.
        :rtype: matplotlib.figure.Figure
        """
        # Get fps and total_num_frames
        fps = (ScanInfo() & self).fetch1['fps']
        num_video_frames = int(round(fps * seconds))
        stop_index = start_index + num_video_frames

        # Load the scan
        scan_filename = (experiment.Scan() & self).local_filenames_as_wildcard
        scan = scanreader.read_scan(scan_filename, dtype=np.float32)
        scan_ = scan[slice_id - 1, :, :, channel - 1, start_index: stop_index]
        original_scan = scan_.copy()

        # Correct the scan
        correct_raster = RasterCorrection.get_correct_raster(self.fetch1())
        correct_motion = MotionCorrection.get_correct_motion({**self.fetch1(), 'slice': slice_id})
        corrected_scan = correct_motion(correct_raster(scan_), slice(start_index, stop_index))

        # Create animation
        import matplotlib.animation as animation

        ## Set the figure
        fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)

        axes[0].set_title('Original')
        im1 = axes[0].imshow(original_scan[:, :, 0], vmin=original_scan.min(),
                             vmax=original_scan.max())  # just a placeholder
        fig.colorbar(im1, ax=axes[0])
        axes[0].axis('off')

        axes[1].set_title('Corrected')
        im2 = axes[1].imshow(corrected_scan[:, :, 0], vmin=corrected_scan.min(),
                         vmax=corrected_scan.max())  # just a placeholder
        fig.colorbar(im2, ax=axes[1])
        axes[1].axis('off')

        ## Make the animation
        def update_img(i):
            im1.set_data(original_scan[:, :, i])
            im2.set_data(corrected_scan[:, :, i])

        video = animation.FuncAnimation(fig, update_img, corrected_scan.shape[2],
                                        interval=1000 / fps)

        # Save animation
        if not filename.endswith('.mp4'):
            filename += '.mp4'
        print('Saving video at:', filename)
        print('If this takes too long, stop it and call again with dpi <', dpi, '(default)')
        video.save(filename, dpi=dpi)

        return fig


@schema
class CorrectionChannel(dj.Manual):
    definition = """ # channel to use for raster and motion correction

    -> experiment.Scan
    -> shared.Slice
    ---
    -> shared.Channel
    """


@schema
class RasterCorrection(dj.Computed):
    definition = """ # raster correction for bidirectional resonant scans

    -> ScanInfo
    ---
    -> shared.Slice                     # slice used for raster correction
    -> shared.Channel                   # channel used for raster correction
    template            : longblob      # average frame from the middle of the movie
    raster_phase        : float         # difference between expected and recorded scan angle
    """

    @property
    def key_source(self):
        # Run make_tuples iff correction channel has been set for all slices
        return (ScanInfo() & CorrectionChannel()) - (ScanInfo.Slice() - CorrectionChannel())

    def _make_tuples(self, key):
        # Read the scan
        scan_filename = (experiment.Scan() & key).local_filenames_as_wildcard
        scan = scanreader.read_scan(scan_filename, dtype=np.float32)

        # Select (middle) slice and channel
        slice_id = int(np.floor(scan.num_fields / 2))
        channel = (CorrectionChannel() & key & {'slice': slice_id + 1}).fetch1['channel'] - 1

        # Create results tuple
        tuple_ = key.copy()
        tuple_['slice'] = slice_id + 1
        tuple_['channel'] = channel + 1

        # Create the template (an average frame from the middle of the scan)
        middle_frame =  int(np.floor(scan.num_frames / 2))
        frames = slice(max(middle_frame - 1000, 0), middle_frame + 1000)
        mini_scan = scan[slice_id, :, :, channel, frames]
        template = np.mean(mini_scan, axis=-1)
        tuple_['template'] = template

        # Compute raster correction parameters
        if scan.is_bidirectional:
            tuple_['raster_phase'] = galvo_corrections.compute_raster_phase(template,
                                                            scan.temporal_fill_fraction)
        else:
            tuple_['raster_phase'] = 0

        # Insert
        self.insert1(tuple_)
        self.notify(key)

    def notify(self, key):
        msg = 'RasterCorrection for `{}` has been populated.'.format(key)
        msg += '\n Raster phase: {}'.format((self & key).fetch1['raster_phase'])
        (notify.SlackUser() & (experiment.Session() & key)).notify(msg)

    @staticmethod
    def get_correct_raster(key):
        """
         :returns: A function to perform raster correction on the scan
                [image_height, image_width, channels, slices, num_frames].
        """
        key = {k: v for k, v in key.items() if k in RasterCorrection().primary_key}

        raster_phase = (RasterCorrection() & key).fetch1['raster_phase']
        fill_fraction = (ScanInfo() & key).fetch1['fill_fraction']
        if raster_phase == 0:
            return lambda scan: scan.astype(np.float32, copy=False)
        else:
            return lambda scan: galvo_corrections.correct_raster(scan, raster_phase,
                                                                 fill_fraction)


@schema
class MotionCorrection(dj.Computed):
    definition = """ # motion correction for galvo scans

    -> RasterCorrection
    -> shared.Slice
    ---
    -> shared.Channel                               # channel used for motion correction
    template                        : longblob      # image used as alignment template
    y_shifts                        : longblob      # (pixels) y motion correction shifts
    x_shifts                        : longblob      # (pixels) x motion correction shifts
    y_std                           : float         # (um) standard deviation of y shifts
    x_std                           : float         # (um) standard deviation of x shifts
    y_outlier_frames                : longblob      # mask with true for frames with high y shifts (already corrected)
    x_outlier_frames                : longblob      # mask with true for frames with high x shifts (already corrected)
    align_times=CURRENT_TIMESTAMP   : timestamp     # automatic
    """

    @property
    def key_source(self):
        return RasterCorrection() # run make_tuples once per scan

    def _make_tuples(self, key):
        """Computes the motion shifts per frame needed to correct the scan."""
        from scipy import ndimage

        # Read the scan
        scan_filename = (experiment.Scan() & key).local_filenames_as_wildcard
        scan = scanreader.read_scan(scan_filename, dtype=np.float32)

        # Get some params
        um_height, px_height, um_width, px_width = \
            (ScanInfo() & key).fetch1['um_height', 'px_height', 'um_width', 'px_width']

        for slice_id in range(scan.num_fields):
            print('Correcting motion in slice', slice_id + 1)

            # Select channel
            channel = (CorrectionChannel() & key & {'slice': slice_id + 1}).fetch1['channel'] - 1

            # Create results tuple
            tuple_ = key.copy()
            tuple_['slice'] = slice_id + 1
            tuple_['channel'] = channel + 1

            # Load scan (we discard some rows/cols to avoid edge artifacts)
            skip_rows = int(round(px_height * 0.10))
            skip_cols = int(round(px_width * 0.10))
            scan_ = scan[slice_id, skip_rows: -skip_rows, skip_cols: -skip_cols, channel, :]  # 3-d (height, width, frames)

            # Correct raster effects (needed for subpixel changes in y)
            correct_raster = RasterCorrection.get_correct_raster(key)
            scan_ = correct_raster(scan_)
            scan_ -= scan_.min() # make nonnegative for fft

            # Create template
            middle_frame =  int(np.floor(scan.num_frames / 2))
            mini_scan = scan_[:, :, max(middle_frame - 1000, 0): middle_frame + 1000]
            mini_scan = 2 * np.sqrt(mini_scan + 3/8) # *
            template = np.mean(mini_scan, axis=-1)
            template = ndimage.gaussian_filter(template, 0.7) # **
            tuple_['template'] = template
            # * Anscombe tranform to normalize noise, increase contrast and decrease outlier's leverage
            # ** Small amount of gaussian smoothing to get rid of high frequency noise

            # Compute smoothing window size
            size_in_ms = 300 # smooth over a 300 milliseconds window
            window_size = int(round(scan.fps * (size_in_ms / 1000))) # in frames
            window_size += 1 if window_size % 2 == 0 else 0 # make odd

            # Get motion correction shifts
            results = galvo_corrections.compute_motion_shifts(scan_, template,
                smoothing_window_size=window_size)
            y_shifts = results[0] - results[0].mean() # center motions around zero
            x_shifts = results[1] - results[1].mean()
            tuple_['y_shifts'] = y_shifts
            tuple_['x_shifts'] = x_shifts
            tuple_['y_outlier_frames'] = results[2]
            tuple_['x_outlier_frames'] = results[3]
            tuple_['y_std'] = np.std(y_shifts)
            tuple_['x_std'] = np.std(x_shifts)

            # Free memory
            del scan_
            gc.collect()

            # Insert
            self.insert1(tuple_)

        self.notify(key, scan)

    def notify(self, key, scan):
        # --  notification
        import matplotlib
        matplotlib.rcParams['backend'] = 'Agg'
        import matplotlib.pyplot as plt
        import seaborn as sns

        fps = (ScanInfo() & key).fetch1['fps']
        seconds = np.arange(scan.num_frames) / fps

        with sns.axes_style('white'):
            fig, axes = plt.subplots(scan.num_fields, 1, figsize=(10, 5 * scan.num_channels))
        for i in range(scan.num_fields):
            y_shifts, x_shifts = (self & key & {'slice': i + 1}).fetch1['y_shifts', 'x_shifts']
            axes[i].set_title('Shifts for slice {}'.format(i + 1))
            axes[i].plot(seconds, y_shifts, label='y shifts')
            axes[i].plot(seconds, x_shifts, label='x shifts')
            axes[i].set_ylabel('Pixels')
            axes[i].set_xlabel('Seconds')
            axes[i].legend()
        fig.tight_layout()
        img_filename = '/tmp/' + key_hash(key) + '.png'
        fig.savefig(img_filename)
        plt.close(fig)

        d = {k: v for k, v in key.items() if k in ['animal_id', 'session', 'scan_idx']}
        msg = 'MotionCorrection for `{}` has been populated.'.format(d)
        (notify.SlackUser() & (experiment.Session() & key)).notify(msg, file=img_filename,
                                                                   file_title='motion shifts')

    @staticmethod
    def get_correct_motion(key):
        """
        :returns: A function to performs motion correction on scans
                  [image_height, image_width, channels, slices, num_frames].
        """
        key = {k: v for k, v in key.items() if k in MotionCorrection().primary_key}

        y_shifts, x_shifts = (MotionCorrection() & key).fetch1['y_shifts', 'x_shifts']
        xy_motion = np.stack([x_shifts, y_shifts])
        def my_lambda_function(scan, indices=None):
            if indices is None:
                return galvo_corrections.correct_motion(scan, xy_motion)
            else:
                return galvo_corrections.correct_motion(scan, xy_motion[:, indices])

        return my_lambda_function



@schema
class SummaryImages(dj.Computed):
    definition = """ # summary images for each slice and channel after corrections

    -> MotionCorrection
    -> shared.Channel
    ---
    average             : longblob          # l6-norm (across time) of each pixel
    correlation         : longblob          # (average) temporal correlation between each pixel and its eight neighbors
    """

    @property
    def key_source(self):
        return MotionCorrection()

    def _make_tuples(self, key):
        from .utils import correlation_image as ci

        print('Computing summary images for', key)

        # Read the scan
        scan_filename = (experiment.Scan() & key).local_filenames_as_wildcard
        scan = scanreader.read_scan(scan_filename, dtype=np.float32)

        # Get raster correction function
        correct_raster = RasterCorrection.get_correct_raster(key)

        for channel in range(scan.num_channels):
            tuple_ = key.copy()
            tuple_['channel'] = channel + 1

            # Correct scan
            correct_motion = MotionCorrection.get_correct_motion(key)
            scan_ = scan[key['slice'] - 1, :, :, channel, :]
            scan_ = correct_motion(correct_raster(scan_))
            scan_ -= scan_.min() # make nonnegative for lp-norm

            # Compute and insert correlation image
            tuple_['correlation'] = ci.compute_correlation_image(scan_)

            # Compute and insert lp-norm of each pixel over time
            p = 6
            scan_ = np.power(scan_, p, out=scan_) # in place
            tuple_['average'] = np.sum(scan_, axis=-1, dtype=np.float64) ** (1 / p)

            # Insert
            self.insert1(tuple_)

        self.notify(key, scan)

    def notify(self, key, scan):
        # --  notification
        import matplotlib
        matplotlib.rcParams['backend'] = 'Agg'
        import matplotlib.pyplot as plt
        import seaborn as sns

        with sns.axes_style('white'):
            fig, ax = plt.subplots(2, scan.num_channels, figsize=(5 * scan.num_channels, 5.5 * 2))
        for i, img_name in enumerate(['average', 'correlation']):
            for channel in range(scan.num_channels):
                image = (self & key & {'channel': channel + 1}).fetch1[img_name]
                ax[i, channel].set_title('Channel {}: {}'.format(channel + 1, img_name))
                ax[i, channel].matshow(image, cmap='gray')
                ax[i, channel].axis('off')
        fig.suptitle('Slice {}'.format(key['slice']))
        fig.tight_layout()
        img_filename = '/tmp/' + key_hash(key) + '.png'
        fig.savefig(img_filename)
        plt.close(fig)

        d = {k: v for k, v in key.items() if k in ['animal_id', 'session', 'scan_idx', 'slice']}
        msg = 'SummaryImages for `{}` has been populated.'.format(d)
        (notify.SlackUser() & (experiment.Session() & key)).notify(msg, file=img_filename,
                                                                   file_title='summary images')


@schema
class SegmentationTask(dj.Manual):
    definition = """ # defines the target of segmentation and the channel to use

    -> experiment.Scan
    -> shared.Slice
    -> shared.Channel
    ---
    -> experiment.Compartment
    """

    def estimate_num_components(self):
        """ Estimates the number of components per slice using simple rules of thumb.

        For somatic scans, estimate number of neurons based on:
        (100x100x100)um^3 = 1e6 um^3 -> 1e2 neurons; (1x1x1)mm^3 = 1e9 um^3 -> 1e5 neurons

        For axonal/dendritic scans, just ten times our estimate of neurons.

        :returns: Number of components
        :rtype: int
        """

        # Get slice dimensions (in micrometers)
        slice_height, slice_width = (ScanInfo() & self).fetch1['um_height', 'um_width']
        slice_thickness = 10  # assumption
        slice_volume = slice_width * slice_height * slice_thickness

        # Estimate number of components
        compartment = self.fetch1['compartment']
        if compartment == 'soma':
            num_components = slice_volume * 0.0001
        elif compartment == 'axon':
            num_components = slice_volume * 0.001 # ten times as many neurons
        else:
            PipelineException("Compartment type '{}' not recognized".format(compartment))

        return int(round(num_components))


@schema
class ManualSegmentation(dj.Manual):
    definition = """ # masks created manually

    -> experiment.Scan
    -> shared.Slice
    -> shared.Channel
    ---
    extract_method = 1              : tinyint       # manual method in shared.SegmentationMethod
    timestamp=CURRENT_TIMESTAMP     : timestamp     # automatic
    """

    def delete(self):
        """ Override delete to also delete tuple from Segmentation. """
        if Segmentation() & self:
            dj.BaseRelation.delete(Segmentation() & self) # simple delete to avoid infinite recursion
        if self:
            super().delete()


@schema
class NMF(dj.Computed):
    definition = """ # source extraction using constrained non-negative matrix factorization (Pnevmatikakis et al., 2016)

    -> MotionCorrection
    -> SegmentationTask
    ---
    extract_method = 2              : tinyint       # nmf method in shared.SegmentationMethod
    """

    class Parameters(dj.Part):
        definition = """ # parameters used to demix and deconvolve the scan with CNMF

        -> NMF
        ---
        num_components                  : smallint      # estimated number of components
        ar_order                        : tinyint       # order of the autoregressive process for impulse response function
        merge_threshold                 : float         # overlapping masks are merged if temporal correlation greater than this
        num_processes = null            : smallint      # number of processes to run in parallel, null=all available
        num_pixels_per_process          : int           # number of pixels processed at a time
        block_size                      : int           # number of pixels per each dot product
        num_background_components       : smallint      # number of background components
        init_method                     : enum("greedy_roi", "sparse_nmf", "local_nmf") # type of initialization used
        soma_radius = null              : blob          # (y, x in pixels) estimated radius for a soma in the scan
        snmf_alpha = null               : float         # regularization parameter for SNMF
        init_on_patches                 : boolean       # whether to run initialization on small patches
        patch_downsampling_factor = null : tinyint      # used to calculate size of patches
        percentage_of_patch_overlap = null : float      # overlap between adjacent patches
        """

    class BackgroundComponents(dj.Part):
        definition = """ # inferred background components

        -> NMF
        ---
        masks               : longblob      # array (im_width x im_height x num_background_components)
        activity            : longblob      # array (num_background_components x timesteps)
        """

    #TODO: Defer this to Activity table, this should be created down there.
    class ARCoefficients(dj.Part):
        definition = """ # fitted parameters for the autoregressive process

        -> NMF
        -> Segmentation.Mask
        ---
        g                   : blob          # g1, g2, ... coefficients for the AR process
    """

    def _make_tuples(self, key):
        """ Use CNMF to extract masks and traces.

        See caiman_interface.demix_and_deconvolve_with_cnmf for explanation of params
        """
        from .utils import caiman_interface as cmn

        print('')
        print('*' * 85)
        print('Processing {}'.format(key))

        # Load scan
        channel = key['channel'] - 1
        slice_id = key['slice'] - 1
        scan_filename = (experiment.Scan() & key).local_filenames_as_wildcard
        scan = scanreader.read_scan(scan_filename, dtype=np.float32)
        scan_ = scan[slice_id, :, :, channel, :]

        # Correct scan
        print('Correcting scan...')
        correct_raster = RasterCorrection.get_correct_raster(key)
        correct_motion = MotionCorrection.get_correct_motion(key)
        scan_ = correct_motion(correct_raster(scan_))
        scan_ -= scan_.min() # make nonnegative for caiman

        # Set CNMF parameters
        ## Estimate number of components per slice
        num_components = (SegmentationTask() & key).estimate_num_components()

        ## Set general parameters
        kwargs = {}
        kwargs['num_components'] = num_components
        kwargs['AR_order'] = 2  # impulse response modelling with AR(2) process
        kwargs['merge_threshold'] = 0.8

        ## Set performance/execution parameters (heuristically), decrease if memory overflows
        kwargs['num_processes'] = 20  # Set to None for all cores available
        kwargs['num_pixels_per_process'] = 10000
        kwargs['block_size'] = 10000

        ## Set params specific to somatic or axonal/dendritic scans
        target = (SegmentationTask() & key).fetch1['compartment']
        if target == 'soma':
            kwargs['init_method'] = 'greedy_roi'
            kwargs['soma_radius'] = 7 / (ScanInfo() & key).microns_per_pixel # 7 microns
            kwargs['num_background_components'] = 4
            kwargs['init_on_patches'] = False
        else: # axons/dendrites
            kwargs['init_method'] = 'sparse_nmf'
            kwargs['snmf_alpha'] = 500  # 10^2 to 10^3.5 is a good range
            kwargs['num_background_components'] = 1
            kwargs['init_on_patches'] = True

        ## Set params specific to initialization on patches
        if kwargs['init_on_patches']:
            kwargs['patch_downsampling_factor'] = 4
            kwargs['percentage_of_patch_overlap'] = .2

        # Extract traces
        print('Extracting mask, traces and spikes (cnmf)...')
        cnmf_result = cmn.demix_and_deconvolve_with_cnmf(scan_, **kwargs)
        (location_matrix, activity_matrix, background_location_matrix,
         background_activity_matrix, raw_traces, spikes, AR_coeffs) = cnmf_result

        # Compute masks' coordinates
        print('Computing mask coordinates...')
        image_height, image_width, num_masks = location_matrix.shape
        px_center = [image_height / 2, image_width / 2]
        um_center = (ScanInfo() & key).fetch1['y', 'x']

        px_centroids = cmn.get_centroids(location_matrix)
        um_centroids = um_center + (px_centroids - px_center) * (ScanInfo() & key).microns_per_pixel
        um_z = (ScanInfo.Slice() & key).fetch1['z']

        # Insert CNMF results
        print('Inserting masks, background components, ar coefficients and traces...')

        ## Insert in NMF, Segmentation and Calcium
        self.insert1(key)
        Segmentation().insert1({**key, 'extract_method': 2})
        Calcium().insert1({**key, 'extract_method': 2}) # nmf also inserts traces

        ## Insert CNMF parameters
        lowercase_kwargs = {k.lower(): v for k, v in kwargs.items()}
        NMF.Parameters().insert1({**key, **lowercase_kwargs})

        ## Insert background components
        background_dict = {**key, 'masks': background_location_matrix,
                           'activity': background_activity_matrix}
        NMF.BackgroundComponents().insert1(background_dict)

        ## Insert masks and traces (masks in Matlab format)
        masks_as_F_ordered_vectors = location_matrix.reshape(-1, num_masks, order='F')
        masks = masks_as_F_ordered_vectors.T # [num_masks x num_pixels]
        for i, (mask, (px_y, px_x), (um_y, um_x), trace, ar_coeffs) in enumerate(
            zip(masks, px_centroids, um_centroids, raw_traces, AR_coeffs)):
            mask_key = {**key, 'extract_method': 2, 'mask_id': i + 1} # ids start at 1

            mask_pixels = np.where(mask)[0]
            mask_weights = mask[mask_pixels]
            mask_pixels += 1  # matlab indices start at 1
            Segmentation.Mask().insert1({**mask_key, 'pixels': mask_pixels,
                                         'weights': mask_weights})

            Segmentation.MaskInfo().insert1({**mask_key, 'type': target, 'px_x': px_x,
                                             'px_y': px_y, 'um_x': um_x, 'um_y': um_y,
                                             'um_z': um_z})

            Calcium.Trace().insert1({**mask_key, 'trace': trace})

            if kwargs['AR_order'] > 0:
               NMF.ARCoefficients().insert1({**mask_key, 'g': ar_coeffs})

        self.notify(key)

    def notify(self, key):
        fig = (Segmentation() & key).plot_contours()
        img_filename = '/tmp/' + key_hash(key) + '.png'
        fig.savefig(img_filename)
        plt.close(fig)

        msg = 'NMF for `{}` has been populated.'.format(key)
        (notify.SlackUser() & (experiment.Session() & key)).notify(msg, file=img_filename,
                                                                   file_title='mask contours')

    def save_video(self, filename='cnmf_results.mp4', start_index=0, seconds=30,
                   dpi=250, first_n=None):
        """ Creates an animation video showing the results of CNMF.

        :param string filename: Output filename (path + filename)
        :param int start_index: Where in the scan to start the video.
        :param int seconds: How long in seconds should the animation run.
        :param int dpi: Dots per inch, controls the quality of the video.
        :param int first_n: Draw only the first n components.

        :returns Figure. You can call show() on it.
        :rtype: matplotlib.figure.Figure
        """
        # Get fps and calculate total number of frames
        fps = (ScanInfo() & self).fetch1['fps']
        num_video_frames = int(round(fps * seconds))
        stop_index = start_index + num_video_frames

        # Load the scan
        channel = self.fetch1['channel'] - 1
        slice_id = self.fetch1['slice'] - 1
        scan_filename = (experiment.Scan() & self).local_filenames_as_wildcard
        scan = scanreader.read_scan(scan_filename, dtype=np.float32)
        scan_ = scan[slice_id, :, :, channel, start_index: stop_index]

        # Correct the scan
        correct_raster = RasterCorrection.get_correct_raster(self.fetch1())
        correct_motion = MotionCorrection.get_correct_motion(self.fetch1())
        scan_ = correct_motion(correct_raster(scan_), slice(start_index, stop_index))

        # Get scan dimensions
        image_height, image_width, _ = scan_.shape
        num_pixels = image_height * image_width

        # Get location and activity matrices
        location_matrix = (Segmentation() & self).get_all_masks()
        activity_matrix = (Calcium() & self).get_all_traces() # always there for CNMF
        background_location_matrix, background_activity_matrix = \
            (NMF.BackgroundComponents() & self).fetch1['masks', 'activity']

        # Select first n components
        if first_n is not None:
            location_matrix = location_matrix[:, :, :first_n]
            activity_matrix = activity_matrix[:first_n, :]

        # Drop frames that won't be displayed
        activity_matrix = activity_matrix[:, start_index: stop_index]
        background_activity_matrix = background_activity_matrix[:, start_index: stop_index]

        # Create movies
        extracted = np.dot(location_matrix.reshape(num_pixels, -1), activity_matrix)
        extracted = extracted.reshape(image_height, image_width, -1)
        background = np.dot(background_location_matrix.reshape(num_pixels, -1),
                            background_activity_matrix)
        background = background.reshape(image_height, image_width, -1)
        residual = scan_ - extracted - background

        # Create animation
        import matplotlib.animation as animation

        ## Set the figure
        fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)

        axes[0, 0].set_title('Original (Y)')
        im1 = axes[0, 0].imshow(scan_[:, :, 0], vmin=scan_.min(), vmax=scan_.max())  # just a placeholder
        fig.colorbar(im1, ax=axes[0, 0])

        axes[0, 1].set_title('Extracted (A*C)')
        im2 = axes[0, 1].imshow(extracted[:, :, 0], vmin=extracted.min(), vmax=extracted.max())
        fig.colorbar(im2, ax=axes[0, 1])

        axes[1, 0].set_title('Background (B*F)')
        im3 = axes[1, 0].imshow(background[:, :, 0], vmin=background.min(),
                                vmax=background.max())
        fig.colorbar(im3, ax=axes[1, 0])

        axes[1, 1].set_title('Residual (Y - A*C - B*F)')
        im4 = axes[1, 1].imshow(residual[:, :, 0], vmin=residual.min(), vmax=residual.max())
        fig.colorbar(im4, ax=axes[1, 1])

        for ax in axes.ravel():
            ax.axis('off')

        ## Make the animation
        def update_img(i):
            im1.set_data(scan_[:, :, i])
            im2.set_data(extracted[:, :, i])
            im3.set_data(background[:, :, i])
            im4.set_data(residual[:, :, i])

        video = animation.FuncAnimation(fig, update_img, scan_.shape[2],
                                        interval=1000 / fps)

        # Save animation
        if not filename.endswith('.mp4'):
            filename += '.mp4'
        print('Saving video at:', filename)
        print('If this takes too long, stop it and call again with dpi <', dpi, '(default)')
        video.save(filename, dpi=dpi)

        return fig

    #TODO: Move this with ARCoefficients to Activity
    def plot_impulse_responses(self, num_timepoints=100):
        """ Plots the impulse response functions for all traces.

        :param int num_timepoints: The number of points after impulse to use for plotting.

        :returns Figure. You can call show() on it.
        :rtype: matplotlib.figure.Figure
        """
        ar_rel = NMF.ARCoefficients() & self
        if ar_rel: # if an AR model was used
            # Get some params
            fps = (ScanInfo() & self).fetch1['fps']
            ar_coeffs = ar_rel.fetch['g']

            # Define the figure
            fig = plt.figure()
            x_axis = np.arange(num_timepoints) / fps  # make it seconds

            # Over each trace
            for g in ar_coeffs:
                AR_order = len(g)

                # Calculate impulse response function
                irf = np.zeros(num_timepoints)
                irf[0] = 1  # initial spike
                for i in range(1, num_timepoints):
                    if i <= AR_order:  # start of the array needs special care
                        irf[i] = np.sum(g[:i] * irf[i - 1:: -1])
                    else:
                        irf[i] = np.sum(g * irf[i - 1: i - AR_order - 1: -1])

                # Plot
                plt.plot(x_axis, irf)

            return fig

    def delete(self):
        """ Override delete to also delete tuple from Segmentation. """
        if Segmentation() & self:
            dj.BaseRelation.delete(Segmentation() & self)
        if self:
            super().delete()


@schema
class Segmentation(dj.Manual):
    definition = """ # Group of different segmentations.

    -> experiment.Scan
    -> shared.Slice
    -> shared.Channel
    -> shared.SegmentationMethod
    """

    class Mask(dj.Part):
        definition = """ # mask produced by segmentation.

        -> Segmentation
        mask_id             : smallint
        ---
        pixels              : longblob      # indices into the image in column major (Fortran) order
        weights = null      : longblob      # weights of the mask at the indices above
        """
        def get_mask_as_image(self):
            """ Return this mask as an image (2-d numpy array)."""
            # Get params
            pixel_indices, weights = self.fetch['pixels', 'weights']
            image_height, image_width = (ScanInfo() & self).fetch1['px_height', 'px_width']

            # Reshape mask
            mask = Segmentation.reshape_masks(pixel_indices, weights, image_height, image_width)

            return np.squeeze(mask)

    class MaskInfo(dj.Part):
        definition = """ # mask type and coordinates in x, y, z

        -> Segmentation
        -> Segmentation.Mask
        ---
        -> shared.MaskType                  # type of the mask assigned during segmentation
        px_x                :smallint       # x-coordinate of centroid in the frame
        px_y                :smallint       # y-coordinate of centroid in the frame
        um_x                :smallint       # x-coordinate of centroid in motor coordinate system
        um_y                :smallint       # y-coordinate of centroid in motor coordinate system
        um_z                :smallint       # z-coordinate of mask relative to surface of the cortex
        """

    @staticmethod
    def reshape_masks(mask_pixels, mask_weights, image_height, image_width):
        """ Reshape masks into an image_height x image_width x num_masks array."""
        masks = np.zeros([image_height, image_width, len(mask_pixels)])

        # Reshape each mask
        for i, (mp, mw) in enumerate(zip(mask_pixels, mask_weights)):
            mask_as_vector = np.zeros(image_height * image_width)
            mask_as_vector[np.squeeze(mp - 1).astype(int)] = np.squeeze(mw)
            masks[:, :, i] = mask_as_vector.reshape(image_height, image_width, order='F')

        return masks

    def get_all_masks(self):
        """Returns an image_height x image_width x num_masks matrix with all masks."""
        mask_rel = (Segmentation.Mask() & self)

        # Get masks
        image_height, image_width = (ScanInfo() & self).fetch1['px_height', 'px_width']
        mask_pixels, mask_weights = mask_rel.fetch.order_by('mask_id')['pixels', 'weights']

        # Reshape masks
        location_matrix = Segmentation.reshape_masks(mask_pixels, mask_weights,
                                                     image_height, image_width)

        return location_matrix

    def plot_contours(self, first_n=None):
        """ Draw contours of masks over the correlation image.

        :param first_n: Number of masks to plot. None for all.
        :returns: None
        """
        from .utils import caiman_interface as cmn

        # Get location matrix
        location_matrix = self.get_all_masks()

        # Select first n components
        if first_n is not None:
            location_matrix = location_matrix[:, :, :first_n]

        # Get correlation image if defined, black background otherwise.
        image_rel = SummaryImages() & self
        if image_rel:
            background_image = image_rel.fetch1['correlation']
        else:
            background_image = np.zeros(location_matrix.shape[:2])

        # Draw contours
        figsize = 7 * (background_image.shape / np.min(background_image.shape))
        fig = plt.figure(figsize=figsize)
        cmn.plot_contours(location_matrix, background_image)

        return fig

    def plot_centroids(self, first_n=None):
        """ Draw centroids of masks over the correlation image.

        :param first_n: Number of masks to plot. None for all.
        :returns: None
        """
        # Get centroids
        centroids = (Segmentation.MaskInfo() & self).fetch.order_by('mask_id')['px_x', 'px_y']

        # Select first n components
        if first_n is not None:
            centroids[0] = centroids[0][:first_n]
            centroids[1] = centroids[1][:first_n]

        # Get correlation image if defined, black background otherwise.
        image_rel = SummaryImages() & self
        if image_rel:
            background_image = image_rel.fetch1['correlation']
        else:
            image_height, image_width = (ScanInfo() & self).fetch1['px_height', 'px_width']
            background_image = np.zeros([image_height, image_width])

        # Plot centroids
        figsize = 7 * (background_image.shape / np.min(background_image.shape))
        fig = plt.figure(figsize=figsize)
        plt.imshow(background_image)
        plt.plot(centroids[0], centroids[1], 'ow', markersize=3)

        return fig

    def plot_centroids_3d(self):
        #TODO: Add different colors for different types, correlation image as 2-d planes
        from mpl_toolkits.mplot3d import Axes3D
        centroids = (Segmentation.MaskInfo() & self).fetch.order_by('mask_id')['um_x', 'um_y', 'um_z']

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(centroids[0], centroids[1], centroids[2])
        ax.set_xlabel('x (um)')
        ax.set_ylabel('y (um)')
        ax.set_zlabel('z (um)')

        return fig

    def delete(self):
        """ Delete entries in appropiate subtables."""
        if ManualSegmentation() & self:
            dj.BaseRelation.delete(ManualSegmentation() & self)
        if NMF() & self:
            dj.BaseRelation.delete(NMF() & self)
        if self:
            super().delete()


@schema
class Calcium(dj.Computed):
    definition = """  # calcium traces before spike extraction or filtering

    -> Segmentation
    """

    class Trace(dj.Part):
        definition = """

        -> Calcium
        -> Segmentation.Mask
        ---
        trace                   : longblob
        """

    def _make_tuples(self, key):
        print('Creating calcium traces for', key)

        # Load scan
        slice_id = key['slice'] - 1
        channel = key['channel'] - 1
        scan_filename = (experiment.Scan() & key).local_filenames_as_wildcard
        scan = scanreader.read_scan(scan_filename, dtype=np.float32)
        scan_ = scan[slice_id, :, :, channel, :]

        # Get masks as images
        mask_ids, mask_pixels, mask_weights = \
            (Segmentation.Mask() & key).fetch['mask_id', 'pixels', 'weights']
        location_matrix = Segmentation.reshape_masks(mask_pixels, mask_weights,
                                                     scan.image_height, scan.image_width)
        masks = location_matrix.transpose([2, 0, 1])

        self.insert1(key)
        for mask_id, mask in zip(mask_ids, masks):
            trace = np.average(scan_.reshape(-1, scan.num_frames), weights=mask.ravel(),
                               axis=0)

            Calcium.Trace().insert1({**key, 'mask_id': mask_id, 'trace': trace})

        self.notify(key)

    def notify(self, key):
        fig = plt.figure()
        plt.plot((Calcium() & key).get_all_traces().T)
        img_filename = '/tmp/' + key_hash(key) + '.png'
        fig.savefig(img_filename)
        plt.close(fig)

        msg = 'Calcium.Trace for `{}` has been populated.'.format(key)
        (notify.SlackUser() & (experiment.Session() & key)).notify(msg, file=img_filename,
                                                                   file_title='calcium traces')

    def get_all_traces(self):
        """ Returns a num_traces x num_timesteps matrix with all traces."""
        traces = (Calcium.Trace() & self).fetch.order_by('mask_id')['trace']
        return np.array([x.squeeze() for x in traces])


@schema
class MaskClassification(dj.Computed):
    definition = """ # automatic classification of segmented masks.

    -> Segmentation
    """
    @property
    def key_source(self):
        return Segmentation() & {'extract_method': '2'} # only for cnmf extraction

    class Type(dj.Part):
        definition = """

        -> MaskClassification
        -> Segmentation.Mask
        ---
        -> shared.MaskType
        """

    def _make_tuples(self, key):
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Get masks as images
        mask_ids, pixels, weights = (Segmentation.Mask() & key).fetch['mask_id', 'pixels', 'weights']
        image_height, image_width = (ScanInfo() & key).fetch1['px_height', 'px_width']
        location_matrix = Segmentation.reshape_masks(pixels, weights, image_height, image_width)
        masks = location_matrix.transpose([2, 0, 1])

        # Get template
        if SummaryImages() & key:
            template = (SummaryImages() & key).fetch1['correlation']
        else:
            raise PipelineException('Manual classification of masks requires SummaryImages.')

        self.insert1(key)
        for mask_id, mask in zip(mask_ids, masks):
            ir = mask.sum(axis=1) > 0
            ic = mask.sum(axis=0) > 0

            il, jl = [max(np.min(np.where(i)[0]) - 10, 0) for i in [ir, ic]]
            ih, jh = [min(np.max(np.where(i)[0]) + 10, len(i)) for i in [ir, ic]]
            tmp_mask = np.array(mask[il:ih, jl:jh])

            with sns.axes_style('white'):
                fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(20, 5))

            ax[0].imshow(template[il:ih, jl:jh], cmap=plt.cm.get_cmap('gray'))
            ax[1].imshow(template[il:ih, jl:jh], cmap=plt.cm.get_cmap('gray'))
            tmp_mask[tmp_mask == 0] = np.NaN
            ax[1].matshow(tmp_mask, cmap=plt.cm.get_cmap('viridis'), alpha=0.5, zorder=10)
            ax[2].matshow(tmp_mask, cmap=plt.cm.get_cmap('viridis'))
            for a in ax:
                a.set_aspect(1)
                a.axis('off')
            fig.tight_layout()
            fig.canvas.manager.window.wm_geometry("+250+250")
            fig.suptitle('S(o)ma, A(x)on, (D)endrite, (N)europil, (A)rtifact or (U)nknown?')

            def on_button(event):
                if event.key == 'o':
                    self.Type().insert1({**key, 'mask_id': mask_id, 'type': 'soma'})
                    print('Soma', {**key, 'mask_id': mask_id})
                    plt.close(fig)
                elif event.key == 'x':
                    self.Type().insert1({**key, 'mask_id': mask_id, 'type': 'axon'})
                    print('Axon', {**key, 'mask_id': mask_id})
                    plt.close(fig)
                elif event.key == 'd':
                    self.Type().insert1({**key, 'mask_id': mask_id, 'type': 'dendrite'})
                    print('Dendrite', {**key, 'mask_id': mask_id})
                    plt.close(fig)
                elif event.key == 'n':
                    self.Type().insert1({**key, 'mask_id': mask_id, 'type': 'neuropil'})
                    print('Neuropil', {**key, 'mask_id': mask_id})
                    plt.close(fig)
                elif event.key == 'a':
                    self.Type().insert1({**key, 'mask_id': mask_id, 'type': 'artifact'})
                    print('Artifact', {**key, 'mask_id': mask_id})
                    plt.close(fig)
                elif event.key == 'u':
                    self.Type().insert1({**key, 'mask_id': mask_id, 'type': 'unknown'})
                    print('Unknown', {**key, 'mask_id': mask_id})
                    plt.close(fig)

            fig.canvas.mpl_connect('key_press_event', on_button)

            plt.show()


@schema
class ScanSet(dj.Computed):
    definition = """ # union of all masks in the same scan

    -> Segmentation         # processing done per slice
    """

    class Unit(dj.Part):
        definition = """ # single unit in the scan
        -> ScanInfo
        -> shared.SegmentationMethod
        unit_id                 :int                # unique per scan & segmentation method
        ---
        -> Segmentation.Mask
        """

    #class Match(dj.Part) # MaskSet?
    #    definition = """ # unit-mask pairs per scan
    #    -> ScanSet.Unit
    #    -> Segmentation.Mask
    #    """

    def _make_tuples(self, key):
        # Insert in ScanSet
        self.insert1(key)

        # Get next unit_id for Scan (& SegmentationMethod)
        scan_key = {k:v for k, v in key.items() if k in ['animal_id', 'session',
                                                         'scan_idx', 'extract_method']}
        unit_rel = ScanSet.Unit() & scan_key
        unit_id = np.max(unit_rel.fetch['unit_id']) + 1 if unit_rel else 1

        # Insert pairs in ScanSet.Unit
        mask_ids = (Segmentation.Mask() & key).fetch['mask_id']
        for mask_id in mask_ids:
            ScanSet.Unit().insert1({**key, 'mask_id': mask_id, 'unit_id': unit_id})
            unit_id += 1

    def delete(self):
        """ Propagate deletion to units in the slice."""
        if ScanSet.Unit() & self:
            (ScanSet.Unit() & self).delete()
        if self:
            super().delete()


@schema
class Activity(dj.Computed):
    definition = """ # deconvolved calcium activity inferred from calcium traces

    -> ScanSet              # processing done per slice
    -> shared.SpikeMethod
    """

    @property
    def key_source(self):
        return ScanSet() * (shared.SpikeMethod() & {'language': 'python'})

    class Trace(dj.Part):
        definition = """ # deconvolved calcium acitivity

        -> ScanSet.Unit
        -> shared.SpikeMethod
        ---
        trace                   : longblob
        """

    def _make_tuples(self, key):
        print('Creating activity traces for', key)

        # Get params
        fps = (ScanInfo() & key).fetch1['fps']
        unit_ids, traces = ((ScanSet.Unit() & key) * Calcium.Trace()).fetch['unit_id', 'trace']
        full_traces = [signal.fill_nans(np.squeeze(trace).copy()) for trace in traces]

        # Insert in Activity
        self.insert1(key)

        # Get scan key (used to insert in Activity.Trace)
        scan_key = {k:v for k, v in key.items() if k in ['animal_id', 'session',
                                                         'scan_idx', 'extract_method',
                                                         'spike_method']}

        method_name = (shared.SpikeMethod() & key).fetch1['spike_method_name']
        if method_name == 'oopsi': # Non-negative sparse deconvolution
            import pyfnnd # Install from https://github.com/cajal/PyFNND.git

            for unit_id, trace in zip(unit_ids, full_traces):
                spike_trace = pyfnnd.deconvolve(trace, dt=1/fps)[0]
                Activity.Trace().insert1({**scan_key, 'unit_id': unit_id, 'trace': spike_trace})

        elif method_name == 'stm': # Spike-triggered mixture model
            import c2s # Install from https://github.com/lucastheis/c2s

            for unit_id, trace in zip(unit_ids, full_traces):
                start = signal.notnan(trace)
                end = signal.notnan(trace, len(trace) - 1, increment=-1)
                trace_dict = {'calcium': np.atleast_2d(trace[start:end + 1]), 'fps': fps}

                data = c2s.predict(c2s.preprocess([trace_dict], fps=fps), verbosity=0)
                spike_trace = np.squeeze(data[0].pop('predictions'))

                Activity.Trace().insert1({**scan_key, 'unit_id': unit_id, 'trace': spike_trace})

        elif method_name == 'nmf': # Noise-constrained deconvolution
            from pipeline.utils import caiman_interface as cmn

            #for unit_id, trace in zip(unit_ids, full_traces):
            #    spike_trace = cmn.deconvolve(trace, fps)
            #    Activity.Trace().insert1({**scan_key, 'unit_id': unit_id, 'trace': spike_trace})
            raise NotImplementedError('NMF not yet implemented')

        else:
            raise NotImplementedError('{} method not implemented.'.format(method_name))

        self.notify(key)

    def notify(self, key):
        fig = plt.figure()
        plt.plot((Activity() & key).get_all_spikes().T)
        img_filename = '/tmp/' + key_hash(key) + '.png'
        fig.savefig(img_filename)
        plt.close(fig)

        msg = 'Activity.Trace for `{}` has been populated.'.format(key)
        (notify.SlackUser() & (experiment.Session() & key)).notify(msg, file=img_filename,
                                                                   file_title='spike traces')

    def get_all_spikes(self):
        """ Returns a num_traces x num_timesteps matrix with all spikes."""
        spikes = (Activity.Trace() & self).fetch.order_by('unit_id')['trace']
        return np.array([x.squeeze() for x in spikes])

    def delete(self):
        """ Propagate deletion to traces in the slice. """
        if Activity.Trace() & self:
            (Activity.Trace() & self).delete()
        if self:
            super().delete()


schema.spawn_missing_classes()