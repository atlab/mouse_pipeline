""" Schemas for mesoscope scans."""
import datajoint as dj
from datajoint.jobs import key_hash
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanreader

from . import experiment, injection, notify, shared
from .utils import galvo_corrections, signal, quality, mask_classification, performance
from .exceptions import PipelineException

import logging

logger = logging.getLogger(__name__)


schema = dj.schema("pipeline_meso", locals(), create_tables=True)
CURRENT_VERSION = 1


@schema
class Version(dj.Manual):
    definition = """ # versions for the meso pipeline

    -> shared.PipelineVersion
    ---
    description = ''                : varchar(256)      # any notes on this version
    date = CURRENT_TIMESTAMP        : timestamp         # automatic
    """


@schema
class ScanInfo(dj.Imported):
    definition = """ # general data about mesoscope scans

    -> experiment.Scan
    -> Version                                  # meso version
    ---
    nfields                 : tinyint           # number of fields
    nchannels               : tinyint           # number of channels
    nframes                 : int               # number of recorded frames
    nframes_requested       : int               # number of requested frames (from header)
    x                       : float             # (um) ScanImage's 0 point in the motor coordinate system
    y                       : float             # (um) ScanImage's 0 point in the motor coordinate system
    fps                     : float             # (Hz) frames per second
    bidirectional           : boolean           # true = bidirectional scanning
    usecs_per_line          : float             # microseconds per scan line
    fill_fraction           : float             # raster scan temporal fill fraction (see scanimage)
    nrois                   : tinyint           # number of ROIs (see scanimage)
    """

    @property
    def key_source(self):
        rigs = [{"rig": "2P4"}, {"rig": "R2P1"}]
        meso_scans = experiment.Scan() & (experiment.Session() & rigs)
        return meso_scans * (Version() & {"pipe_version": CURRENT_VERSION})

    class Field(dj.Part):
        definition = """ # field-specific scan information

        -> ScanInfo
        -> shared.Field
        ---
        px_height           : smallint      # height in pixels
        px_width            : smallint      # width in pixels
        um_height           : float         # height in microns
        um_width            : float         # width in microns
        x                   : float         # (um) center of field in the motor coordinate system
        y                   : float         # (um) center of field in the motor coordinate system
        z                   : float         # (um) absolute depth with respect to the surface of the cortex
        delay_image         : longblob      # (ms) delay between the start of the scan and pixels in this field
        roi                 : tinyint       # ROI to which this field belongs
        valid_depth=false   : boolean       # whether depth has been manually check
        """

        def make(self, key, scan, field_id):
            # Create results tuple
            tuple_ = key.copy()
            tuple_["field"] = field_id + 1

            # Get attributes
            (
                x_zero,
                y_zero,
                _,
            ) = scan.motor_position_at_zero  # motor x, y at ScanImage's 0
            surf_z = (experiment.Scan() & key).fetch1(
                "depth"
            )  # surface depth in fastZ coordinates
            tuple_["px_height"] = scan.field_heights[field_id]
            tuple_["px_width"] = scan.field_widths[field_id]
            tuple_["um_height"] = scan.field_heights_in_microns[field_id]
            tuple_["um_width"] = scan.field_widths_in_microns[field_id]
            tuple_["x"] = x_zero + scan._degrees_to_microns(scan.fields[field_id].x)
            tuple_["y"] = y_zero + scan._degrees_to_microns(scan.fields[field_id].y)
            tuple_["z"] = scan.field_depths[field_id] - surf_z  # fastZ only
            tuple_["delay_image"] = scan.field_offsets[field_id]
            tuple_["roi"] = scan.field_rois[field_id][0]

            # Insert
            self.insert1(tuple_)

        @property
        def microns_per_pixel(self):
            """ Returns an array with microns per pixel in height and width. """
            um_height, px_height, um_width, px_width = self.fetch1(
                "um_height", "px_height", "um_width", "px_width"
            )
            return np.array([um_height / px_height, um_width / px_width])

    def make(self, key):
        """ Read and store some scan parameters."""
        # Read the scan
        print("Reading header...")
        scan_filename = (experiment.Scan() & key).local_filenames_as_wildcard
        scan = scanreader.read_scan(scan_filename)

        # Get attributes
        tuple_ = key.copy()  # in case key is reused somewhere else
        tuple_["nfields"] = scan.num_fields
        tuple_["nchannels"] = scan.num_channels
        tuple_["nframes"] = scan.num_frames
        tuple_["nframes_requested"] = scan.num_requested_frames
        tuple_["x"] = scan.motor_position_at_zero[0]
        tuple_["y"] = scan.motor_position_at_zero[1]
        tuple_["fps"] = scan.fps
        tuple_["bidirectional"] = scan.is_bidirectional
        tuple_["usecs_per_line"] = scan.seconds_per_line * 1e6
        tuple_["fill_fraction"] = scan.temporal_fill_fraction
        tuple_["nrois"] = scan.num_rois
        tuple_["valid_depth"] = True

        # Insert in ScanInfo
        self.insert1(tuple_)

        # Insert field information
        for field_id in range(scan.num_fields):
            ScanInfo.Field().make(key, scan, field_id)

        # Fill in CorrectionChannel if only one channel
        if scan.num_channels == 1:
            CorrectionChannel().fill(key)

        # Fill SegmentationTask if scan in autosegment
        if experiment.AutoProcessing() & key & {"autosegment": True}:
            SegmentationTask().fill(key)


@schema
class FieldAnnotation(dj.Manual):
    definition = """ # Annotations for specific fields within one scan
    -> ScanInfo.Field
    -> shared.ExpressionConstruct
    -> shared.Channel
    ---
    -> [nullable] injection.InjectionSite
    field_notes                   : varchar(256)
    """


@schema
class Quality(dj.Computed):
    definition = """ # different quality metrics for a scan (before corrections)

    -> ScanInfo
    """

    @property
    def key_source(self):
        return ScanInfo() & {"pipe_version": CURRENT_VERSION}

    class MeanIntensity(dj.Part):
        definition = """ # mean intensity values across time

        -> Quality
        -> shared.Field
        -> shared.Channel
        ---
        intensities                 : longblob
        """

    class SummaryFrames(dj.Part):
        definition = """ # 16-part summary of the scan (mean of 16 blocks)

        -> Quality
        -> shared.Field
        -> shared.Channel
        ---
        summary                     : longblob      # h x w x 16
        """

    class Contrast(dj.Part):
        definition = """ # difference between 99 and 1 percentile across time

        -> Quality
        -> shared.Field
        -> shared.Channel
        ---
        contrasts                   : longblob
        """

    class QuantalSize(dj.Part):
        definition = """ # quantal size in images

        -> Quality
        -> shared.Field
        -> shared.Channel
        ---
        min_intensity               : int           # min value in movie
        max_intensity               : int           # max value in movie
        quantal_size                : float         # variance slope, corresponds to quantal size
        zero_level                  : int           # level corresponding to zero (computed from variance dependence)
        quantal_frame               : longblob      # average frame expressed in quanta
        """

    class EpileptiformEvents(dj.Part):
        definition = """ # compute frequency of epileptiform events

        -> Quality
        -> shared.Field
        -> shared.Channel
        ---
        frequency       : float         # (events / sec) frequency of epileptiform events
        abn_indices     : longblob      # indices of epileptiform events (0-based)
        peak_indices    : longblob      # indices of all local maxima peaks (0-based)
        prominences     : longblob      # peak prominence for all peaks
        widths          : longblob      # (secs) width at half prominence for all peaks
        """

    def make(self, key):
        # Read the scan
        scan_filename = (experiment.Scan() & key).local_filenames_as_wildcard
        scan = scanreader.read_scan(scan_filename)

        # Insert in Quality
        self.insert1(key)

        for field_id in range(scan.num_fields):
            print("Computing quality metrics for field", field_id + 1)
            for channel in range(scan.num_channels):
                # Map: Compute quality metrics in parallel
                results = performance.map_frames(
                    performance.parallel_quality_metrics,
                    scan,
                    field_id=field_id,
                    channel=channel,
                )

                # Reduce
                mean_intensities = np.zeros(scan.num_frames)
                contrasts = np.zeros(scan.num_frames)
                for frames, chunk_mis, chunk_contrasts, _ in results:
                    mean_intensities[frames] = chunk_mis
                    contrasts[frames] = chunk_contrasts
                sorted_results = sorted(results, key=lambda res: res[0])
                mean_groups = np.array_split(
                    [r[3] for r in sorted_results], 16
                )  # 16 groups
                frames = np.stack(
                    [np.mean(g, axis=0) for g in mean_groups if g.any()], axis=-1
                )

                # Compute quantal size
                middle_frame = int(np.floor(scan.num_frames / 2))
                mini_scan = scan[
                    field_id,
                    :,
                    :,
                    channel,
                    max(middle_frame - 2000, 0) : middle_frame + 2000,
                ]
                mini_scan = mini_scan.astype(np.float32)
                results = quality.compute_quantal_size(mini_scan)
                min_intensity, max_intensity, _, _, quantal_size, zero_level = results
                quantal_frame = (
                    np.mean(mini_scan, axis=-1) - zero_level
                ) / quantal_size

                # Compute abnormal event frequency
                deviations = (
                    mean_intensities - mean_intensities.mean()
                ) / mean_intensities.mean()
                peaks, prominences, widths = quality.find_peaks(deviations)
                widths = [w / scan.fps for w in widths]  # in seconds
                abnormal = peaks[
                    [p > 0.2 and w < 0.4 for p, w in zip(prominences, widths)]
                ]
                abnormal_freq = len(abnormal) / (scan.num_frames / scan.fps)

                # Insert
                field_key = {**key, "field": field_id + 1, "channel": channel + 1}
                self.MeanIntensity().insert1(
                    {**field_key, "intensities": mean_intensities}
                )
                self.Contrast().insert1({**field_key, "contrasts": contrasts})
                self.SummaryFrames().insert1({**field_key, "summary": frames})
                self.QuantalSize().insert1(
                    {
                        **field_key,
                        "min_intensity": min_intensity,
                        "max_intensity": max_intensity,
                        "quantal_size": quantal_size,
                        "zero_level": zero_level,
                        "quantal_frame": quantal_frame,
                    }
                )
                self.EpileptiformEvents.insert1(
                    {
                        **field_key,
                        "frequency": abnormal_freq,
                        "abn_indices": abnormal,
                        "peak_indices": peaks,
                        "prominences": prominences,
                        "widths": widths,
                    }
                )

                self.notify(field_key, frames, mean_intensities, contrasts)

    @notify.ignore_exceptions
    def notify(self, key, summary_frames, mean_intensities, contrasts):
        # Send summary frames
        import imageio

        video_filename = "/tmp/" + key_hash(key) + ".gif"
        percentile_99th = np.percentile(summary_frames, 99.5)
        summary_frames = np.clip(summary_frames, None, percentile_99th)
        summary_frames = signal.float2uint8(summary_frames).transpose([2, 0, 1])
        imageio.mimsave(video_filename, summary_frames, duration=0.4)

        msg = (
            "summary frames for {animal_id}-{session}-{scan_idx} field {field} "
            "channel {channel}"
        ).format(**key)
        slack_user = notify.SlackUser() & (experiment.Session() & key)
        slack_user.notify(file=video_filename, file_title=msg)

        # Send intensity and contrasts
        fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
        axes[0].set_title("Mean intensity", size="small")
        axes[0].plot(mean_intensities)
        axes[0].set_ylabel("Pixel intensities")
        axes[1].set_title("Contrast (99 - 1 percentile)", size="small")
        axes[1].plot(contrasts)
        axes[1].set_xlabel("Frames")
        axes[1].set_ylabel("Pixel intensities")
        img_filename = "/tmp/" + key_hash(key) + ".png"
        fig.savefig(img_filename, bbox_inches="tight")
        plt.close(fig)

        msg = (
            "quality traces for {animal_id}-{session}-{scan_idx} field {field} "
            "channel {channel}"
        ).format(**key)
        slack_user.notify(file=img_filename, file_title=msg)


@schema
class CorrectionChannel(dj.Manual):
    definition = """ # channel to use for raster and motion correction

    -> experiment.Scan
    -> shared.Field
    ---
    -> shared.Channel
    """

    def fill(self, key, channel=1):
        for field_key in (ScanInfo.Field() & key).fetch(dj.key):
            self.insert1(
                {**field_key, "channel": channel},
                ignore_extra_fields=True,
                skip_duplicates=True,
            )


@schema
class RasterCorrection(dj.Computed):
    definition = """ # raster correction for bidirectional resonant scans

    -> ScanInfo                         # animal_id, session, scan_idx, version
    -> CorrectionChannel                # animal_id, session, scan_idx, field
    ---
    raster_template     : longblob      # average frame from the middle of the movie
    raster_phase        : float         # difference between expected and recorded scan angle
    """

    @property
    def key_source(self):
        return ScanInfo * CorrectionChannel & {"pipe_version": CURRENT_VERSION}

    def make(self, key):
        from scipy.signal import tukey

        # Read the scan
        scan_filename = (experiment.Scan() & key).local_filenames_as_wildcard
        scan = scanreader.read_scan(scan_filename, dtype=np.float32)

        # Select correction channel
        channel = (CorrectionChannel() & key).fetch1("channel") - 1
        field_id = key["field"] - 1

        # Load some frames from the middle of the scan
        middle_frame = int(np.floor(scan.num_frames / 2))
        frames = slice(max(middle_frame - 1000, 0), middle_frame + 1000)
        mini_scan = scan[field_id, :, :, channel, frames]

        # Create results tuple
        tuple_ = key.copy()

        # Create template (average frame tapered to avoid edge artifacts)
        taper = np.sqrt(
            np.outer(
                tukey(scan.field_heights[field_id], 0.4),
                tukey(scan.field_widths[field_id], 0.4),
            )
        )
        anscombed = 2 * np.sqrt(
            mini_scan - mini_scan.min() + 3 / 8
        )  # anscombe transform
        template = np.mean(anscombed, axis=-1) * taper
        tuple_["raster_template"] = template

        # Compute raster correction parameters
        if scan.is_bidirectional:
            tuple_["raster_phase"] = galvo_corrections.compute_raster_phase(
                template, scan.temporal_fill_fraction
            )
        else:
            tuple_["raster_phase"] = 0

        # Insert
        self.insert1(tuple_)

    def get_correct_raster(self):
        """ Returns a function to perform raster correction on the scan. """
        raster_phase = self.fetch1("raster_phase")
        fill_fraction = (ScanInfo() & self).fetch1("fill_fraction")
        if abs(raster_phase) < 1e-7:
            correct_raster = lambda scan: scan.astype(np.float32, copy=False)
        else:
            correct_raster = lambda scan: galvo_corrections.correct_raster(
                scan, raster_phase, fill_fraction
            )
        return correct_raster


@schema
class MotionCorrection(dj.Computed):
    definition = """ # motion correction for galvo scans

    -> RasterCorrection
    ---
    motion_template                 : longblob      # image used as alignment template
    y_shifts                        : longblob      # (pixels) y motion correction shifts
    x_shifts                        : longblob      # (pixels) x motion correction shifts
    y_std                           : float         # (pixels) standard deviation of y shifts
    x_std                           : float         # (pixels) standard deviation of x shifts
    outlier_frames                  : longblob      # mask with true for frames with outlier shifts (already corrected)
    align_time=CURRENT_TIMESTAMP    : timestamp     # automatic
    """

    @property
    def key_source(self):
        return RasterCorrection() & {"pipe_version": CURRENT_VERSION}

    def make(self, key):
        """Computes the motion shifts per frame needed to correct the scan."""
        from scipy import ndimage

        # Read the scan
        scan_filename = (experiment.Scan() & key).local_filenames_as_wildcard
        scan = scanreader.read_scan(scan_filename)

        # Get some params
        px_height, px_width = (ScanInfo.Field() & key).fetch1("px_height", "px_width")
        channel = (CorrectionChannel() & key).fetch1("channel") - 1
        field_id = key["field"] - 1

        # Load some frames from middle of scan to compute template
        skip_rows = int(
            round(px_height * 0.10)
        )  # we discard some rows/cols to avoid edge artifacts
        skip_cols = int(round(px_width * 0.10))
        middle_frame = int(np.floor(scan.num_frames / 2))
        mini_scan = scan[
            field_id,
            skip_rows:-skip_rows,
            skip_cols:-skip_cols,
            channel,
            max(middle_frame - 1000, 0) : middle_frame + 1000,
        ]
        mini_scan = mini_scan.astype(np.float32, copy=False)

        # Correct mini scan
        correct_raster = (RasterCorrection() & key).get_correct_raster()
        mini_scan = correct_raster(mini_scan)

        # Create template
        mini_scan = 2 * np.sqrt(mini_scan - mini_scan.min() + 3 / 8)  # *
        template = np.mean(mini_scan, axis=-1)
        template = ndimage.gaussian_filter(template, 0.7)  # **
        # * Anscombe tranform to normalize noise, increase contrast and decrease outliers' leverage
        # ** Small amount of gaussian smoothing to get rid of high frequency noise

        # Map: compute motion shifts in parallel
        f = performance.parallel_motion_shifts  # function to map
        raster_phase = (RasterCorrection() & key).fetch1("raster_phase")
        fill_fraction = (ScanInfo() & key).fetch1("fill_fraction")
        kwargs = {
            "raster_phase": raster_phase,
            "fill_fraction": fill_fraction,
            "template": template,
        }
        results = performance.map_frames(
            f,
            scan,
            field_id=field_id,
            y=slice(skip_rows, -skip_rows),
            x=slice(skip_cols, -skip_cols),
            channel=channel,
            kwargs=kwargs,
        )

        # Reduce
        y_shifts = np.zeros(scan.num_frames)
        x_shifts = np.zeros(scan.num_frames)
        for frames, chunk_y_shifts, chunk_x_shifts in results:
            y_shifts[frames] = chunk_y_shifts
            x_shifts[frames] = chunk_x_shifts

        # Detect outliers
        max_y_shift, max_x_shift = 20 / (ScanInfo.Field() & key).microns_per_pixel
        y_shifts, x_shifts, outliers = galvo_corrections.fix_outliers(
            y_shifts, x_shifts, max_y_shift, max_x_shift
        )

        # Center shifts around zero
        y_shifts -= np.median(y_shifts)
        x_shifts -= np.median(x_shifts)

        # Create results tuple
        tuple_ = key.copy()
        tuple_["motion_template"] = template
        tuple_["y_shifts"] = y_shifts
        tuple_["x_shifts"] = x_shifts
        tuple_["outlier_frames"] = outliers
        tuple_["y_std"] = np.std(y_shifts)
        tuple_["x_std"] = np.std(x_shifts)

        # Insert
        self.insert1(tuple_)

        # Notify after all fields have been processed
        scan_key = {
            "animal_id": key["animal_id"],
            "session": key["session"],
            "scan_idx": key["scan_idx"],
            "pipe_version": key["pipe_version"],
        }
        if len(MotionCorrection - CorrectionChannel & scan_key) > 0:
            self.notify(scan_key, scan.num_frames, scan.num_fields)

    @notify.ignore_exceptions
    def notify(self, key, num_frames, num_fields):
        fps = (ScanInfo() & key).fetch1("fps")
        seconds = np.arange(num_frames) / fps

        fig, axes = plt.subplots(
            num_fields, 1, figsize=(15, 4 * num_fields), sharey=True
        )
        axes = [axes] if num_fields == 1 else axes  # make list if single axis object
        for i in range(num_fields):
            y_shifts, x_shifts = (self & key & {"field": i + 1}).fetch1(
                "y_shifts", "x_shifts"
            )
            axes[i].set_title("Shifts for field {}".format(i + 1))
            axes[i].plot(seconds, y_shifts, label="y shifts")
            axes[i].plot(seconds, x_shifts, label="x shifts")
            axes[i].set_ylabel("Pixels")
            axes[i].set_xlabel("Seconds")
            axes[i].legend()
        fig.tight_layout()
        img_filename = "/tmp/" + key_hash(key) + ".png"
        fig.savefig(img_filename, bbox_inches="tight")
        plt.close(fig)

        msg = "motion shifts for {animal_id}-{session}-{scan_idx}".format(**key)
        slack_user = notify.SlackUser() & (experiment.Session() & key)
        slack_user.notify(file=img_filename, file_title=msg)

    def save_video(
        self,
        filename="galvo_corrections.mp4",
        channel=1,
        start_index=0,
        seconds=30,
        dpi=250,
    ):
        """Creates an animation video showing the original vs corrected scan.

        :param string filename: Output filename (path + filename)
        :param int channel: What channel from the scan to use. Starts at 1
        :param int start_index: Where in the scan to start the video.
        :param int seconds: How long in seconds should the animation run.
        :param int dpi: Dots per inch, controls the quality of the video.

        :returns Figure. You can call show() on it.
        :rtype: matplotlib.figure.Figure
        """
        # Get fps and total_num_frames
        fps = (ScanInfo() & self).fetch1("fps")
        num_video_frames = int(round(fps * seconds))
        stop_index = start_index + num_video_frames

        # Load the scan
        scan_filename = (experiment.Scan() & self).local_filenames_as_wildcard
        scan = scanreader.read_scan(scan_filename, dtype=np.float32)
        scan_ = scan[
            self.fetch1("field") - 1, :, :, channel - 1, start_index:stop_index
        ]
        original_scan = scan_.copy()

        # Correct the scan
        correct_raster = (RasterCorrection() & self).get_correct_raster()
        correct_motion = self.get_correct_motion()
        corrected_scan = correct_motion(
            correct_raster(scan_), slice(start_index, stop_index)
        )

        # Create animation
        import matplotlib.animation as animation

        ## Set the figure
        fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)

        axes[0].set_title("Original")
        im1 = axes[0].imshow(
            original_scan[:, :, 0], vmin=original_scan.min(), vmax=original_scan.max()
        )  # just a placeholder
        fig.colorbar(im1, ax=axes[0])
        axes[0].axis("off")

        axes[1].set_title("Corrected")
        im2 = axes[1].imshow(
            corrected_scan[:, :, 0],
            vmin=corrected_scan.min(),
            vmax=corrected_scan.max(),
        )  # just a placeholder
        fig.colorbar(im2, ax=axes[1])
        axes[1].axis("off")

        ## Make the animation
        def update_img(i):
            im1.set_data(original_scan[:, :, i])
            im2.set_data(corrected_scan[:, :, i])

        video = animation.FuncAnimation(
            fig, update_img, corrected_scan.shape[2], interval=1000 / fps
        )

        # Save animation
        if not filename.endswith(".mp4"):
            filename += ".mp4"
        print("Saving video at:", filename)
        print(
            "If this takes too long, stop it and call again with dpi <",
            dpi,
            "(default)",
        )
        video.save(filename, dpi=dpi)

        return fig

    def get_correct_motion(self):
        """ Returns a function to perform motion correction on scans. """
        x_shifts, y_shifts = self.fetch1("x_shifts", "y_shifts")

        return lambda scan, indices=slice(None): galvo_corrections.correct_motion(
            scan, x_shifts[indices], y_shifts[indices]
        )


@schema
class SummaryImages(dj.Computed):
    definition = """ # summary images for each field and channel after corrections

    -> MotionCorrection
    -> shared.Channel
    """

    @property
    def key_source(self):
        return MotionCorrection() & {"pipe_version": CURRENT_VERSION}

    class Average(dj.Part):
        definition = """ # mean of each pixel across time

        -> master
        ---
        average_image           : longblob
        """

    class Correlation(dj.Part):
        definition = """ # average temporal correlation between each pixel and its eight neighbors

        -> master
        ---
        correlation_image           : longblob
        """

    class L6Norm(dj.Part):
        definition = """ # l6-norm of each pixel across time

        -> master
        ---
        l6norm_image           : longblob
        """

    def make(self, key):
        # Read the scan
        scan_filename = (experiment.Scan() & key).local_filenames_as_wildcard
        scan = scanreader.read_scan(scan_filename)

        for channel in range(scan.num_channels):
            # Map: Compute some statistics in different chunks of the scan
            f = performance.parallel_summary_images  # function to map
            raster_phase = (RasterCorrection() & key).fetch1("raster_phase")
            fill_fraction = (ScanInfo() & key).fetch1("fill_fraction")
            y_shifts, x_shifts = (MotionCorrection() & key).fetch1(
                "y_shifts", "x_shifts"
            )
            kwargs = {
                "raster_phase": raster_phase,
                "fill_fraction": fill_fraction,
                "y_shifts": y_shifts,
                "x_shifts": x_shifts,
            }
            results = performance.map_frames(
                f, scan, field_id=key["field"] - 1, channel=channel, kwargs=kwargs
            )

            # Reduce: Compute average images
            average_image = np.sum([r[0] for r in results], axis=0) / scan.num_frames
            l6norm_image = np.sum([r[1] for r in results], axis=0) ** (1 / 6)

            # Reduce: Compute correlation image
            sum_x = np.sum([r[2] for r in results], axis=0)  # h x w
            sum_sqx = np.sum([r[3] for r in results], axis=0)  # h x w
            sum_xy = np.sum([r[4] for r in results], axis=0)  # h x w x 8
            denom_factor = np.sqrt(scan.num_frames * sum_sqx - sum_x ** 2)
            corrs = np.zeros(sum_xy.shape)
            for k in [0, 1, 2, 3]:
                rotated_corrs = np.rot90(corrs, k=k)
                rotated_sum_x = np.rot90(sum_x, k=k)
                rotated_dfactor = np.rot90(denom_factor, k=k)
                rotated_sum_xy = np.rot90(sum_xy, k=k)

                # Compute correlation
                rotated_corrs[1:, :, k] = (
                    scan.num_frames * rotated_sum_xy[1:, :, k]
                    - rotated_sum_x[1:] * rotated_sum_x[:-1]
                ) / (rotated_dfactor[1:] * rotated_dfactor[:-1])
                rotated_corrs[1:, 1:, 4 + k] = (
                    scan.num_frames * rotated_sum_xy[1:, 1:, 4 + k]
                    - rotated_sum_x[1:, 1:] * rotated_sum_x[:-1, :-1]
                ) / (rotated_dfactor[1:, 1:] * rotated_dfactor[:-1, :-1])

                # Return back to original orientation
                corrs = np.rot90(rotated_corrs, k=4 - k)

            correlation_image = np.sum(corrs, axis=-1)
            norm_factor = 5 * np.ones(correlation_image.shape)  # edges
            norm_factor[[0, -1, 0, -1], [0, -1, -1, 0]] = 3  # corners
            norm_factor[1:-1, 1:-1] = 8  # center
            correlation_image /= norm_factor

            # Insert
            field_key = {**key, "channel": channel + 1}
            self.insert1(field_key)
            self.Average().insert1({**field_key, "average_image": average_image})
            self.L6Norm().insert1({**field_key, "l6norm_image": l6norm_image})
            self.Correlation().insert1(
                {**field_key, "correlation_image": correlation_image}
            )

        self.notify(key, scan.num_channels)

    @notify.ignore_exceptions
    def notify(self, key, num_channels):
        fig, axes = plt.subplots(
            num_channels, 2, squeeze=False, figsize=(12, 5 * num_channels)
        )
        axes[0, 0].set_title("L6-Norm", size="small")
        axes[0, 1].set_title("Correlation", size="small")
        for ax in axes.ravel():
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])

        for channel in range(num_channels):
            axes[channel, 0].set_ylabel(
                "Channel {}".format(channel + 1),
                size="large",
                rotation="horizontal",
                ha="right",
            )
            corr = (
                SummaryImages.Correlation() & key & {"channel": channel + 1}
            ).fetch1("correlation_image")
            l6norm = (SummaryImages.L6Norm() & key & {"channel": channel + 1}).fetch1(
                "l6norm_image"
            )
            axes[channel, 0].imshow(l6norm)
            axes[channel, 1].imshow(corr)

        fig.tight_layout()
        img_filename = "/tmp/" + key_hash(key) + ".png"
        fig.savefig(img_filename, bbox_inches="tight")
        plt.close(fig)

        msg = (
            "summary images for {animal_id}-{session}-{scan_idx} field {field}".format(
                **key
            )
        )
        slack_user = notify.SlackUser() & (experiment.Session() & key)
        slack_user.notify(
            file=img_filename, file_title=msg, channel="#pipeline_quality"
        )


@schema
class SegmentationTask(dj.Manual):
    definition = """ # defines the target of segmentation and the channel to use

    -> experiment.Scan
    -> shared.Field
    -> shared.Channel
    -> shared.SegmentationMethod
   ---
    -> experiment.Compartment
    """

    def fill(self, key, channel=1, segmentation_method=6, compartment="soma"):
        for field_key in (ScanInfo.Field() & key).fetch(dj.key):
            tuple_ = {
                **field_key,
                "channel": channel,
                "compartment": compartment,
                "segmentation_method": segmentation_method,
            }
            self.insert1(tuple_, ignore_extra_fields=True, skip_duplicates=True)

    def estimate_num_components(self):
        """Estimates the number of components per field using simple rules of thumb.

        For somatic scans, estimate number of neurons based on:
        (100x100x100)um^3 = 1e6 um^3 -> 100 neurons; (1x1x1)mm^3 = 1e9 um^3 -> 100K neurons

        For axonal/dendritic scans, just ten times our estimate of neurons.

        :returns: Number of components
        :rtype: int
        """

        # Get field dimensions (in micrometers)
        scan = ScanInfo.Field() & self & {"pipe_version": CURRENT_VERSION}
        field_height, field_width = scan.fetch1("um_height", "um_width")
        field_thickness = 10  # assumption
        field_volume = field_width * field_height * field_thickness

        # Estimate number of components
        compartment = self.fetch1("compartment")
        if compartment == "soma":
            num_components = field_volume * 0.0001
        elif compartment == "axon":
            num_components = field_volume * 0.0005  # five times as many neurons
        elif compartment == "bouton":
            num_components = field_volume * 0.001  # ten times as many neurons
        else:
            PipelineException(
                "Compartment type '{}' not recognized".format(compartment)
            )

        return int(round(num_components))


@schema
class DoNotSegment(dj.Manual):
    definition = """ # field/channels that should not be segmented (used for web interface only)

    -> experiment.Scan
    -> shared.Field
    -> shared.Channel
    """


@schema
class Segmentation(dj.Computed):
    definition = """ # Different mask segmentations.

    -> MotionCorrection         # animal_id, session, scan_idx, version, field
    -> SegmentationTask         # animal_id, session, scan_idx, field, channel, segmentation_method
    ---
    segmentation_time=CURRENT_TIMESTAMP     : timestamp     # automatic
    """

    @property
    def key_source(self):
        return MotionCorrection() * SegmentationTask() & {
            "pipe_version": CURRENT_VERSION
        }

    class Mask(dj.Part):
        definition = """ # mask produced by segmentation.

        -> Segmentation
        mask_id         : smallint
        ---
        pixels          : longblob      # indices into the image in column major (Fortran) order
        weights         : longblob      # weights of the mask at the indices above
        """

        def get_mask_as_image(self):
            """ Return this mask as an image (2-d numpy array)."""
            # Get params
            pixels, weights = self.fetch("pixels", "weights")
            image_height, image_width = (ScanInfo.Field() & self).fetch1(
                "px_height", "px_width"
            )

            # Reshape mask
            mask = Segmentation.reshape_masks(
                pixels, weights, image_height, image_width
            )

            return np.squeeze(mask)

    class Manual(dj.Part):
        definition = """ # masks created manually

        -> Segmentation
        """

        def make(self, key):
            print("Warning: Manual segmentation is not implemented in Python.")
            # Copy any masks (and MaskClassification) that were there before
            # Delete key from Segmentation (this is needed for trace and ScanSet and Activity computation to restart when things are added)
            # Show GUI with the current masks
            # User modifies it somehow to produce the new set of masks
            # Insert info in Segmentation -> Segmentation.Manual -> Segmentation.Mask -> MaskClassification -> MaskClassification.Type
            # http://scikit-image.org/docs/dev/api/skimage.future.html#manual-lasso-segmentation (Python lasso masks)

    class CNMF(dj.Part):
        definition = """ # source extraction using constrained non-negative matrix factorization

        -> Segmentation
        ---
        params              : varchar(1024)     # parameters send to CNMF as JSON array
        """

        def make(self, key):
            """Use CNMF to extract masks and traces.

            See caiman_interface.extract_masks for explanation of parameters
            """
            from .utils import caiman_interface as cmn
            import json
            import uuid
            import os

            print("")
            print("*" * 85)
            print("Processing {}".format(key))

            # Get some parameters
            field_id = key["field"] - 1
            channel = key["channel"] - 1
            image_height, image_width = (ScanInfo.Field() & key).fetch1(
                "px_height", "px_width"
            )
            num_frames = (ScanInfo() & key).fetch1("nframes")

            # Read scan
            print("Reading scan...")
            scan_filename = (experiment.Scan() & key).local_filenames_as_wildcard
            scan = scanreader.read_scan(scan_filename)

            # Create memory mapped file (as expected by CaImAn)
            print("Creating memory mapped file...")
            filename = "/tmp/caiman-{}_d1_{}_d2_{}_d3_1_order_C_frames_{}_.mmap".format(
                uuid.uuid4(), image_height, image_width, num_frames
            )
            mmap_shape = (image_height * image_width, num_frames)
            mmap_scan = np.memmap(
                filename, mode="w+", shape=mmap_shape, dtype=np.float32
            )

            # Map: Correct scan and save in memmap scan
            f = performance.parallel_save_memmap  # function to map
            raster_phase = (RasterCorrection() & key).fetch1("raster_phase")
            fill_fraction = (ScanInfo() & key).fetch1("fill_fraction")
            y_shifts, x_shifts = (MotionCorrection() & key).fetch1(
                "y_shifts", "x_shifts"
            )
            kwargs = {
                "raster_phase": raster_phase,
                "fill_fraction": fill_fraction,
                "y_shifts": y_shifts,
                "x_shifts": x_shifts,
                "mmap_scan": mmap_scan,
            }
            results = performance.map_frames(
                f, scan, field_id=field_id, channel=channel, kwargs=kwargs
            )

            # Reduce: Use the minimum values to make memory mapped scan nonnegative
            mmap_scan -= np.min(results)  # bit inefficient but necessary

            # Set CNMF parameters
            ## Set general parameters
            kwargs = {}
            kwargs["num_background_components"] = 1
            kwargs["merge_threshold"] = 0.7
            kwargs["fps"] = (ScanInfo() & key).fetch1("fps")

            # Set params specific to method and segmentation target
            target = (SegmentationTask() & key).fetch1("compartment")
            if key["segmentation_method"] == 2:  # nmf
                if target == "axon":
                    kwargs["init_on_patches"] = True
                    kwargs["proportion_patch_overlap"] = 0.2  # 20% overlap
                    kwargs["num_components_per_patch"] = 15
                    kwargs["init_method"] = "sparse_nmf"
                    kwargs["snmf_alpha"] = 500  # 10^2 to 10^3.5 is a good range
                    kwargs["patch_size"] = tuple(
                        50 / (ScanInfo.Field() & key).microns_per_pixel
                    )  # 50 x 50 microns
                elif target == "bouton":
                    kwargs["init_on_patches"] = False
                    kwargs["num_components"] = (
                        SegmentationTask() & key
                    ).estimate_num_components()
                    kwargs["init_method"] = "greedy_roi"
                    kwargs["soma_diameter"] = tuple(
                        2 / (ScanInfo.Field() & key).microns_per_pixel
                    )
                else:  # soma
                    kwargs["init_on_patches"] = False
                    kwargs["num_components"] = (
                        SegmentationTask() & key
                    ).estimate_num_components()
                    kwargs["init_method"] = "greedy_roi"
                    kwargs["soma_diameter"] = tuple(
                        14 / (ScanInfo.Field() & key).microns_per_pixel
                    )
            else:  # nmf-new
                kwargs["init_on_patches"] = True
                kwargs["proportion_patch_overlap"] = 0.2  # 20% overlap
                if target == "axon":
                    kwargs["num_components_per_patch"] = 15
                    kwargs["init_method"] = "sparse_nmf"
                    kwargs["snmf_alpha"] = 500  # 10^2 to 10^3.5 is a good range
                    kwargs["patch_size"] = tuple(
                        50 / (ScanInfo.Field() & key).microns_per_pixel
                    )  # 50 x 50 microns
                elif target == "bouton":
                    kwargs["num_components_per_patch"] = 5
                    kwargs["init_method"] = "greedy_roi"
                    kwargs["patch_size"] = tuple(
                        20 / (ScanInfo.Field() & key).microns_per_pixel
                    )  # 20 x 20 microns
                    kwargs["soma_diameter"] = tuple(
                        2 / (ScanInfo.Field() & key).microns_per_pixel
                    )
                else:  # soma
                    kwargs["num_components_per_patch"] = 6
                    kwargs["init_method"] = "greedy_roi"
                    kwargs["patch_size"] = tuple(
                        50 / (ScanInfo.Field() & key).microns_per_pixel
                    )
                    kwargs["soma_diameter"] = tuple(
                        8 / (ScanInfo.Field() & key).microns_per_pixel
                    )

            ## Set performance/execution parameters (heuristically), decrease if memory overflows
            kwargs["num_processes"] = 8  # Set to None for all cores available
            kwargs["num_pixels_per_process"] = 10000

            # Extract traces
            print("Extracting masks and traces (cnmf)...")
            scan_ = mmap_scan.reshape(
                (image_height, image_width, num_frames), order="F"
            )
            cnmf_result = cmn.extract_masks(scan_, mmap_scan, **kwargs)
            (
                masks,
                traces,
                background_masks,
                background_traces,
                raw_traces,
            ) = cnmf_result

            # Delete memory mapped scan
            print("Deleting memory mapped scan...")
            os.remove(mmap_scan.filename)

            # Insert CNMF results
            print("Inserting masks, background components and traces...")
            dj.conn()

            ## Insert in CNMF, Segmentation and Fluorescence
            self.insert1({**key, "params": json.dumps(kwargs)})
            Fluorescence().insert1(
                key, allow_direct_insert=True
            )  # we also insert traces

            ## Insert background components
            Segmentation.CNMFBackground().insert1(
                {**key, "masks": background_masks, "activity": background_traces}
            )

            ## Insert masks and traces (masks in Matlab format)
            num_masks = masks.shape[-1]
            masks = masks.reshape(
                -1, num_masks, order="F"
            ).T  # [num_masks x num_pixels] in F order
            raw_traces = raw_traces.astype(np.float32, copy=False)
            for mask_id, mask, trace in zip(range(1, num_masks + 1), masks, raw_traces):
                mask_pixels = np.where(mask)[0]
                mask_weights = mask[mask_pixels]
                mask_pixels += 1  # matlab indices start at 1
                Segmentation.Mask().insert1(
                    {
                        **key,
                        "mask_id": mask_id,
                        "pixels": mask_pixels,
                        "weights": mask_weights,
                    }
                )

                Fluorescence.Trace().insert1(
                    {**key, "mask_id": mask_id, "trace": trace},
                    allow_direct_insert=True,
                )

            Segmentation().notify(key)

        def save_video(
            self,
            filename="cnmf_results.mp4",
            start_index=0,
            seconds=30,
            dpi=250,
            first_n=None,
        ):
            """Creates an animation video showing the results of CNMF.

            :param string filename: Output filename (path + filename)
            :param int start_index: Where in the scan to start the video.
            :param int seconds: How long in seconds should the animation run.
            :param int dpi: Dots per inch, controls the quality of the video.
            :param int first_n: Draw only the first n components.

            :returns Figure. You can call show() on it.
            :rtype: matplotlib.figure.Figure
            """
            # Get fps and calculate total number of frames
            fps = (ScanInfo() & self).fetch1("fps")
            num_video_frames = int(round(fps * seconds))
            stop_index = start_index + num_video_frames

            # Load the scan
            channel = self.fetch1("channel") - 1
            field_id = self.fetch1("field") - 1
            scan_filename = (experiment.Scan() & self).local_filenames_as_wildcard
            scan = scanreader.read_scan(scan_filename, dtype=np.float32)
            scan_ = scan[field_id, :, :, channel, start_index:stop_index]

            # Correct the scan
            correct_raster = (RasterCorrection() & self).get_correct_raster()
            correct_motion = (MotionCorrection() & self).get_correct_motion()
            scan_ = correct_motion(
                correct_raster(scan_), slice(start_index, stop_index)
            )

            # Get scan dimensions
            image_height, image_width, _ = scan_.shape
            num_pixels = image_height * image_width

            # Get masks and traces
            masks = (Segmentation() & self).get_all_masks()
            traces = (Fluorescence() & self).get_all_traces()
            background_masks, background_traces = (
                Segmentation.CNMFBackground() & self
            ).fetch1("masks", "activity")

            # Select first n components
            if first_n is not None:
                masks = masks[:, :, :first_n]
                traces = traces[:first_n, :]

            # Drop frames that won't be displayed
            traces = traces[:, start_index:stop_index]
            background_traces = background_traces[:, start_index:stop_index]

            # Create movies
            extracted = np.dot(masks.reshape(num_pixels, -1), traces)
            extracted = extracted.reshape(image_height, image_width, -1)
            background = np.dot(
                background_masks.reshape(num_pixels, -1), background_traces
            )
            background = background.reshape(image_height, image_width, -1)
            residual = scan_ - extracted - background

            # Create animation
            import matplotlib.animation as animation

            ## Set the figure
            fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)

            axes[0, 0].set_title("Original (Y)")
            im1 = axes[0, 0].imshow(
                scan_[:, :, 0], vmin=scan_.min(), vmax=scan_.max()
            )  # just a placeholder
            fig.colorbar(im1, ax=axes[0, 0])

            axes[0, 1].set_title("Extracted (A*C)")
            im2 = axes[0, 1].imshow(
                extracted[:, :, 0], vmin=extracted.min(), vmax=extracted.max()
            )
            fig.colorbar(im2, ax=axes[0, 1])

            axes[1, 0].set_title("Background (B*F)")
            im3 = axes[1, 0].imshow(
                background[:, :, 0], vmin=background.min(), vmax=background.max()
            )
            fig.colorbar(im3, ax=axes[1, 0])

            axes[1, 1].set_title("Residual (Y - A*C - B*F)")
            im4 = axes[1, 1].imshow(
                residual[:, :, 0], vmin=residual.min(), vmax=residual.max()
            )
            fig.colorbar(im4, ax=axes[1, 1])

            for ax in axes.ravel():
                ax.axis("off")

            ## Make the animation
            def update_img(i):
                im1.set_data(scan_[:, :, i])
                im2.set_data(extracted[:, :, i])
                im3.set_data(background[:, :, i])
                im4.set_data(residual[:, :, i])

            video = animation.FuncAnimation(
                fig, update_img, scan_.shape[2], interval=1000 / fps
            )

            # Save animation
            if not filename.endswith(".mp4"):
                filename += ".mp4"
            print("Saving video at:", filename)
            print(
                "If this takes too long, stop it and call again with dpi <",
                dpi,
                "(default)",
            )
            video.save(filename, dpi=dpi)

            return fig

    class CNMFBackground(dj.Part):
        definition = """ # inferred background components

        -> Segmentation.CNMF
        ---
        masks               : longblob      # array (im_height x im_width x num_background_components)
        activity            : longblob      # array (num_background_components x timesteps)
        """

    def make(self, key):
        # Create masks
        if key["segmentation_method"] == 1:  # manual
            Segmentation.Manual().make(key)
        elif key["segmentation_method"] in [2, 6]:  # nmf
            self.insert1(key)
            Segmentation.CNMF().make(key)
        elif key["segmentation_method"] in [3, 4]:  # nmf_patches, nmf-boutons
            msg = "This method has been deprecated, use segmentation_method 6"
            raise PipelineException(msg)
        else:
            msg = "Unrecognized segmentation method {}".format(
                key["segmentation_method"]
            )
            raise PipelineException(msg)

    @notify.ignore_exceptions
    def notify(self, key):
        fig = (Segmentation() & key).plot_masks()
        img_filename = "/tmp/" + key_hash(key) + ".png"
        fig.savefig(img_filename, bbox_inches="tight")
        plt.close(fig)

        msg = "segmentation for {animal_id}-{session}-{scan_idx} field {field}".format(
            **key
        )
        slack_user = notify.SlackUser() & (experiment.Session() & key)
        slack_user.notify(file=img_filename, file_title=msg)

    @staticmethod
    def reshape_masks(mask_pixels, mask_weights, image_height, image_width):
        """ Reshape masks into an image_height x image_width x num_masks array."""
        masks = np.zeros(
            [image_height, image_width, len(mask_pixels)], dtype=np.float32
        )

        # Reshape each mask
        for i, (mp, mw) in enumerate(zip(mask_pixels, mask_weights)):
            mask_as_vector = np.zeros(image_height * image_width)
            mask_as_vector[np.squeeze(mp - 1).astype(int)] = np.squeeze(mw)
            masks[:, :, i] = mask_as_vector.reshape(
                image_height, image_width, order="F"
            )

        return masks

    def get_all_masks(self):
        """Returns an image_height x image_width x num_masks matrix with all masks."""
        mask_rel = Segmentation.Mask() & self

        # Get masks
        image_height, image_width = (ScanInfo.Field() & self).fetch1(
            "px_height", "px_width"
        )
        mask_pixels, mask_weights = mask_rel.fetch(
            "pixels", "weights", order_by="mask_id"
        )

        # Reshape masks
        masks = Segmentation.reshape_masks(
            mask_pixels, mask_weights, image_height, image_width
        )

        return masks

    def plot_masks(self, threshold=0.97, first_n=None):
        """Draw contours of masks over the correlation image (if available).

        :param threshold: Threshold on the cumulative mass to define mask contours. Lower
            for tighter contours.
        :param first_n: Number of masks to plot. None for all.

        :returns Figure. You can call show() on it.
        :rtype: matplotlib.figure.Figure
        """
        # Get masks
        masks = self.get_all_masks()
        if first_n is not None:
            masks = masks[:, :, :first_n]

        # Get correlation image if defined, black background otherwise.
        image_rel = SummaryImages.Correlation() & self
        if image_rel:
            background_image = image_rel.fetch1("correlation_image")
        else:
            background_image = np.zeros(masks.shape[:-1])

        # Plot background
        image_height, image_width, num_masks = masks.shape
        figsize = np.array([image_width, image_height]) / min(image_height, image_width)
        fig = plt.figure(figsize=figsize * 7)
        plt.imshow(background_image)

        # Draw contours
        cumsum_mask = np.empty([image_height, image_width])
        for i in range(num_masks):
            mask = masks[:, :, i]

            ## Compute cumulative mass (similar to caiman)
            indices = np.unravel_index(
                np.flip(np.argsort(mask, axis=None), axis=0), mask.shape
            )  # max to min value in mask
            cumsum_mask[indices] = np.cumsum(mask[indices] ** 2) / np.sum(mask ** 2)

            ## Plot contour at desired threshold (with random color)
            random_color = (np.random.rand(), np.random.rand(), np.random.rand())
            plt.contour(cumsum_mask, [threshold], linewidths=0.8, colors=[random_color])

        return fig


@schema
class Fluorescence(dj.Computed):
    definition = """  # fluorescence traces before spike extraction or filtering

    -> Segmentation
    """

    @property
    def key_source(self):
        return Segmentation() & {"pipe_version": CURRENT_VERSION}

    class Trace(dj.Part):
        definition = """

        -> Fluorescence
        -> Segmentation.Mask
        ---
        trace                   : longblob
        """

    def make(self, key):
        # Load scan
        print("Reading scan...")
        field_id = key["field"] - 1
        channel = key["channel"] - 1
        scan_filename = (experiment.Scan() & key).local_filenames_as_wildcard
        scan = scanreader.read_scan(scan_filename)

        # Map: Extract traces
        print("Creating fluorescence traces...")
        f = performance.parallel_fluorescence  # function to map
        raster_phase = (RasterCorrection() & key).fetch1("raster_phase")
        fill_fraction = (ScanInfo() & key).fetch1("fill_fraction")
        y_shifts, x_shifts = (MotionCorrection() & key).fetch1("y_shifts", "x_shifts")
        mask_ids, pixels, weights = (Segmentation.Mask() & key).fetch(
            "mask_id", "pixels", "weights"
        )
        kwargs = {
            "raster_phase": raster_phase,
            "fill_fraction": fill_fraction,
            "y_shifts": y_shifts,
            "x_shifts": x_shifts,
            "mask_pixels": pixels,
            "mask_weights": weights,
        }
        results = performance.map_frames(
            f, scan, field_id=field_id, channel=channel, kwargs=kwargs
        )

        # Reduce: Concatenate
        traces = np.zeros((len(mask_ids), scan.num_frames), dtype=np.float32)
        for frames, chunk_traces in results:
            traces[:, frames] = chunk_traces

        # Insert
        self.insert1(key)
        for mask_id, trace in zip(mask_ids, traces):
            Fluorescence.Trace().insert1({**key, "mask_id": mask_id, "trace": trace})

        self.notify(key)

    @notify.ignore_exceptions
    def notify(self, key):
        fig = plt.figure(figsize=(15, 4))
        plt.plot((Fluorescence() & key).get_all_traces().T)
        img_filename = "/tmp/" + key_hash(key) + ".png"
        fig.savefig(img_filename, bbox_inches="tight")
        plt.close(fig)

        msg = (
            "calcium traces for {animal_id}-{session}-{scan_idx} field {field}".format(
                **key
            )
        )
        slack_user = notify.SlackUser() & (experiment.Session() & key)
        slack_user.notify(file=img_filename, file_title=msg)

    def get_all_traces(self):
        """ Returns a num_traces x num_timesteps matrix with all traces."""
        traces = (Fluorescence.Trace() & self).fetch("trace", order_by="mask_id")
        return np.array([x.squeeze() for x in traces])


@schema
class MaskClassification(dj.Computed):
    definition = """ # classification of segmented masks.

    -> Segmentation                     # animal_id, session, scan_idx, reso_version, field, channel, segmentation_method
    -> shared.ClassificationMethod
    ---
    classif_time=CURRENT_TIMESTAMP    : timestamp     # automatic
    """

    @property
    def key_source(self):
        return Segmentation() * shared.ClassificationMethod() & {
            "pipe_version": CURRENT_VERSION
        }

    class Type(dj.Part):
        definition = """

        -> MaskClassification
        -> Segmentation.Mask
        ---
        -> shared.MaskType
        """

    def make(self, key):
        # Skip axonal scans
        target = (SegmentationTask() & key).fetch1("compartment")
        if key["classification_method"] == 2 and target != "soma":
            print(
                "Warning: Skipping {}. Automatic classification works only with somatic "
                "scans".format(key)
            )
            return

        # Get masks
        image_height, image_width = (ScanInfo.Field() & key).fetch1(
            "px_height", "px_width"
        )
        mask_ids, pixels, weights = (Segmentation.Mask() & key).fetch(
            "mask_id", "pixels", "weights"
        )
        masks = Segmentation.reshape_masks(pixels, weights, image_height, image_width)

        # Classify masks
        if key["classification_method"] == 1:  # manual
            if not SummaryImages() & key:
                msg = "Need to populate SummaryImages before manual mask classification"
                raise PipelineException(msg)

            template = (SummaryImages.Correlation() & key).fetch1("correlation_image")
            masks = masks.transpose([2, 0, 1])  # num_masks, image_height, image_width
            mask_types = mask_classification.classify_manual(masks, template)
        elif key["classification_method"] == 2:  # cnn-caiman
            from .utils import caiman_interface as cmn

            soma_diameter = tuple(14 / (ScanInfo.Field() & key).microns_per_pixel)
            probs = cmn.classify_masks(masks, soma_diameter)
            mask_types = ["soma" if prob > 0.75 else "artifact" for prob in probs]
        else:
            msg = "Unrecognized classification method {}".format(
                key["classification_method"]
            )
            raise PipelineException(msg)

        print("Generated types:", mask_types)

        # Insert results
        self.insert1(key)
        for mask_id, mask_type in zip(mask_ids, mask_types):
            MaskClassification.Type().insert1(
                {**key, "mask_id": mask_id, "type": mask_type}
            )

        self.notify(key, mask_types)

    @notify.ignore_exceptions
    def notify(self, key, mask_types):
        fig = (MaskClassification() & key).plot_masks()
        img_filename = "/tmp/" + key_hash(key) + ".png"
        fig.savefig(img_filename, bbox_inches="tight")
        plt.close(fig)

        msg = (
            "mask classification for {animal_id}-{session}-{scan_idx} field {field}: "
            "{somas} somas and {arts} artifacts"
        ).format(
            **key, somas=mask_types.count("soma"), arts=mask_types.count("artifact")
        )
        slack_user = notify.SlackUser() & (experiment.Session() & key)
        slack_user.notify(
            file=img_filename, file_title=msg, channel="#pipeline_quality"
        )

    def plot_masks(self, threshold=0.99):
        """Draw contours of masks over the correlation image (if available) with different
        colors per type

        :param threshold: Threshold on the cumulative mass to define mask contours. Lower
            for tighter contours.

        :returns Figure. You can call show() on it.
        :rtype: matplotlib.figure.Figure
        """
        # Get masks
        masks = (Segmentation() & self).get_all_masks()
        mask_types = (MaskClassification.Type() & self).fetch("type")
        colormap = {
            "soma": "b",
            "axon": "k",
            "dendrite": "c",
            "neuropil": "y",
            "artifact": "r",
            "unknown": "w",
        }

        # Get correlation image if defined, black background otherwise.
        image_rel = SummaryImages.Correlation() & self
        if image_rel:
            background_image = image_rel.fetch1("correlation_image")
        else:
            background_image = np.zeros(masks.shape[:-1])

        # Plot background
        image_height, image_width, num_masks = masks.shape
        figsize = np.array([image_width, image_height]) / min(image_height, image_width)
        fig = plt.figure(figsize=figsize * 7)
        plt.imshow(background_image)

        # Draw contours
        cumsum_mask = np.empty([image_height, image_width])
        for i in range(num_masks):
            mask = masks[:, :, i]
            color = colormap[mask_types[i]]

            ## Compute cumulative mass (similar to caiman)
            indices = np.unravel_index(
                np.flip(np.argsort(mask, axis=None), axis=0), mask.shape
            )  # max to min value in mask
            cumsum_mask[indices] = np.cumsum(mask[indices] ** 2) / np.sum(mask ** 2)

            ## Plot contour at desired threshold
            plt.contour(cumsum_mask, [threshold], linewidths=0.8, colors=[color])

        return fig


@schema
class ScanSet(dj.Computed):
    definition = """ # set of all units in the same scan

    -> Fluorescence         # processing done per field
    """

    @property
    def key_source(self):
        return Fluorescence() & {"pipe_version": CURRENT_VERSION}

    class Unit(dj.Part):
        definition = """ # single unit in the scan

        -> ScanInfo
        -> shared.SegmentationMethod
        unit_id                 : int               # unique per scan & segmentation method
        ---
        -> ScanSet                                  # for it to act as a part table of ScanSet
        -> Fluorescence.Trace
        """

    class UnitInfo(dj.Part):
        definition = """ # unit type, coordinates and delay time

        -> ScanSet.Unit
        ---
        um_x                : smallint      # x-coordinate of centroid in motor coordinate system
        um_y                : smallint      # y-coordinate of centroid in motor coordinate system
        um_z                : smallint      # z-coordinate of mask relative to surface of the cortex
        px_x                : smallint      # x-coordinate of centroid in the frame
        px_y                : smallint      # y-coordinate of centroid in the frame
        ms_delay            : smallint      # (ms) delay from start of frame to recording of this unit
        """

    def _job_key(self, key):
        # Force reservation key to be per scan so diff fields are not run in parallel
        return {k: v for k, v in key.items() if k not in ["field", "channel"]}

    def make(self, key):
        from pipeline.utils import caiman_interface as cmn

        # Get masks
        image_height, image_width = (ScanInfo.Field() & key).fetch1(
            "px_height", "px_width"
        )
        mask_ids, pixels, weights = (Segmentation.Mask() & key).fetch(
            "mask_id", "pixels", "weights"
        )
        masks = Segmentation.reshape_masks(pixels, weights, image_height, image_width)

        # Compute units' coordinates
        px_center = [image_height / 2, image_width / 2]
        um_center = (ScanInfo.Field() & key).fetch1("y", "x")
        um_z = (ScanInfo.Field() & key).fetch1("z")
        px_centroids = cmn.get_centroids(masks)
        um_centroids = (
            um_center
            + (px_centroids - px_center) * (ScanInfo.Field() & key).microns_per_pixel
        )

        # Compute units' delays
        delay_image = (ScanInfo.Field() & key).fetch1("delay_image")
        delays = np.sum(masks * np.expand_dims(delay_image, -1), axis=(0, 1)) / np.sum(
            masks, axis=(0, 1)
        )
        delays = np.round(delays * 1e3).astype(np.int16)  # in milliseconds

        # Get next unit_id for scan
        unit_rel = ScanSet.Unit().proj() & key
        unit_id = np.max(unit_rel.fetch("unit_id")) + 1 if unit_rel else 1

        # Insert in ScanSet
        self.insert1(key)

        # Insert units
        unit_ids = range(unit_id, unit_id + len(mask_ids) + 1)
        for unit_id, mask_id, (um_y, um_x), (px_y, px_x), delay in zip(
            unit_ids, mask_ids, um_centroids, px_centroids, delays
        ):
            ScanSet.Unit().insert1({**key, "unit_id": unit_id, "mask_id": mask_id})

            unit_info = {
                **key,
                "unit_id": unit_id,
                "um_x": um_x,
                "um_y": um_y,
                "um_z": um_z,
                "px_x": px_x,
                "px_y": px_y,
                "ms_delay": delay,
            }
            ScanSet.UnitInfo().insert1(
                unit_info, ignore_extra_fields=True
            )  # ignore field and channel

    def plot_centroids(self, first_n=None):
        """Draw masks centroids over the correlation image. Works on a single field/channel

        :param first_n: Number of masks to plot. None for all

        :returns Figure. You can call show() on it.
        :rtype: matplotlib.figure.Figure
        """
        # Get centroids
        centroids = self.get_all_centroids(centroid_type="px")
        if first_n is not None:
            centroids = centroids[:, :first_n]  # select first n components

        # Get correlation image if defined, black background otherwise.
        image_rel = SummaryImages.Correlation() & self
        if image_rel:
            background_image = image_rel.fetch1("correlation_image")
        else:
            image_height, image_width = (ScanInfo.Field() & self).fetch1(
                "px_height", "px_width"
            )
            background_image = np.zeros([image_height, image_width])

        # Plot centroids
        image_height, image_width = background_image.shape
        figsize = np.array([image_width, image_height]) / min(image_height, image_width)
        fig = plt.figure(figsize=figsize * 7)
        plt.imshow(background_image)
        plt.plot(centroids[:, 0], centroids[:, 1], "ow", markersize=3)

        return fig

    def plot_centroids3d(self):
        """Plots the centroids of all units in the motor coordinate system (in microns)

        :returns Figure. You can call show() on it.
        :rtype: matplotlib.figure.Figure
        """
        from mpl_toolkits.mplot3d import Axes3D

        # Get centroids
        centroids = self.get_all_centroids()

        # Plot
        # TODO: Add different colors for different types, correlation image as 2-d planes
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2])
        ax.invert_zaxis()
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
        ax.set_zlabel("z (um)")

        return fig

    def get_all_centroids(self, centroid_type="um"):
        """Returns the centroids for all units in ScanSet (could be limited to field).

        Centroid type is either 'um' or 'px':
            'um': Array (num_units x 3) with x, y, z in motor coordinate system (microns).
            'px': Array (num_units x 2) with x, y pixel coordinates.
        """
        units_rel = ScanSet.UnitInfo() & (ScanSet.Unit() & self)
        if centroid_type == "um":
            xs, ys, zs = units_rel.fetch("um_x", "um_y", "um_z", order_by="unit_id")
            centroids = np.stack([xs, ys, zs], axis=1)
        else:
            xs, ys = units_rel.fetch("px_x", "px_y", order_by="unit_id")
            centroids = np.stack([xs, ys], axis=1)
        return centroids


@schema
class Activity(dj.Computed):
    definition = """ # activity inferred from fluorescence traces

    -> ScanSet                                      # processing done per field
    -> shared.SpikeMethod
    ---
    activity_time=CURRENT_TIMESTAMP   : timestamp     # automatic
    """

    @property
    def key_source(self):
        return ScanSet() * shared.SpikeMethod() & {"pipe_version": CURRENT_VERSION}

    class Trace(dj.Part):
        definition = """ # deconvolved calcium acitivity

        -> ScanSet.Unit
        -> shared.SpikeMethod
        ---
        -> Activity                         # for it to act as part table of Activity
        trace               : longblob
        """

    class ARCoefficients(dj.Part):
        definition = """ # fitted parameters for the autoregressive process (nmf deconvolution)

        -> Activity.Trace
        ---
        g                   : blob          # g1, g2, ... coefficients for the AR process
    """

    def make(self, key):
        print("Creating activity traces for", key)

        # Get fluorescence
        fps = (ScanInfo() & key).fetch1("fps")
        unit_ids, traces = (ScanSet.Unit() * Fluorescence.Trace() & key).fetch(
            "unit_id", "trace"
        )
        full_traces = [signal.fill_nans(np.squeeze(trace).copy()) for trace in traces]

        # Insert in Activity
        self.insert1(key)
        if key["spike_method"] == 2:  # oopsie
            import pyfnnd  # Install from https://github.com/cajal/PyFNND.git

            for unit_id, trace in zip(unit_ids, full_traces):
                spike_trace = pyfnnd.deconvolve(trace, dt=1 / fps)[0].astype(
                    np.float32, copy=False
                )
                Activity.Trace().insert1(
                    {**key, "unit_id": unit_id, "trace": spike_trace}
                )

        elif key["spike_method"] == 3:  # stm
            import c2s  # Install from https://github.com/lucastheis/c2s

            for unit_id, trace in zip(unit_ids, full_traces):
                start = signal.notnan(trace)
                end = signal.notnan(trace, len(trace) - 1, increment=-1)
                trace_dict = {
                    "calcium": np.atleast_2d(trace[start : end + 1]),
                    "fps": fps,
                }

                data = c2s.predict(c2s.preprocess([trace_dict], fps=fps), verbosity=0)
                spike_trace = np.squeeze(data[0].pop("predictions")).astype(
                    np.float32, copy=False
                )

                Activity.Trace().insert1(
                    {**key, "unit_id": unit_id, "trace": spike_trace}
                )

        elif key["spike_method"] == 5:  # nmf
            from pipeline.utils import caiman_interface as cmn
            import multiprocessing as mp

            with mp.Pool(10) as pool:
                results = pool.map(cmn.deconvolve, full_traces)
            for unit_id, (spike_trace, ar_coeffs) in zip(unit_ids, results):
                spike_trace = spike_trace.astype(np.float32, copy=False)
                Activity.Trace().insert1(
                    {**key, "unit_id": unit_id, "trace": spike_trace}
                )
                Activity.ARCoefficients().insert1(
                    {**key, "unit_id": unit_id, "g": ar_coeffs},
                    ignore_extra_fields=True,
                )

        elif key["spike_method"] == 6:  # dnmf
            from pipeline.utils import caiman_interface as cmn
            import multiprocessing as mp
            from functools import partial

            scan_fps = (ScanInfo & key).fetch1('fps')
            deconvolve = partial(cmn.deconvolve_detrended, scan_fps=scan_fps)
            with mp.Pool(10) as pool:
                results = pool.map(deconvolve, full_traces)
            for unit_id, (spike_trace, ar_coeffs) in zip(unit_ids, results):
                spike_trace = spike_trace.astype(np.float32, copy=False)
                Activity.Trace().insert1(
                    {**key, "unit_id": unit_id, "trace": spike_trace}
                )
                Activity.ARCoefficients().insert1(
                    {**key, "unit_id": unit_id, "g": ar_coeffs},
                    ignore_extra_fields=True,
                )
        else:
            msg = "Unrecognized spike method {}".format(key["spike_method"])
            raise PipelineException(msg)

        self.notify(key)

    @notify.ignore_exceptions
    def notify(self, key):
        fig = plt.figure(figsize=(15, 4))
        plt.plot((Activity() & key).get_all_spikes().T)
        img_filename = "/tmp/" + key_hash(key) + ".png"
        fig.savefig(img_filename, bbox_inches="tight")
        plt.close(fig)

        msg = "spike traces for {animal_id}-{session}-{scan_idx} field {field}".format(
            **key
        )
        slack_user = notify.SlackUser() & (experiment.Session() & key)
        slack_user.notify(file=img_filename, file_title=msg)

    def plot_impulse_responses(self, num_timepoints=100):
        """Plots the impulse response functions for all traces.

        :param int num_timepoints: The number of points after impulse to use for plotting.

        :returns Figure. You can call show() on it.
        :rtype: matplotlib.figure.Figure
        """
        ar_rel = Activity.ARCoefficients() & (Activity.Trace() & self)
        if ar_rel:  # if an AR model was used
            # Get some params
            fps = (ScanInfo() & self).fetch1("fps")
            ar_coeffs = ar_rel.fetch("g")

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
                        irf[i] = np.sum(g[:i] * irf[i - 1 :: -1])
                    else:
                        irf[i] = np.sum(g * irf[i - 1 : i - AR_order - 1 : -1])

                # Plot
                plt.plot(x_axis, irf)
                plt.xlabel("Seconds")

            return fig

    def get_all_spikes(self):
        """ Returns a num_traces x num_timesteps matrix with all spikes."""
        spikes = (Activity.Trace() & self).fetch("trace", order_by="unit_id")
        return np.array([x.squeeze() for x in spikes])


@schema
class ScanDone(dj.Computed):
    definition = """ # scans that are fully processed (updated every time a field is added)

    -> ScanInfo
    -> shared.SegmentationMethod
    -> shared.SpikeMethod
    """

    @property
    def key_source(self):
        return Activity() & {"pipe_version": CURRENT_VERSION}

    @property
    def target(self):
        return (
            ScanDone.Partial()
        )  # trigger make_tuples for fields in Activity that aren't in ScanDone.Partial

    def _job_key(self, key):
        # Force reservation key to be per scan so diff fields are not run in parallel
        return {k: v for k, v in key.items() if k not in ["field", "channel"]}

    class Partial(dj.Part):
        definition = """ # fields that have been processed in the current scan

        -> ScanDone
        -> Activity
        """

    def make(self, key):
        scan_key = {k: v for k, v in key.items() if k in self.heading}

        # Delete current ScanDone entry
        with dj.config(safemode=False):
            (ScanDone() & scan_key).delete()

        # Reinsert in ScanDone
        self.insert1(scan_key)

        # Insert all processed fields in Partial
        ScanDone.Partial().insert((Activity() & scan_key).proj())


from . import stack


@schema
class StackCoordinates(dj.Computed):
    definition = """ # centroids of each unit in motor/stack coordinate system

    -> ScanSet          # animal_id, session, scan_idx, channel, field, segmentation_method, pipe_version
    -> stack.Registration.proj(session='scan_session')  # animal_id, stack_session, stack_idx, volume_id, session, scan_idx, field, stack_channel, scan_channel, registration_method
    """

    class UnitInfo(dj.Part):
        definition = """ # ScanSet.UnitInfo centroids mapped to stack coordinates

        -> master                       # this will add field and channels back
        -> ScanSet.Unit
        ---
        stack_x         : float
        stack_y         : float
        stack_z         : float
        """

    def make(self, key):
        from scipy import ndimage

        # Get registration grid (px -> stack_coordinate)
        stack_key = {**key, "scan_session": key["session"]}
        field_res = (ScanInfo.Field & key).microns_per_pixel
        grid = (stack.Registration & stack_key).get_grid(
            type="affine", desired_res=field_res
        )

        self.insert1(key)
        field_units = ScanSet.UnitInfo & (ScanSet.Unit & key)
        for unit_key, px_x, px_y in zip(*field_units.fetch("KEY", "px_x", "px_y")):
            px_coords = np.array([[px_y], [px_x]])
            unit_x, unit_y, unit_z = [
                ndimage.map_coordinates(grid[..., i], px_coords, order=1)[0]
                for i in range(3)
            ]
            StackCoordinates.UnitInfo.insert1(
                {
                    **key,
                    **unit_key,
                    "stack_x": unit_x,
                    "stack_y": unit_y,
                    "stack_z": unit_z,
                }
            )


anatomy = dj.create_virtual_module("pipeline_anatomy", "pipeline_anatomy")


@schema
class AreaMembership(dj.Computed):
    definition = """ # cell membership in visual areas according to stack registered retinotopy
    ret_hash             : varchar(32)                  # single attribute representation of retinotopic map key
    -> StackCoordinates
    ---
    """

    class UnitInfo(dj.Part):
        definition = """ # confidence in area assignment per unit according to stack coordinates
        -> master
        -> anatomy.Area
        -> StackCoordinates.UnitInfo
        ---
        confidence             : float                        # confidence in area assignment
        """

    @property
    def key_source(self):
        ret_rel = stack.Area.proj(
            ret_session="scan_session",
            ret_scan_idx="scan_idx",
            ret_channel="scan_channel",
        )
        key_source = ret_rel * StackCoordinates
        heading_str = list(self.heading)

        return dj.U(*heading_str) & key_source

    def make(self, key):

        from scipy.interpolate import griddata

        mask_rel = stack.Area.Mask.proj(
            "mask",
            ret_session="scan_session",
            ret_scan_idx="scan_idx",
            ret_channel="scan_channel",
        )
        mask_keys, masks = (mask_rel & key).fetch("KEY", "mask")

        fetch_str = ["x", "y", "um_width", "um_height", "px_width", "px_height"]
        stack_rel = stack.CorrectedStack.proj(*fetch_str, stack_session="session") & key
        cent_x, cent_y, um_w, um_h, px_w, px_h = stack_rel.fetch1(*fetch_str)

        stack_edges = np.array((cent_x - um_w / 2, cent_y - um_h / 2))
        stack_px_dims = np.array((px_w, px_h))
        stack_um_dims = np.array((um_w, um_h))

        stack_px_grid = np.meshgrid(*[np.arange(d) + 0.5 for d in stack_px_dims])

        ks, sxs, sys = (StackCoordinates.UnitInfo & key).fetch(
            "KEY", "stack_x", "stack_y"
        )
        pxs, pys = [
            np.array((coord - edge) * px_per_um)
            for coord, edge, px_per_um in zip(
                (sxs, sys), stack_edges, stack_px_dims / stack_um_dims
            )
        ]

        unit_tups = []
        for mask_key, mask in zip(mask_keys, masks):
            grid_locs = np.array([grid.ravel() for grid in stack_px_grid]).T
            grid_vals = mask.ravel()
            grid_query = np.vstack((pxs, pys)).T

            confs = griddata(grid_locs, grid_vals, grid_query, method="nearest")
            mems = confs > 0

            unit_tups.append(
                [
                    {**mask_key, **k, "confidence": conf}
                    for k, conf in zip(np.array(ks)[mems], confs[mems])
                ]
            )

        self.insert1(key)
        self.UnitInfo.insert(np.concatenate(unit_tups), ignore_extra_fields=True)


@schema
class Func2StructMatching(dj.Computed):
    definition = """ # match functional masks to structural masks

    -> ScanSet                  # animal_id, session, scan_idx, pipe_version, field, channel
    -> stack.FieldSegmentation.proj(session='scan_session') # animal_id, stack_session, stack_idx, volume_id, session, scan_idx, field, stack_channel, scan_channel, registration_method, stacksegm_channel, stacksegm_method
    ---
    key_hash        : varchar(32)       # single attribute representation of the key (used to avoid going over 16 attributes in the key)
    """

    class AllMatches(dj.Part):
        definition = """ # store all possible matches (one functional cell could match with more than one structural mask and viceversa)

        key_hash        : varchar(32)       # master key
        unit_id         : int               # functional unit id
        sunit_id        : int               # structural unit id
        ---
        iou             : float             # intersection-over-union of the 2-d masks
        """
        # Used key_hash because key using ScanSet.Unit, FieldSegmentation.StackUnit has
        # more than 16 attributes and MySQL complains. I added the foreign key constraints
        # manually

    class Match(dj.Part):
        definition = """ # match of a functional mask to a structural mask (1:1 relation)

        -> master
        -> ScanSet.Unit
        ---
        -> stack.FieldSegmentation.StackUnit.proj(session='scan_session')
        iou             : float         # Intersection-over-Union of the 2-d masks
        distance2d      : float         # distance between centroid of 2-d masks
        distance3d      : float         # distance between functional centroid and structural centroid
        """

    def make(self, key):
        from .utils import registration
        from scipy import ndimage

        # Get caiman masks and resize them
        field_dims = (ScanInfo.Field & key).fetch1("um_height", "um_width")
        masks = np.moveaxis((Segmentation & key).get_all_masks(), -1, 0)
        masks = np.stack(
            [registration.resize(m, field_dims, desired_res=1) for m in masks]
        )
        scansetunit_keys = (ScanSet.Unit & key).fetch("KEY", order_by="mask_id")

        # Binarize masks
        binary_masks = np.zeros(masks.shape, dtype=bool)
        for i, mask in enumerate(masks):
            ## Compute cumulative mass (similar to caiman)
            indices = np.unravel_index(
                np.flip(np.argsort(mask, axis=None), axis=0), mask.shape
            )  # max to min value in mask
            cumsum_mask = np.cumsum(mask[indices] ** 2) / np.sum(mask ** 2)  # + 1e-9)
            binary_masks[i][indices] = cumsum_mask < 0.9

        # Get structural segmentation and registration grid
        stack_key = {**key, "scan_session": key["session"]}
        segmented_field = (stack.FieldSegmentation & stack_key).fetch1("segm_field")
        grid = (stack.Registration & stack_key).get_grid(type="affine", desired_res=1)
        sunit_ids = (stack.FieldSegmentation.StackUnit & stack_key).fetch(
            "sunit_id", order_by="sunit_id"
        )

        # Create matrix with IOU values (rows for structural units, columns for functional units)
        ious = []
        for sunit_id in sunit_ids:
            binary_sunit = segmented_field == sunit_id
            intersection = np.logical_and(binary_masks, binary_sunit).sum(
                axis=(1, 2)
            )  # num_masks
            union = np.logical_or(binary_masks, binary_sunit).sum(
                axis=(1, 2)
            )  # num_masks
            ious.append(intersection / union)
        iou_matrix = np.stack(ious)

        # Save all possible matches / iou_matrix > 0
        self.insert1({**key, "key_hash": key_hash(key)})
        for mask_idx, func_idx in zip(*np.nonzero(iou_matrix)):
            self.AllMatches.insert1(
                {
                    "key_hash": key_hash(key),
                    "unit_id": scansetunit_keys[func_idx]["unit_id"],
                    "sunit_id": sunit_ids[mask_idx],
                    "iou": iou_matrix[mask_idx, func_idx],
                }
            )

        # Iterate over matches (from best to worst), insert
        while iou_matrix.max() > 0:
            # Get next best
            best_mask, best_func = np.unravel_index(
                np.argmax(iou_matrix), iou_matrix.shape
            )
            best_iou = iou_matrix[best_mask, best_func]

            # Get stack unit coordinates
            coords = (
                stack.FieldSegmentation.StackUnit
                & stack_key
                & {"sunit_id": sunit_ids[best_mask]}
            ).fetch1("sunit_z", "sunit_y", "sunit_x", "mask_z", "mask_y", "mask_x")
            sunit_z, sunit_y, sunit_x, mask_z, mask_y, mask_x = coords

            # Compute distance to 2-d and 3-d mask
            px_y, px_x = ndimage.measurements.center_of_mass(binary_masks[best_func])
            px_coords = np.array([[px_y], [px_x]])
            func_x, func_y, func_z = [
                ndimage.map_coordinates(grid[..., i], px_coords, order=1)[0]
                for i in range(3)
            ]
            distance2d = np.sqrt(
                (func_z - mask_z) ** 2 + (func_y - mask_y) ** 2 + (func_x - mask_x) ** 2
            )
            distance3d = np.sqrt(
                (func_z - sunit_z) ** 2
                + (func_y - sunit_y) ** 2
                + (func_x - sunit_x) ** 2
            )

            self.Match.insert1(
                {
                    **key,
                    **scansetunit_keys[best_func],
                    "sunit_id": sunit_ids[best_mask],
                    "iou": best_iou,
                    "distance2d": distance2d,
                    "distance3d": distance3d,
                }
            )

            # Deactivate match
            iou_matrix[best_mask, :] = 0
            iou_matrix[:, best_func] = 0


def rename(d, **kwargs):
    """
    Rename the dictionary's key according to name mapping specified in **kwargs.
    If a key is renamed into another key that already existed, the renamed key and the
    associated value is used.

    Example:
    >> rename({'a': 5, 'b': 7}, a='x', b='y')
    {'x': 5, 'y': 7}

    >> rename({'a': 5', 'b': 7}, a='b')
    {'b': 5}

    """
    return {
        kwargs[k] if k in kwargs else k: v
        for k, v in d.items()
        if k not in kwargs.values() or k in kwargs
    }


@schema
class SourceUnitsToMatch(dj.Manual):
    """
    Fill this table with units for which the proximity cell match
    is to be computed. ProximityCellMatch will only be performed
    against these units.
    """

    definition = """
    -> ScanSet.Unit
    """


@schema
class ScansToMatch(dj.Manual):
    """
    Fill this table with source and target scans to be matched.
    ProximityCellMatch and BestProximityCellMatch are performed
    only against these pairs of scans.
    """

    definition = """
    -> ScanInfo.proj(src_session='session', src_scan_idx='scan_idx') # src scan
    -> ScanInfo  # target scan
    """


@schema
class ProximityCellMatch(dj.Computed):
    """
    Finds units in the target scan that is closest in distance to the source unit with repsect to a structural
    scan. To fill this table, you would first have to fill out two manual tables: `ScansToMatch` and `SourceUnitsToMatch`.

    Refer to the notebook `how_to_fill_proxmity_cell_match.ipynb` found in `example` directory on how to most
    efficiently fill out this table.
    """

    definition = """
    -> StackCoordinates.UnitInfo.proj(src_session='session', src_scan_idx='scan_idx', src_field='field', src_unit_id='unit_id')
    -> ScanInfo # target cells
    ---
    -> ScanSet.Unit
    match_distance: float   # distance between matched units in um
    prox_ts: TIMESTAMP    # timestamp of the processing of the particular match
    """

    @property
    def key_source(self):
        return (
            super().key_source
            & ScansToMatch
            & SourceUnitsToMatch.proj(
                src_session="session", src_scan_idx="scan_idx", src_unit_id="unit_id"
            )
        )

    def make(self, key):
        src_map = dict(
            src_session="session",
            src_scan_idx="scan_idx",
            src_field="field",
            src_unit_id="unit_id",
        )
        src_key = rename(key, **src_map)
        src_x, src_y, src_z = (StackCoordinates.UnitInfo() & src_key).fetch1(
            "stack_x", "stack_y", "stack_z"
        )

        # only consider masks classified as "soma"
        soma_units = ScanSet.Unit() * MaskClassification.Type() & 'type="soma"'
        dest_unit_keys, dest_x, dest_y, dest_z = (
            StackCoordinates.UnitInfo() & key & soma_units
        ).fetch("KEY", "stack_x", "stack_y", "stack_z", order_by="unit_id")
        if len(dest_unit_keys) == 0:
            logger.info("Target scan is not matched to this structural set")
            return

        dist = np.sqrt(
            (dest_x - src_x) ** 2 + (dest_y - src_y) ** 2 + (dest_z - src_z) ** 2
        )
        pos = dist.argmin()
        min_dist = dist[pos]

        key = dict(key, **dest_unit_keys[pos])
        key["match_distance"] = min_dist
        self.insert1(key, ignore_extra_fields=True)


@schema
class BestProximityCellMatch(dj.Computed):
    """
    Given a set of source units matched against the target scan via one or more structural scans,
    find the "concensus" match by picking the target unit that was matched most frequently across
    the structural scans.

    WARNING:

    The result of this table depends on the state of ProximityCellMatch table. For an example,
    if a source unit dict(animal_id=12345, src_session=1, src_scan_idx=15, src_unit_id=10) was matched
    against target scan dict(animal_id=12345, session=3, scan_idx=5) via three structural scans, there are
    three "matched" entries found in ProximityCellMatch. The "best" match are then assessed across the
    existing three such matches, to find the unit with most frequent match.
    If, however, you then later provide additional structural scans and/or fill out ProximityCellMatch
    such that there are now five entries for the same source unit - target scan combination,
    the result of BestProximityCellMatch that you have computed earlier will NOT reflect the result you
    would get by computing the entry again now based on five such matches.

    In other words, BestProximityCellMatch only yields the "best" match to the extent the information
    is available in ProximityCellMatch at the time this table was populated. There is no guarantee that
    the result of this table's computation will remain the best / consistent.

    If ever in doubt, you are recommended to drop the relevant best matches and recompute the entry.

    Also note that the table is filled at the level of source SCAN - target scan pair, and not the
    source scan unit - target scan pair. In other words, if you want to recompute the best match between
    a single unit in a source scan - target scan, you will have to delete ALL best match entries found
    between the source and target scan.
    """

    definition = """
    -> ScanSet.Unit.proj(src_session='session', src_scan_idx='scan_idx', src_unit_id='unit_id')
    -> ScanInfo
    ---
    -> ScanSet.Unit        # most frequently matched unit id
    match_freq:  int       # how many times it was matched
    total_stacks: int      # number of stacks used to match
    mean_distance: float   # average match distance (in um)
    best_prox_ts: TIMESTAMP    # timestamp of the processing
    """

    @property
    def key_source(self):
        src_scan = ScanInfo.proj(src_session="session", src_scan_idx="scan_idx")
        dst_scan = ScanInfo
        return (
            src_scan * dst_scan * shared.SegmentationMethod
            & ScansToMatch
            & ProximityCellMatch()
        )

    def make(self, key):
        df = pd.DataFrame((ProximityCellMatch & key).fetch())
        attrs = list(key) + ["src_unit_id"]

        def get_info(df):
            vc = df["unit_id"].value_counts()
            vals = {}
            vals["unit_id"] = int(vc.index[0])
            vals["match_freq"] = vc.iloc[0]
            vals["total_stacks"] = len(df)
            vals["mean_distance"] = (
                df.groupby("unit_id").mean().loc[vals["unit_id"]]["match_distance"]
            )
            return pd.Series(vals)

        v = df.groupby(attrs).apply(get_info).reset_index()
        self.insert(v.to_records(index=False))
