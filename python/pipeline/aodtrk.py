import datajoint as dj
import pandas as pd
from . import aodpre
import warnings
from IPython import embed
import glob
import numpy as np
import dateutil.parser
from . import utils
import cv2
import os,shutil
try:
    from pupil_tracking.pupil_tracker_aod import PupilTracker
except ImportError:
    warnings.warn("Failed to import pupil_tacking library. You won't be able to populate trk.EyeFrame")

schema = dj.schema('pipeline_aod_pupiltracking', locals())


@schema
class TrackInfo(dj.Imported):
    definition = """
    # machine independent path of eye videos
    
    ->aodpre.Scan
    ---
    base_video_path: varchar(100) # base path of the video
    """

    def _make_tuples(self, key):
        print("key = ", key)
        # embed()
        path = (aodpre.Scan() & key).fetch1['hdf5_file']
        path.replace('\\', '/')

        # words = path.split('\\')
        # if len(words) == 1:

        words = words[0].split('/')
        i = words.index('Mouse')
        ymd = words[i + 3].split('_')[0]
        hms = words[i + 3].split('_')[1].replace("-", ":")
        time_hdf5 = dateutil.parser.parse("{ymd} {hms}".format(ymd=ymd, hms=hms))

        # time_str = words[i+3].split('_')[1].split('-')
        # time_hdf5 = int(time_str[0])*10000 + int(time_str[1])*100 + int(time_str[2])
        folders = glob.glob(r"/m/Mouse/{f1}/20*".format(f1=words[i + 1]))
        time_coll = []
        time_diff = []
        for name in folders:
            ymd = name.split('/')[4].split('_')[0]
            hms = name.split('/')[4].split('_')[1].replace("-", ":")
            # t = name.split('/')[-1].split('_')[1].split('-')
            # time = int(t[0])*10000 + int(t[1])*100 + int(t[2])
            time = dateutil.parser.parse("{ymd} {hms}".format(ymd=ymd, hms=hms))
            time_coll.append(time)
            diff = abs((time_hdf5 - time).total_seconds())
            time_diff.append(diff)

        time_diff = np.asarray(time_diff)
        fo = folders[np.argmin(abs(time_diff))]
        avi_path = glob.glob(r"{fo}/*.avi".format(fo=fo))
        assert len(avi_path) == 1, "Found 0 or more than 1 videos: {videos}".format(videos=str(avi_path))
        key['base_video_path'] = avi_path[0]
        self.insert1(key)

    def get_frames(self, key):
        # path = (aodpre.Scan() & key).fetch1['hdf5_file']
        video_file = (self & key).fetch1['base_video_path']
        # embed()
        import cv2
        cap = cv2.VideoCapture(video_file)
        fr_count = 0
        while cap.isOpened():
            fr_count += 1
            ret, frame = cap.read()
            if fr_count == 1000:
                return frame


@schema
class Roi(dj.Manual):
    definition = """
    # table that stores the correct ROI of the Eye in the video
    ->TrackInfo
    ---
    x_roi_min                     : int                         # x coordinate of roi
    y_roi_min                     : int                         # y coordinate of roi
    x_roi_max                     : int                         # x coordinate of roi
    y_roi_max                     : int                         # y coordinate of roi
    """

    def dump_video(self):
        print("Entered dump")
        vid_coll = self.fetch.as_dict()
        for video in vid_coll:
            video_path = (TrackInfo() & video).fetch1['base_video_path']
            if not (EyeFrame() & video):
                print("EyeFrame for (mouse_id,scan_idx)= (", video['mouse_id'], video['scan_idx'],
                      ") not found. Please populate EyeFrame before dumping video")
            else:
                print("Dumping video for parameters (mouse_id,scan_idx) = (", video['mouse_id'], video['scan_idx'], ")")
                try:
                    shutil.rmtree("temp_images")
                except:
                    pass
                # print("Debug2")
                os.makedirs("temp_images")
                cap = cv2.VideoCapture(video_path)
                length_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fr_count = 0
                # print("Debug 1")
                while cap.isOpened():
                    ret, frame = cap.read()
                    fr_count += 1
                    # print("Debug 3")
                    if fr_count % 1000 == 0:
                        print("Processing frame = ", fr_count, "/", length_video)
                        # break
                    if fr_count % 6 == 0:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        if fr_count >= (length_video-10):
                            print("Video: ", video_path, " is over")
                            break

                        data = (EyeFrame.Detection() & video & dict(frame=fr_count)).fetch.as_dict()
                        if data:
                            data = data[0]
                            ellipse = ((int(data['pupil_x']),int(data['pupil_y'])),(int(data['pupil_r_minor']),int(data['pupil_r_major'])),
                                       int(data['pupil_angle']))
                            _ = cv2.ellipse(gray, ellipse, (0, 0, 255), 2)
                        name = "temp_images/img%06d.png" % (fr_count,)
                        cv2.imwrite(name, gray)
                print("Dumped images for parameters (mouse_id,scan_idx) = (", video['mouse_id'], video['scan_idx'], ")")
                print("Stitching images into a video")
                file_name = "video_%s_%s.mp4" % (video['mouse_id'], video['scan_idx'])
                try:
                    os.remove(file_name)
                except:
                    pass
                command = "ffmpeg -f image2 -pattern_type glob -i 'temp_images/*.png' "+file_name
                # command = "ffmpeg -framerate 5 -i temp_images\%06d.png -c:v libx264 -r 5 -pix_fmt yuv420p "+file_name
                os.system(command)

                # embed()
                try:
                    shutil.rmtree("temp_images")
                except:
                    pass




@schema
class ParamEyeFrame(dj.Lookup):
    definition = """
    # table that stores the paths for the params for pupil_tracker
    pupil_tracker_param_id                      : int            # id for param collection
    ---
    convex_weight_high = Null                : float        # parameter for weighting higher pixel intensity value to decide threshold. condition = if (maxr < radius1 - p * (pow(pow((center1[0] - full_patch_size / 2), 2) + pow((center1[1] - full_patch_size / 2), 2), po)) and (center1[1] > ll * full_patch_size) and (center1[1] < rl * full_patch_size) and (center1[0] > ll * full_patch_size) and (center1[0] < rl * full_patch_size) and (radius1 > mir) and (radius1 < mar) and len(contours1[j]) >= 5):
    convex_weight_low = Null                 : float        # parameter for weighting lower pixel intensity for threshold
    thres_perc_high = Null                   : float        # percentile parameter to pick most bright pixel value
    thres_perc_low = Null                    : float        # percentile parameter to pick least bright pixel value
    pupil_left_limit = Null                  : float        # parameter in percentage to restrict pupil centre in roi
    pupil_right_limit = Null                 : float        # parameter in percentage to restrict pupil centre in roi
    min_radius = Null                        : float        # parameter to restrict pupil radius while selecting pupil from multiple contours
    max_radius = Null                        : float        # parameter to restrict pupil radius while selecting pupil from multiple contours
    centre_dislocation_penalty               : float        # parameter for penalty as to force selection of contour which is in the centre as pupil
    distance_sq_pow                          : float        # parameter for selecting method of calculating distance for penalty

    """

    contents = [
        {'pupil_tracker_param_id': 0, 'convex_weight_high': 0.5, 'convex_weight_low': 0.5, 'thres_perc_high': 99,
         'distance_sq_pow': 1,
         'thres_perc_low': 1, 'pupil_left_limit': 0.2, 'pupil_right_limit': 0.8, 'min_radius': 5, 'max_radius': 180,
         'centre_dislocation_penalty': 0.001},
        {'pupil_tracker_param_id': 1, 'convex_weight_high': 0.75, 'convex_weight_low': 0.25, 'thres_perc_high': 97,
         'distance_sq_pow': 0.5,
         'thres_perc_low': 3, 'pupil_left_limit': 0.2, 'pupil_right_limit': 0.8, 'min_radius': 5, 'max_radius': 180,
         'centre_dislocation_penalty': 0.05}
    ]


@schema
class EyeFrame(dj.Computed):
    definition = """
    # eye tracking info for each frame of a movie
    -> Roi
    -> ParamEyeFrame
    frame                       : int                           # frame number in movie
    ---
    eye_frame_ts=CURRENT_TIMESTAMP    : timestamp               # automatic
    """

    @property
    def populated_from(self):
        return Roi()

    def _make_tuples(self, key):
        # embed()
        param = (ParamEyeFrame() & 'pupil_tracker_param_id=0').fetch.as_dict()[0]
        key['pupil_tracker_param_id'] = param['pupil_tracker_param_id']
        video_path = (TrackInfo() & key).fetch1['base_video_path']
        eye_roi = (Roi() & key).fetch1['x_roi_min', 'y_roi_min', 'x_roi_max', 'y_roi_max']
        param['centre_dislocation_penalty'] = 0.001
        param['distance_sq_pow'] = 1

        tr = PupilTracker(param)
        trace = tr.track_without_svm(video_path, eye_roi)

        # CODE to insert data after tracking
        print("Tracking complete... Now inserting data to datajoint")
        efd = EyeFrame.Detection()
        # embed()
        for index, data in trace.iterrows():
            key['frame'] = index
            self.insert1(key)
            if pd.notnull(data['pupil_x']):
                values = data.to_dict()
                values.update(key)
                # embed()
                efd.insert1(values)

    class Detection(dj.Part):
        definition = """
        # eye frames with detected eye
        ->EyeFrame
        ---
        pupil_x                     : float                         # pupil x position
        pupil_y                     : float                         # pupil y position
        pupil_r_minor               : float                         # pupil radius minor axis
        pupil_r_major               : float                         # pupil radius major axis
        pupil_angle                 : float                         # angle of major axis vs. horizontal axis in radians
        pupil_x_std                 : float                         # pupil x position std
        pupil_y_std                 : float                         # pupil y position std
        pupil_r_minor_std            : float                         # pupil radius minor axis std
        pupil_r_major_std           : float                         # pupil radius major axis std
        pupil_angle_std             : float                         # angle of major axis vs. horizontal axis in radians
        intensity_std               : float                         # standard deviation of the ROI pixel values
        """
