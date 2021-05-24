from collections import defaultdict
from itertools import count
from operator import attrgetter
from os import path as op
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle

from ..exceptions import PipelineException

try:
    import cv2
except ImportError:
    print("Could not find cv2. You won't be able to use the pupil tracker.")

ANALOG_PACKET_LEN = 2000


class CVROIGrabber:
    start = None
    end = None
    roi = None

    def __init__(self, img):
        self.img = img
        self.draw_img = np.asarray(img / img.max(), dtype=float)
        self.mask = 1 + 0 * img
        self.exit = False
        self.r = 40
        self.X, self.Y = np.mgrid[:img.shape[0], :img.shape[1]]

    def grab(self):
        print('Contrast (std)', np.std(self.img))
        img = np.asarray(self.img / self.img.max(), dtype=float)
        cv2.namedWindow('real image')
        cv2.setMouseCallback('real image', self, 0)

        while not self.exit:
            cv2.imshow('real image', img)
            if (cv2.waitKey(0) & 0xFF) == ord('q'):
                cv2.waitKey(1)
                cv2.destroyAllWindows()
                break
        cv2.waitKey(2)

    def __call__(self, event, x, y, flags, params):
        # img = np.asarray(self.img , dtype=np.uint8)[...,None] * np.ones((1,1,3), dtype=np.uint8)
        img = np.asarray(self.img / self.img.max(), dtype=float)
        cv2.imshow('real image', self.draw_img)

        if event == cv2.EVENT_LBUTTONDOWN:
            print('Start Mouse Position: ' + str(x) + ', ' + str(y))
            self.start = np.asarray([x, y])

        elif event == cv2.EVENT_LBUTTONUP:
            self.end = np.asarray([x, y])
            x = np.vstack((self.start, self.end))
            tmp = np.hstack((x.min(axis=0), x.max(axis=0)))
            roi = np.asarray([[tmp[1], tmp[3]], [tmp[0], tmp[2]]], dtype=int) + 1
            crop = img[roi[0, 0]:roi[0, 1], roi[1, 0]:roi[1, 1]]
            crop = np.asarray(crop / crop.max(), dtype=float)
            self.roi = roi
            cv2.imshow('crop', crop)

            # m = (img * self.mask).copy() # needed for a weird reason
            self.draw_img = (img * self.mask).copy()
            cv2.rectangle(self.draw_img, tuple(self.start), tuple(self.end), (0, 255, 0), 2)

            cv2.imshow('real image', self.draw_img)
            key = (cv2.waitKey(0) & 0xFF)
            if key == ord('q'):
                cv2.destroyAllWindows()
                self.exit = True
            elif key == ord('c'):
                self.mask = 0 * self.mask + 1

        elif event == cv2.EVENT_MBUTTONDOWN:
            self.mask[(self.X - y) ** 2 + (self.Y - x) ** 2 < self.r ** 2] = 0.
            self.draw_img[(self.X - y) ** 2 + (self.Y - x) ** 2 < self.r ** 2] = 0.
            cv2.imshow('real image', self.draw_img)

            key = (cv2.waitKey(0) & 0xFF)
            if key == ord('q'):
                cv2.destroyAllWindows()
                self.exit = True
            elif key == ord('c'):
                self.mask = 0 * self.mask + 1


import math
class Point:
    """ A point in a 2-d figure. """
    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y

    def is_near(self, x, y, thresh=4):
        distance = math.sqrt((self.x - x)**2 + (self.y - y)**2)
        return distance < thresh

    def __repr__(self):
        return 'Point({}, {})'.format(self.x, self.y)


class ROISelector:
    """ Matplotlib interface to select an ROI from an image

    Arguments:
        image (np.array): A 2-d image to use for background.

    Usage:
        roi_selector = ROISelector(img)  # opens a window that lets you select an ROI
        (x1, y1), (x2, y2) = roi_selector.roi # P1 is always the upper left corner and P2 is the lower right one
    """
    def __init__(self, image):
        self.image = image
        self.point1 = None
        self.point2 = None
        self.current = None

        # Create figure
        fig = plt.figure()
        plt.imshow(image)
        plt.gca().set_aspect('equal')
        plt.gray()
        plt.title('Click and drag to select ROI. Press <ENTER> to save.')

        # Bind events
        fig.canvas.mpl_connect('button_press_event', self.on_click)
        fig.canvas.mpl_connect('motion_notify_event', self.on_move)
        fig.canvas.mpl_connect('button_release_event', self.on_release)
        fig.canvas.mpl_connect('key_press_event', self.on_press)

        plt.show(block=True)

    @property
    def roi(self):
        if self.point1 is None or self.point2 is None:
            raise ValueError('No ROI was drawn')
        else:
            points = np.sort([[self.point1.x, self.point1.y],
                              [self.point2.x, self.point2.y]], axis=0) + 0.5
            # we add 0.5 to have the upper corner be (0, 0) rather than (-0.5, -0.5)
        return tuple(points[0]), tuple(points[1])

    def on_click(self, event):
        """ Start a new ROI or modify a previously drawn ROI"""
        if event.xdata is not None and event.ydata is not None:
            first_click = self.point1 is None or self.point2 is None
            if (first_click or not (self.point1.is_near(event.xdata, event.ydata) or
                                    self.point2.is_near(event.xdata, event.ydata))):
                self.point1 = Point(event.xdata, event.ydata)
                self.point2 = Point(event.xdata, event.ydata)
                self.current = self.point2
            else: # click is close to a previous point
                self.current = (self.point2 if self.point2.is_near(event.xdata, event.ydata)
                                else self.point1)
                self.current.x = event.xdata
                self.current.y = event.ydata
            self.redraw()

    def on_move(self, event):
        """ Update the current point if it is being dragged. """
        if (self.current is not None and event.xdata is not None and
                event.ydata is not None):
            self.current.x = event.xdata
            self.current.y = event.ydata
            self.redraw()

    def on_release(self, event):
        """ Release the current point."""
        self.current = None
        self.redraw()

    def on_press(self, event):
        """ Close window if <ENTER> is pressed."""
        if event.key == 'enter':
            plt.close()

    def redraw(self):
        """ Draw points and a rectangle between them"""
        plt.gca().clear()
        plt.title('Click and drag to select ROI. Press <ENTER> to save.')
        plt.imshow(self.image)
        self.draw_rectangle(self.point1, self.point2, color=('dodgerblue' if self.current
                                                              else 'lime'))
        plt.draw()

    def draw_rectangle(self, p1, p2, color='dodgerblue'):
        low_x, high_x = (p1.x, p2.x) if p1.x <= p2.x else (p2.x, p1.x)
        low_y, high_y = (p1.y, p2.y) if p1.y <= p2.y else (p2.y, p1.y)
        plt.plot([low_x, low_x], [low_y, high_y], color=color, lw=2)
        plt.plot([high_x, high_x], [low_y, high_y], color=color, lw=2)
        plt.plot([low_x, high_x], [low_y, low_y], color=color, lw=2)
        plt.plot([low_x, high_x], [high_y, high_y], color=color, lw=2)

        plt.plot(p1.x, p1.y, 'ok', mfc='gold')
        plt.plot(p2.x, p2.y, 'ok', mfc='deeppink')


class PointLabeler:
    """ Matplotlib interface to label points in an image.

    Arguments:
        image (np.array): A 2-d image to use for background.
        percentile (float): Higher percentile used to clip the image to improve contrast.

    Usage:
        point_labeler = PointLabeler(img)  # opens a window that lets you select an ROI
        [[p1.x, p1.y], [p2.x, p2.y], ...] = point_labeler.points
    """
    def __init__(self, image, percentile=100):
        self.image = image
        self._points = []
        self.current = None
        self.percentile = percentile
        self._vmax = np.percentile(image, percentile) # vmax send to plt.imshow()

        # Create figure
        fig = plt.figure(figsize=(12, 12))
        plt.imshow(image, vmax=self._vmax)
        plt.gca().set_aspect('equal')
        plt.gray()
        plt.title('Click/drag points. Press d to delete last point, <ENTER> to save.')

        # Bind events
        fig.canvas.mpl_connect('button_press_event', self.on_click)
        fig.canvas.mpl_connect('button_release_event', self.on_release)
        fig.canvas.mpl_connect('key_press_event', self.on_press)

        plt.show(block=True)

    @property
    def points(self):
        return [[p.x + 0.5, p.y + 0.5] for p in self._points] # 0.5 to have the upper corner be (0, 0) rather than (-0.5, -0.5)

    def on_click(self, event):
        """ Create a new point or select a previous point. """
        if event.xdata is not None and event.ydata is not None:
            nearby_point = [p.is_near(event.xdata, event.ydata) for p in self._points]
            if len(self._points) == 0 or not any(nearby_point):
                new_point = Point()
                self._points.append(new_point)
                self.current = new_point
            else:
                self.current = self._points[nearby_point.index(True)]

    def on_release(self, event):
        """ Save point and release."""
        if (self.current is not None and event.xdata is not None and
                event.ydata is not None):
            self.current.x = event.xdata
            self.current.y = event.ydata
            self.current = None
            self.redraw()

    def on_press(self, event):
        """ Close window if <ENTER> is pressed."""
        if event.key == 'enter':
            plt.close()
        if event.key == 'd':
            if len(self._points) > 0:
                self._points.pop()
                self.redraw()
        if event.key == '=' or event.key == '-':
            self.percentile += (1 if event.key == '-' else -1)
            self.percentile = np.clip(self.percentile, 0, 100)
            self._vmax = np.percentile(self.image, self.percentile)
            self.redraw()

    def redraw(self):
        """ Draw the points and lines between them. """
        plt.gca().clear()
        plt.title('Click/drag points. Press d to delete last point, <ENTER> to save.')
        plt.imshow(self.image, vmax=self._vmax)

        for i, p in enumerate(self._points):
            plt.plot(p.x, p.y, 'ok', mfc='C{}'.format(i%10))
        for p1, p2 in zip(self._points[:-1], self._points[1:]):
            plt.plot([p1.x, p2.x], [p1.y, p2.y], color='lime', lw=1.5)

        plt.draw()


class PupilTracker:
    """
    Parameters:

    perc_high                    : float        # upper percentile for bright pixels
    perc_low                     : float        # lower percentile for dark pixels
    perc_weight                  : float        # threshold will be perc_weight*perc_low + (1- perc_weight)*perc_high
    relative_area_threshold      : float        # enclosing rotating rectangle has to have at least that amount of area
    ratio_threshold              : float        # ratio of major and minor radius cannot be larger than this
    error_threshold              : float        # threshold on the RMSE of the ellipse fit
    min_contour_len              : int          # minimal required contour length (must be at least 5)
    margin                       : float        # relative margin the pupil center should not be in
    contrast_threshold           : float        # contrast below that threshold are considered dark
    speed_threshold              : float        # eye center can at most move that fraction of the roi between frames
    dr_threshold                 : float        # maximally allow relative change in radius

    """

    def __init__(self, param, mask=None):
        self._params = param
        self._center = None
        self._radius = None
        self._mask = mask
        self._last_detection = 1
        self._last_ellipse = None

    @staticmethod
    def goodness_of_fit(contour, ellipse):
        center, size, angle = ellipse
        angle *= np.pi / 180
        err = 0
        for coord in contour.squeeze().astype(np.float):
            posx = (coord[0] - center[0]) * np.cos(-angle) - (coord[1] - center[1]) * np.sin(-angle)
            posy = (coord[0] - center[0]) * np.sin(-angle) + (coord[1] - center[1]) * np.cos(-angle)
            err += ((posx / size[0]) ** 2 + (posy / size[1]) ** 2 - 0.25) ** 2

        return np.sqrt(err / len(contour))

    @staticmethod
    def restrict_to_long_axis(contour, ellipse, corridor):
        center, size, angle = ellipse
        angle *= np.pi / 180
        R = np.asarray([[np.cos(-angle), - np.sin(-angle)], [np.sin(-angle), np.cos(-angle)]])
        contour = np.dot(contour.squeeze() - center, R.T)
        contour = contour[np.abs(contour[:, 0]) < corridor * ellipse[1][1] / 2]
        return (np.dot(contour, R) + center).astype(np.int32)

    def get_pupil_from_contours(self, contours, small_gray, mask, show_matching=5):
        ratio_thres = self._params['ratio_threshold']
        area_threshold = self._params['relative_area_threshold']
        error_threshold = self._params['error_threshold']
        min_contour = self._params['min_contour_len']
        margin = self._params['margin']
        speed_thres = self._params['speed_threshold']
        dr_thres = self._params['dr_threshold']
        err = np.inf
        best_ellipse = None
        best_contour = None
        kernel = np.ones((3, 3))

        results, cond = defaultdict(list), defaultdict(list)
        for j, cnt in enumerate(contours):

            mask2 = cv2.erode(mask, kernel, iterations=1)
            idx = mask2[cnt[..., 1], cnt[..., 0]] > 0
            cnt = cnt[idx]

            if len(cnt) < min_contour:  # otherwise fitEllipse won't work
                continue

            ellipse = cv2.fitEllipse(cnt)
            ((x, y), axes, angle) = ellipse
            if min(axes) == 0:  # otherwise ratio won't work
                continue
            ratio = max(axes) / min(axes)
            area = np.prod(ellipse[1]) / np.prod(small_gray.shape)
            curr_err = self.goodness_of_fit(cnt, ellipse)

            results['ratio'].append(ratio)
            results['area'].append(area)
            results['rmse'].append(curr_err)
            results['x coord'].append(x / small_gray.shape[1])
            results['y coord'].append(y / small_gray.shape[0])

            center = np.array([x / small_gray.shape[1], y / small_gray.shape[0]])
            r = max(axes)

            dr = 0 if self._radius is None else np.abs(r - self._radius) / self._radius
            dx = 0 if self._center is None else np.sqrt(np.sum((center - self._center) ** 2))

            results['dx'].append(dx)
            results['dr/r'].append(dr)
            matching_conditions = 1 * (ratio <= ratio_thres) + 1 * (area >= area_threshold) \
                                  + 1 * (curr_err < error_threshold) \
                                  + 1 * (margin < center[0] < 1 - margin) \
                                  + 1 * (margin < center[1] < 1 - margin) \
                                  + 1 * (dx < speed_thres * self._last_detection) \
                                  + 1 * (dr < dr_thres * self._last_detection)
            cond['ratio'].append(ratio <= ratio_thres)
            cond['area'].append(area >= area_threshold)
            cond['rmse'].append(curr_err < error_threshold)
            cond['x coord'].append(margin < center[0] < 1 - margin)
            cond['y coord'].append(margin < center[1] < 1 - margin)
            cond['dx'].append(dx < speed_thres * self._last_detection)
            cond['dr/r'].append(dr < dr_thres * self._last_detection)

            results['conditions'] = matching_conditions
            cond['conditions'].append(True)

            if curr_err < err and matching_conditions == 7:
                best_ellipse = ellipse
                best_contour = cnt
                err = curr_err
                cv2.ellipse(small_gray, ellipse, (0, 0, 255), 2)
            elif matching_conditions >= show_matching:
                cv2.ellipse(small_gray, ellipse, (255, 0, 0), 2)

        if best_ellipse is None:
            df = pd.DataFrame(results)
            df2 = pd.DataFrame(cond)

            print('-', end="", flush=True)
            if 'conditions' in df.columns and np.any(df['conditions'] >= show_matching):
                idx = df['conditions'] >= show_matching
                df = df[idx]
                df2 = df2[idx]
                df[df2] = np.nan
                print("\n", df, flush=True)
            self._last_detection += 1
        else:
            self._last_detection = 1

        return best_contour, best_ellipse

    _running_avg = None

    def preprocess_image(self, frame, eye_roi):
        h = int(self._params['gaussian_blur'])
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_std = np.std(gray)

        small_gray = gray[slice(*eye_roi[0]), slice(*eye_roi[1])]

        # Manual meso settins
        if 'extreme_meso' in self._params and self._params['extreme_meso']:
            c = self._params['running_avg']
            p = self._params['exponent']
            if self._running_avg is None:
                self._running_avg = np.array(small_gray / 255) ** p * 255
            else:
                self._running_avg = c * np.array(small_gray / 255) ** p * 255 + (1 - c) * self._running_avg
                small_gray = self._running_avg.astype(np.uint8)
                cv2.imshow('power', small_gray)
                # small_gray += self._running_avg.astype(np.uint8) - small_gray  # big hack
        # --- mesosetting end

        blur = cv2.GaussianBlur(small_gray, (2 * h + 1, 2 * h + 1), 0)  # play with blur

        _, thres = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return gray, small_gray, img_std, thres, blur

    @staticmethod
    def display(gray, blur, thres, eye_roi, fr_count, n_frames, ncontours=0, contour=None, ellipse=None,
                eye_center=None,
                font=None):
        if font is None:
            font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.imshow('blur', blur)
        cv2.imshow('threshold', thres)
        cv2.putText(gray, "Frames {fr_count}/{frames} | Found contours {ncontours}".format(fr_count=fr_count,
                                                                                           frames=n_frames,
                                                                                           ncontours=ncontours),
                    (10, 30), font, 1, (255, 255, 255), 2)
        # cv.drawContours(mask, contours, -1, (255), 1)
        if contour is not None and ellipse is not None and eye_center is not None:
            ellipse = list(ellipse)
            ellipse[0] = tuple(eye_center)
            ellipse = tuple(ellipse)
            cv2.drawContours(gray, [contour], 0, (255, 0, 0), 1, offset=tuple(eye_roi[::-1, 0]))
            cv2.ellipse(gray, ellipse, (0, 0, 255), 2)
            epy, epx = np.round(eye_center).astype(int)
            gray[epx - 3:epx + 3, epy - 3:epy + 3] = 0
        cv2.imshow('frame', gray)

    def track(self, videofile, eye_roi, display=False):
        contrast_low = self._params['contrast_threshold']
        mask_kernel = np.ones((3, 3))

        print("Tracking videofile", videofile)
        cap = cv2.VideoCapture(videofile)
        traces = []

        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fr_count = 0
        if self._mask is not None:
            small_mask = self._mask[slice(*eye_roi[0]), slice(*eye_roi[1])].squeeze()
        else:
            small_mask = np.ones(np.diff(eye_roi, axis=1).squeeze().astype(int), dtype=np.uint8)

        while cap.isOpened():
            if fr_count >= n_frames:
                print("Reached end of videofile ", videofile)
                break

            # --- read frame
            ret, frame = cap.read()
            fr_count += 1

            # --- if we don't get a frame, don't add any tracking results
            if not ret:
                traces.append(dict(frame_id=fr_count))
                continue

            # --- print out if there's not display
            if fr_count % 500 == 0:
                print("\tframe ({}/{})".format(fr_count, n_frames))

            # --- preprocess and treshold images
            gray, small_gray, img_std, thres, blur = self.preprocess_image(frame, eye_roi)

            # --- if contrast is too low, skip it
            if img_std < contrast_low:
                traces.append(dict(frame_id=fr_count,
                                   frame_intensity=img_std))
                print('_', end="", flush=True)
                if display:
                    self.display(gray, blur, thres, eye_roi, fr_count, n_frames)
                continue

            # --- detect contours
            ellipse, eye_center, contour = None, None, None

            if self._last_ellipse is not None:
                mask = np.zeros(small_mask.shape, dtype=np.uint8)
                cv2.ellipse(mask, tuple(self._last_ellipse), (255), thickness=cv2.FILLED)
                # cv2.drawContours(mask, [self._last_contour], -1, (255), thickness=cv2.FILLED)
                mask = cv2.dilate(mask, mask_kernel, iterations=self.dilation_iter.value)
                thres *= mask
            thres *= small_mask

            _, contours, hierarchy1 = cv2.findContours(thres.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contour, ellipse = self.get_pupil_from_contours(contours, blur, small_mask)

            self._last_ellipse = ellipse

            if contour is None:
                traces.append(dict(frame_id=fr_count, frame_intensity=img_std))
            else:
                eye_center = eye_roi[::-1, 0] + np.asarray(ellipse[0])
                self._center = np.asarray(ellipse[0]) / np.asarray(small_gray.shape[::-1])
                self._radius = max(ellipse[1])

                traces.append(dict(center=eye_center,
                                   major_r=np.max(ellipse[1]),
                                   rotated_rect=np.hstack(ellipse),
                                   contour=contour.astype(np.int16),
                                   frame_id=fr_count,
                                   frame_intensity=img_std
                                   ))
            if display:
                self.display(self._mask * gray if self._mask is not None else gray, blur, thres, eye_roi,
                             fr_count, n_frames, ellipse=ellipse,
                             eye_center=eye_center, contour=contour, ncontours=len(contours))
            if (cv2.waitKey(1) & 0xFF == ord('q')):
                raise PipelineException('Tracking aborted')

        cap.release()
        cv2.destroyAllWindows()

        return traces


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def identity(x):
    return x

def div10(x):
    return x/10

class Parameter:
    def __init__(self, name, value, min=None, max=None, log_size=None,
                 set_transform=None, get_transform=None):
        self._value = value
        self.name = name
        self.min = min
        self.max = max
        self.log_size = log_size
        self.set_transform = set_transform if set_transform is not None else identity
        self.get_transform = get_transform if get_transform is not None else identity
        self.flush_log()

    @property
    def value(self):
        return self.get_transform(self._value)

    def set(self, val):
        self._value = self.set_transform(val)
        if self.min is not None:
            self._value = max(self.min, self._value)
        if self.max is not None:
            self._value = min(self._value, self.max)
        print(self.name, 'new value:', self._value)

    def log(self, i):
        self._log[i] = self.value

    @property
    def logtrace(self):
        return np.array(self._log)

    def flush_log(self):
        if self.log_size is not None:
            self._log = [None] * self.log_size
        else:
            self._log = None


class ManualTracker:
    MAIN_WINDOW = "Main Window"
    ROI_WINDOW = "ROI"
    THRESHOLDED_WINDOW = "Thresholded"
    PROGRESS_WINDOW = "Progress"
    GRAPH_WINDOW = "Area"

    MERGE = 1
    BLOCK = 0

    DEBUG = True

    def add_track_bar(self, description, parameter, window=None):
        cv2.createTrackbar(description,
                           window if window is not None else self.MAIN_WINDOW,
                           parameter.value, parameter.max, parameter.set)

    @staticmethod
    def from_backup(file):
        trk = pickle.load(open(file, 'rb'))
        trk.register_callbacks()
        return trk

    def __init__(self, videofile):
        self.reset()

        self.videofile = videofile

        self.register_callbacks()

        self.update_frame = True  # must be true to ensure correct starting conditions
        self.contours_detected = None
        self.contours = None
        self.area = None
        self._mixing_log = None
        self._progress_len = 1600
        self._progress_height = 100
        self._width = 800
        self.window_size = 2000


        self.dilation_factor = 1.3

    def register_callbacks(self):
        cv2.namedWindow(self.MAIN_WINDOW)
        cv2.namedWindow(self.GRAPH_WINDOW)
        self.add_track_bar("mask brush size", self.brush)
        self.add_track_bar("frame tolerance", self.frame_tolerance)
        self.add_track_bar("Gaussian blur filter half width", self.blur)
        self.add_track_bar("Exponent", self.power)
        self.add_track_bar("erosion/dilation iterations", self.dilation_iter)
        self.add_track_bar("min contour length", self.min_contour_len)
        cv2.createTrackbar("10x weight of current frame in running avg.", self.MAIN_WINDOW,
                           int(self.mixing_constant.value*10), 10, self.mixing_constant.set)

        cv2.setMouseCallback(self.MAIN_WINDOW, self.mouse_callback)
        cv2.setMouseCallback(self.GRAPH_WINDOW, self.graph_mouse_callback)


    def recompute_area(self):
        print('Recomputing areas')
        assert self.contours is not None, 'contours must not be None'
        assert self.contours_detected is not None, 'contours_detected must not be None'

        self.area = np.zeros(len(self.contours))
        _, frame = self.read_frame()
        area = np.zeros(frame.shape[:2], dtype=np.uint8)
        for i, c, ok in tqdm(zip(count(), self.contours, self.contours_detected), total=len(self.area)):
            if c is None:
                self.contours_detected[i] = False
                self.area[i] = 0
            else:
                area = cv2.drawContours(area, [c], -1, (255), thickness=cv2.FILLED)
                self.area[i] = (area > 0).sum()
                area *= 0
        self.plot_area()

    def reset(self):
        self.pause = False
        self.step = 50
        self._cap = None
        self._frame_number = None
        self._n_frames = None
        self._last_frame = None

        self._mask = None
        self._merge_mask = None
        self.mask_mode = self.BLOCK

        # Parameters
        self.brush = Parameter(name='mask_brush_size', value=20, min=1, max=100)
        self.roi = Parameter(name='roi', value=None)
        self.blur = Parameter(name='gauss_blur', value=3, min=1, max=20, get_transform=int)
        self.power = Parameter(name='exponent', value=3, min=1, max=10)
        self.dilation_iter = Parameter(name='dilation_iter', value=7, min=1, max=20)
        self.min_contour_len = Parameter(name='min_contour_len', value=10, min=5, max=50)
        self.mixing_constant = Parameter(name='running_avg_mix', value=1., min=.1, max=1.,
                                         set_transform=div10)
        self.frame_tolerance = Parameter(name='frame_tolerance', value=0, min=0, max=5)

        self._skipped_frames = 0

        self.roi_start = None
        self.roi_end = None

        self.t0 = 0
        self.t1 = None
        self.scroll_window = False

        self.dilation_kernel = np.ones((3, 3))

        self.histogram_equalize = False

        self.skip = False

        self.help = True
        self._scale_factor = None
        self.dsize = None

        self._running_mean = None

        self.backup_interval = 1000
        self.backup_file = '/tmp/tracker.pkl'

        self._drag = False

        self.parameters = []
        for e in self.__dict__.values():
            if isinstance(e, Parameter):
                self.parameters.append(e)

    def set_log_size(self, n):
        for p in self.parameters:
            p.log_size = n

    def flush_parameter_log(self):
        for p in self.parameters:
            p.flush_log()

    def log_parameters(self, i):
        for p in self.parameters:
            p.log(i)


    def mouse_callback(self, event, x, y, flags, param):
        if self._scale_factor is not None:
            x, y = map(int, (i / self._scale_factor for i in (x, y)))
        if event == cv2.EVENT_LBUTTONDOWN:
            mask = self._mask if self.mask_mode == self.BLOCK else self._merge_mask
            color = (0, 0, 0) if self.mask_mode == self.BLOCK else (255, 255, 255)
            if mask is not None:
                cv2.circle(mask, (x, y), self.brush.value, color, -1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            mask = self._mask if self.mask_mode == self.BLOCK else self._merge_mask
            color = (0, 0, 0) if self.mask_mode == self.MERGE else (255, 255, 255)
            if mask is not None:
                cv2.circle(mask, (x, y), self.brush.value, color, -1)
        elif event == cv2.EVENT_MBUTTONDOWN:
            self.roi_start = (x, y)
            self._drag = True
        elif event == cv2.EVENT_MOUSEMOVE and self._drag:
            self.roi_end = (x, y)
        elif event == cv2.EVENT_MBUTTONUP:
            self.roi_end = (x, y)
            self._drag = False
            if self.roi_end[0] != self.roi_start[0] and self.roi_end[1] != self.roi_start[1]:
                x = np.vstack((self.roi_start, self.roi_end))
                tmp = np.hstack((x.min(axis=0), x.max(axis=0)))
                self.roi.set(np.asarray([[tmp[1], tmp[3]], [tmp[0], tmp[2]]], dtype=int) + 1)
            else:
                print('ROI endpoints are not different Paul! Setting ROI to None!')
                self.roi.set(None)

    def graph_mouse_callback(self, event, x, y, flags, param):
        t0, t1 = self.t0, self.t1
        dt = t1 - t0
        sanitize = lambda t: int(max(min(t, self._n_frames - 1), 0))
        if event == cv2.EVENT_MBUTTONDOWN:
            frame = sanitize(t0 + x / self._progress_len * dt)
            self.goto_frame(frame)
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.t0_tmp = sanitize(t0 + x / self._progress_len * dt)
        elif event == cv2.EVENT_LBUTTONUP:
            t1 = sanitize(t0 + x / self._progress_len * dt)
            if t1 < self.t0_tmp:
                self.t0, self.t1 = t1, self.t0_tmp
            elif self.t0_tmp == t1:
                self.t0, self.t1 = self.t0_tmp, self.t0_tmp + 1
            else:
                self.t0, self.t1 = self.t0_tmp, t1
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.del_tmp = sanitize(t0 + x / self._progress_len * dt)
        elif event == cv2.EVENT_RBUTTONUP:
            t1 = sanitize(t0 + x / self._progress_len * dt)
            if t1 < self.del_tmp:
                t0, t1 = t1, self.del_tmp
            else:
                t0, t1 = self.del_tmp, t1
            self.contours_detected[t0:t1] = False
            self.contours[t0:t1] = None

    def process_key(self, key):
        if key == ord('q'):
            return False
        elif key == ord(' '):
            self.pause = not self.pause
            return True
        elif key == ord('s'):
            self.skip = not self.skip
            return True
        elif key == ord('a'):
            self.t0, self.t1 = 0, self._n_frames
            return True
        elif key == ord('b'):
            self.goto_frame(self._frame_number - self.step)
            return True
        elif key == ord('e'):
            self.histogram_equalize = not self.histogram_equalize
            return True
        elif key == ord('r'):
            self.roi_start = None
            self.roi_end = None
            self.roi = None
            return True
        elif key == ord('c'):
            self._mask = np.ones_like(self._mask) * 255
            self._merge_mask = np.zeros_like(self._merge_mask)
        elif key == ord('h'):
            self.help = not self.help
            return True
        elif key == ord('m'):
            self.mask_mode = self.MERGE if self.mask_mode == self.BLOCK else self.BLOCK
            return True
        elif key == ord('t'):
            self.focus_window()
            return True
        elif key == ord('w'):
            self.scroll_window = ~self.scroll_window
            self.t0 = max(0, self._frame_number - self.window_size)
            self.t1 = min(self._n_frames, self.t0 + self.window_size)
            return True

        return True

    help_text = """
        KEYBOARD:
        
        q       : quits
        space   : (un)pause
        a       : reset area
        s       : toggle skip
        b       : jump back 10 frames
        r       : delete roi
        c       : delete mask
        e       : toggle histogram equalization
        h       : toggle help
        m       : toggle mask mode
        t       : focus window on cursor
        w       : toggle scrolling window
        
        MOUSE:  
        middle drag               : drag ROI
        left click                : add to mask
        right click               : delete from mask 
        middle click in area      : jump to location 
        drag and drop in area     : zoom in 
        drag and drop in area     : drop frames
        """

    def display_frame_number(self, img):
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = .6
        cv2.putText(img, "[{fr_count:05d}/{frames:05d}]".format(fr_count=self._frame_number, frames=self._n_frames),
                    (10, 30), font, fs, (255, 144, 30), 2)
        if self.contours[self._frame_number] is not None:
            cv2.putText(img, "OK", (200, 30), font, fs, (0, 255, 0), 2)
        else:
            cv2.putText(img, "NOT OK", (200, 30), font, fs, (0, 0, 255), 2)
        cv2.putText(img, "Mask Mode {}".format('MERGE' if self.mask_mode == self.MERGE else 'BLOCK'),
                    (500, 30), font,
                    fs, (0, 140, 255), 2)
        cv2.putText(img, "Skipped Frames {}/{}".format(self._skipped_frames, self.frame_tolerance.value),
                    (700, 30), font,
                    fs, (127, 255, 127), 2)
        if self.skip:
            cv2.putText(img, "Skip", (10, 70), font, fs, (0, 0, 255), 2)
        if self.help:
            y0, dy = 70, 20
            for i, line in enumerate(self.help_text.replace('\t', '    ').split('\n')):
                y = y0 + i * dy
                cv2.putText(img, line, (10, y), font, fs, (255, 144, 30), 2)

    def read_frame(self):
        if not self.pause or self.update_frame:
            if not self.update_frame:
                self._frame_number += 1

            self.update_frame = False
            ret, frame = self._cap.read()

            self._last_frame = ret, frame
            if self._mask is None:
                self._mask = np.ones_like(frame) * 255
            if self._merge_mask is None:
                self._merge_mask = np.zeros_like(frame)

            self._last_frame = ret, frame
            if ret and frame is not None:
                return ret, frame.copy()
            else:
                return ret, None
        else:
            ret, frame = self._last_frame
            return ret, frame.copy()

    def preprocess_image(self, frame):
        h = self.blur.value

        if self.power.value > 1:
            frame = np.array(frame / 255) ** self.power.value * 255
            frame = frame.astype(np.uint8)

        if self.histogram_equalize:
            cv2.equalizeHist(frame, frame)

        if self._running_mean is None or frame.shape != self._running_mean.shape:
            self._running_mean = np.array(frame)
        elif not self.pause:
            a = self.mixing_constant.value
            self._running_mean = np.uint8(a * frame + (1 - a) * self._running_mean)
            frame = np.array(self._running_mean)

        blur = cv2.GaussianBlur(frame, (2 * h + 1, 2 * h + 1), 0)
        _, thres = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = cv2.erode(thres, self.dilation_kernel, iterations=self.dilation_iter.value)
        mask = cv2.dilate(mask, self.dilation_kernel, iterations=int(self.dilation_factor * self.dilation_iter.value))

        return thres, blur, mask

    def find_contours(self, thres):
        contours, hierarchy = cv2.findContours(thres, cv2.RETR_TREE,
                                                  cv2.CHAIN_APPROX_SIMPLE)  # remove copy when cv2=3.2 is installed
        if len(contours) > 1:
            contours = [c for i, c in enumerate(contours) if hierarchy[0, i, 3] == -1]
        contours = [cv2.convexHull(c) for c in contours]

        if len(contours) > 1 and self._merge_mask is not None and np.any(self._merge_mask > 0):
            small_merge_mask = self._merge_mask[slice(*self.roi.value[0]), slice(*(self.roi.value[1] + 1)), 0]
            merge = []
            other = []
            for i in range(len(contours)):
                tmp = np.zeros_like(thres)
                cv2.drawContours(tmp, contours, i, (255), thickness=cv2.FILLED)
                cv2.bitwise_and(tmp, small_merge_mask, dst=tmp)
                if tmp.sum() > 0:
                    merge.append(contours[i])
                else:
                    other.append(contours[i])

            contours = ([cv2.convexHull(np.vstack(merge))] if len(merge) > 0 else []) + other

        contours = [c + self.roi.value[::-1, 0][None, None, :] for c in contours if len(c) >= self.min_contour_len.value]

        return contours

    def focus_window(self):
        self.t0 = max(self._frame_number - 250, 0)
        self.t1 = min(self._frame_number + 750, self._n_frames)

    def goto_frame(self, no):
        self._running_mean = None
        self._frame_number = min(max(no, 0), self._n_frames - 1)
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, self._frame_number)
        self.update_frame = True

    def normalize_graph(self, signal, min_zero=True):
        height = self._progress_height
        if not min_zero:
            v = np.abs(signal).max() + 1
            signal = (signal / v + 1) / 2
        else:
            signal = signal / (signal.max() + 1)
        return (height - signal * height).astype(int)

    def plot_area(self):
        t0, t1 = self.t0, self.t1
        dt = t1 - t0
        idx = np.linspace(t0, t1, self._progress_len, endpoint=False).astype(int)
        height = self._progress_height
        graph = (self.contours_detected[idx].astype(np.float) * 255)[None, :, None]
        graph = np.tile(graph, (height, 1, 3)).astype(np.uint8)

        area = self.normalize_graph(self.area[idx])

        detected = self.contours_detected[idx]
        for x, y1, y2, det1, det2 in zip(count(), area[:-1], area[1:], detected[:-1], detected[1:]):
            if det1 and det2:
                graph = cv2.line(graph, (x, y1), (x + 1, y2), (209, 133, 4), thickness=2)

        if t0 <= self._frame_number <= t1:
            x = int((self._frame_number - t0) / dt * self._progress_len)
            graph = cv2.line(graph, (x, 0), (x, height), (0, 255, 0), 2)
        cv2.imshow(self.GRAPH_WINDOW, graph)

    def parameter_names(self):
        return tuple(p.name for p in self.parameters)

    def parameter_iter(self):
        names = self.parameter_names()
        for frame_number, *params in zip(count(), *map(attrgetter('logtrace'), self.parameters)):
            yield dict(zip(names, params), frame_id=frame_number)

    def backup(self):
        cap = self._cap
        self._cap = None
        print('Saving tracker to', self.backup_file)
        pickle.dump(self, open(self.backup_file, 'wb'), pickle.HIGHEST_PROTOCOL)
        self._cap = cap

    def run(self):
        iterations = 0
        self._cap = cap = cv2.VideoCapture(self.videofile)

        self._n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.set_log_size(self._n_frames)
        self.flush_parameter_log()

        self._frame_number = 0
        self.update_frame = True  # ensure correct starting conditions
        self.t0 = 0
        self.t1 = self._n_frames

        if self.contours_detected is not None and self.contours_detected is not None:
            self.recompute_area()
            self.pause = True
        else:
            self.area = np.zeros(self._n_frames)
            self.contours_detected = np.zeros(self._n_frames, dtype=bool)
            self.contours = np.zeros(self._n_frames, dtype=object)
            self.contours[:] = None

        while cap.isOpened():
            if not self.pause:
                iterations += 1
                if iterations % self.backup_interval == self.backup_interval - 1:
                    self.backup()

            if self._frame_number >= self._n_frames - 1:
                if not self.pause:
                    print("Reached end of videofile. Press Q to exit. Or go back to fix stuff.", self.videofile)
                self.pause = True

            self.log_parameters(self._frame_number)
            ret, frame = self.read_frame()

            if self.scroll_window and not self.pause:
                self.t0 = min(self.t0 + 1, self._n_frames - self.scroll_window)
                self.t1 = min(self.t1 + 1, self._n_frames)

            if ret and self.roi_start is not None and self.roi_end is not None:
                cv2.rectangle(frame, self.roi_start, self.roi_end, (0, 255, 255), 2)

            if ret and not self.skip and self.roi.value is not None:
                small_gray = cv2.cvtColor(frame[slice(*self.roi.value[0]), slice(*self.roi.value[1]), :],
                                          cv2.COLOR_BGR2GRAY)

                try:
                    thres, small_gray, dilation_mask = self.preprocess_image(small_gray)
                except Exception as e:
                    print('Problems with processing reversing to frame', self._frame_number - 10, 'Please redraw ROI')
                    print('Error message is', str(e))
                    self.goto_frame(self._frame_number - 10)
                    self.roi_start = self.roi_end = self.roi = None
                    self.pause = True
                    if self.DEBUG:
                        raise
                else:
                    if self._mask is not None:
                        small_mask = self._mask[slice(*self.roi.value[0]), slice(*(self.roi.value[1] + 1)), 0]
                        cv2.bitwise_and(thres, small_mask, dst=thres)
                        cv2.bitwise_and(thres, dilation_mask, dst=thres)

                    contours = self.find_contours(thres)
                    cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
                    cv2.drawContours(small_gray, contours, -1, (127, 127, 127), 3,
                                     offset=tuple(-self.roi.value[::-1, 0]))
                    if len(contours) > 1:
                        if not self.pause:
                            self._skipped_frames += 1
                        if self._skipped_frames > self.frame_tolerance.value:
                            self.pause = True
                    elif len(contours) == 1:
                        self._skipped_frames = 0
                        area = np.zeros_like(small_gray)
                        area = cv2.drawContours(area, contours, -1, (255), thickness=cv2.FILLED,
                                                offset=tuple(-self.roi.value[::-1, 0]))
                        self.area[self._frame_number] = (area > 0).sum()
                        self.contours_detected[self._frame_number] = True
                        self.contours[self._frame_number] = contours[0]
                    else:
                        self._skipped_frames = 0

                    cv2.imshow(self.ROI_WINDOW, small_gray)
                    cv2.imshow(self.THRESHOLDED_WINDOW, thres)

            # --- plotting
            if self._merge_mask is not None:
                if np.any(self._merge_mask > 0):
                    tm = cv2.cvtColor(self._merge_mask, cv2.COLOR_BGR2GRAY)
                    _, tm = cv2.threshold(tm, 127, 255, cv2.THRESH_BINARY)
                    _, mcontours, _ = cv2.findContours(tm, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(frame, mcontours, -1, (0, 140, 255), thickness=3)

            self.display_frame_number(frame)
            cv2.bitwise_and(frame, self._mask, dst=frame)
            if self._scale_factor is None:
                self._scale_factor = self._width / frame.shape[1]
                self.dsize = tuple(int(self._scale_factor * s) for s in frame.shape[:2])[::-1]
            frame = cv2.resize(frame, self.dsize)
            cv2.imshow(self.MAIN_WINDOW, frame)
            self.plot_area()

            if not self.process_key(cv2.waitKey(5) & 0xFF):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    tracker = ManualTracker('video2.mp4')
    tracker.run()
