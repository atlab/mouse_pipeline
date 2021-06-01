"""Interface to the CaImAn package (https://github.com/simonsfoundation/CaImAn)."""
import numpy as np
import multiprocessing as mp
from caiman import components_evaluation
from caiman.utils import visualization
from caiman.source_extraction.cnmf import map_reduce, initialization, pre_processing, \
                                          merging, spatial, temporal, deconvolution
from .caiman_stats import df_percentile
from scipy.ndimage import percentile_filter
import glob, os, sys, time


def log(*messages):
    """ Simple logging function."""
    formatted_time = "[{}]".format(time.ctime())
    print(formatted_time, *messages, flush=True, file=sys.__stdout__)


def mute_function(f):
    """ Decorator to ignore any standard output of the function."""
    def wrapper(*args, **kwargs):
        try:
            sys.stdout = open(os.devnull, 'w')
            return f(*args, **kwargs)
        finally:
            sys.stdout = sys.__stdout__ # go back to normal (even after exceptions)
    return wrapper


@mute_function
def extract_masks(scan, mmap_scan, num_components=200, num_background_components=1,
                  merge_threshold=0.8, init_on_patches=True, init_method='greedy_roi',
                  soma_diameter=(14, 14), snmf_alpha=None, patch_size=(50, 50),
                  proportion_patch_overlap=0.2, num_components_per_patch=5,
                  num_processes=8, num_pixels_per_process=5000, fps=15):
    """ Extract masks from multi-photon scans using CNMF.

    Uses constrained non-negative matrix factorization to find spatial components (masks)
    and their fluorescence traces in a scan. Default values work well for somatic scans.

    Performed operations are:
        [Initialization on full image | Initialization on patches -> merge components] ->
        spatial update -> temporal update -> merge components -> spatial update ->
        temporal update

    :param np.array scan: 3-dimensional scan (image_height, image_width, num_frames).
    :param np.memmap mmap_scan: 2-d scan (image_height * image_width, num_frames)
    :param int num_components: An estimate of the number of spatial components in the scan
    :param int num_background_components: Number of components to model the background.
    :param int merge_threshold: Maximal temporal correlation allowed between the activity
        of overlapping components before merging them.
    :param bool init_on_patches: If True, run the initialization methods on small patches
        of the scan rather than on the whole image.
    :param string init_method: Initialization method for the components.
        'greedy_roi': Look for a gaussian-shaped patch, apply rank-1 NMF, store
            components, calculate residual scan and repeat for num_components.
        'sparse_nmf': Regularized non-negative matrix factorization (as impl. in sklearn)
    :param (float, float) soma_diameter: Estimated neuron size in y and x (pixels). Used
        in'greedy_roi' initialization to search for neurons of this size.
    :param int snmf_alpha: Regularization parameter (alpha) for sparse NMF (if used).
    :param (float, float) patch_size: Size of the patches in y and x (pixels).
    :param float proportion_patch_overlap: Patches are sampled in a sliding window. This
        controls how much overlap is between adjacent patches (0 for none, 0.9 for 90%).
    :param int num_components_per_patch: Number of components per patch (used if
        init_on_patches=True)
    :param int num_processes: Number of processes to run in parallel. None for as many
        processes as available cores.
    :param int num_pixels_per_process: Number of pixels that a process handles each
        iteration.
    :param fps: Frame rate. Used for temporal downsampling and to remove bad components.

    :returns: Weighted masks (image_height x image_width x num_components). Inferred
        location of each component.
    :returns: Denoised fluorescence traces (num_components x num_frames).
    :returns: Masks for background components (image_height x image_width x
        num_background_components).
    :returns: Traces for background components (image_height x image_width x
        num_background_components).
    :returns: Raw fluorescence traces (num_components x num_frames). Fluorescence of each
        component in the scan minus activity from other components and background.

    ..warning:: The produced number of components is not exactly what you ask for because
        some components will be merged or deleted.
    ..warning:: Better results if scans are nonnegative.
    """
    # Get some params
    image_height, image_width, num_frames = scan.shape

    # Start processes
    log('Starting {} processes...'.format(num_processes))
    pool = mp.Pool(processes=num_processes)

    # Initialize components
    log('Initializing components...')
    if init_on_patches:
        # TODO: Redo this (per-patch initialization) in a nicer/more efficient way

        # Make sure they are integers
        patch_size = np.array(patch_size)
        half_patch_size = np.int32(np.round(patch_size / 2))
        num_components_per_patch = int(round(num_components_per_patch))
        patch_overlap = np.int32(np.round(patch_size * proportion_patch_overlap))

        # Create options dictionary (needed for run_CNMF_patches)
        options = {'patch_params': {'ssub': 'UNUSED.', 'tsub': 'UNUSED', 'nb': num_background_components,
                                    'only_init': True, 'skip_refinement': 'UNUSED.',
                                    'remove_very_bad_comps': False}, # remove_very_bads_comps unnecesary (same as default)
                   'preprocess_params': {'check_nan': False}, # check_nan is unnecessary (same as default value)
                   'spatial_params': {'nb': num_background_components}, # nb is unnecessary, it is pased to the function and in init_params
                   'temporal_params': {'p': 0, 'method': 'UNUSED.', 'block_size': 'UNUSED.'},
                   'init_params': {'K': num_components_per_patch, 'gSig': np.array(soma_diameter)/2,
                                   'gSiz': None, 'method': init_method, 'alpha_snmf': snmf_alpha,
                                   'nb': num_background_components, 'ssub': 1, 'tsub': max(int(fps / 2), 1),
                                   'options_local_NMF': 'UNUSED.', 'normalize_init': True,
                                   'rolling_sum': True, 'rolling_length': 100, 'min_corr': 'UNUSED',
                                   'min_pnr': 'UNUSED', 'deconvolve_options_init': 'UNUSED',
                                   'ring_size_factor': 'UNUSED', 'center_psf': 'UNUSED'},
                                   # gSiz, ssub, tsub, options_local_NMF, normalize_init, rolling_sum unnecessary (same as default values)
                   'merging' : {'thr': 'UNUSED.'}}

        # Initialize per patch
        res = map_reduce.run_CNMF_patches(mmap_scan.filename, (image_height, image_width, num_frames),
                                          options, rf=half_patch_size, stride=patch_overlap,
                                          gnb=num_background_components, dview=pool)
        initial_A, initial_C, YrA, initial_b, initial_f, pixels_noise, _ = res

        # Merge spatially overlapping components
        merged_masks = ['dummy']
        while len(merged_masks) > 0:
            res = merging.merge_components(mmap_scan, initial_A, initial_b, initial_C,
                                           initial_f, initial_C, pixels_noise,
                                           {'p': 0, 'method': 'cvxpy'}, spatial_params='UNUSED',
                                           dview=pool, thr=merge_threshold, mx=np.Inf)
            initial_A, initial_C, num_components, merged_masks, S, bl, c1, neurons_noise, g = res

        # Delete log files (one per patch)
        log_files = glob.glob('caiman*_LOG_*')
        for log_file in log_files:
            os.remove(log_file)
    else:
        from scipy.sparse import csr_matrix
        if init_method == 'greedy_roi':
            res = _greedyROI(scan, num_components, soma_diameter, num_background_components)
            log('Refining initial components (HALS)...')
            res = initialization.hals(scan, res[0].reshape([image_height * image_width, -1], order='F'),
                                      res[1], res[2].reshape([image_height * image_width, -1], order='F'),
                                      res[3], maxIter=3)
            initial_A, initial_C, initial_b, initial_f = res
        else:
            print('Warning: Running sparse_nmf initialization on the entire field of view '
                  'takes a lot of time.')
            res = initialization.initialize_components(scan, K=num_components, nb=num_background_components,
                                                       method=init_method, alpha_snmf=snmf_alpha)
            initial_A, initial_C, initial_b, initial_f, _ = res
        initial_A = csr_matrix(initial_A)
    log(initial_A.shape[-1], 'components found...')

    # Remove bad components (based on spatial consistency and spiking activity)
    log('Removing bad components...')
    good_indices, _ = components_evaluation.estimate_components_quality(initial_C, scan,
        initial_A, initial_C, initial_b, initial_f, final_frate=fps, r_values_min=0.7,
        fitness_min=-20, fitness_delta_min=-20, dview=pool)
    initial_A = initial_A[:, good_indices]
    initial_C = initial_C[good_indices]
    log(initial_A.shape[-1], 'components remaining...')

    # Estimate noise per pixel
    log('Calculating noise per pixel...')
    pixels_noise, _ = pre_processing.get_noise_fft_parallel(mmap_scan, num_pixels_per_process, pool)

    # Update masks
    log('Updating masks...')
    A, b, C, f = spatial.update_spatial_components(mmap_scan, initial_C, initial_f, initial_A, b_in=initial_b,
                                                   sn=pixels_noise, dims=(image_height, image_width),
                                                   method='dilate', dview=pool,
                                                   n_pixels_per_process=num_pixels_per_process,
                                                   nb=num_background_components)

    # Update traces (no impulse response modelling p=0)
    log('Updating traces...')
    res = temporal.update_temporal_components(mmap_scan, A, b, C, f, nb=num_background_components,
                                              block_size=10000, p=0, method='cvxpy', dview=pool)
    C, A, b, f, S, bl, c1, neurons_noise, g, YrA, _ = res


    # Merge components
    log('Merging overlapping (and temporally correlated) masks...')
    merged_masks = ['dummy']
    while len(merged_masks) > 0:
        res = merging.merge_components(mmap_scan, A, b, C, f, S, pixels_noise, {'p': 0, 'method': 'cvxpy'},
                                       'UNUSED', dview=pool, thr=merge_threshold, bl=bl, c1=c1,
                                       sn=neurons_noise, g=g)
        A, C, num_components, merged_masks, S, bl, c1, neurons_noise, g = res

    # Refine masks
    log('Refining masks...')
    A, b, C, f = spatial.update_spatial_components(mmap_scan, C, f, A, b_in=b, sn=pixels_noise,
                                                   dims=(image_height, image_width),
                                                   method='dilate', dview=pool,
                                                   n_pixels_per_process=num_pixels_per_process,
                                                   nb=num_background_components)

    # Refine traces
    log('Refining traces...')
    res = temporal.update_temporal_components(mmap_scan, A, b, C, f, nb=num_background_components,
                                              block_size=10000, p=0, method='cvxpy', dview=pool)
    C, A, b, f, S, bl, c1, neurons_noise, g, YrA, _ = res

    # Removing bad components (more stringent criteria)
    log('Removing bad components...')
    good_indices, _ = components_evaluation.estimate_components_quality(C + YrA, scan, A,
        C, b, f, final_frate=fps, r_values_min=0.8, fitness_min=-40, fitness_delta_min=-40,
        dview=pool)
    A = A.toarray()[:, good_indices]
    C = C[good_indices]
    YrA = YrA[good_indices]
    log(A.shape[-1], 'components remaining...')

    # Stop processes
    log('Done.')
    pool.close()

    # Get results
    masks = A.reshape((image_height, image_width, -1), order='F') # h x w x num_components
    traces = C  # num_components x num_frames
    background_masks = b.reshape((image_height, image_width, -1), order='F') # h x w x num_components
    background_traces = f  # num_background_components x num_frames
    raw_traces = C + YrA  # num_components x num_frames

    # Rescale traces to match scan range
    scaling_factor = np.sum(masks**2, axis=(0, 1)) / np.sum(masks, axis=(0, 1))
    traces = traces * np.expand_dims(scaling_factor, -1)
    raw_traces = raw_traces * np.expand_dims(scaling_factor, -1)
    masks = masks / scaling_factor
    background_scaling_factor = np.sum(background_masks**2, axis=(0, 1)) / np.sum(background_masks,
                                                                                  axis=(0,1))
    background_traces = background_traces * np.expand_dims(background_scaling_factor, -1)
    background_masks = background_masks / background_scaling_factor

    return masks, traces, background_masks, background_traces, raw_traces


def _save_as_memmap(scan, base_name='caiman', chunk_size=5000):
    """Save the scan as a memory mapped file as expected by caiman

    :param np.array scan: Scan to save shaped (image_height, image_width, num_frames)
    :param string base_name: Base file name for the scan. No underscores.
    :param int chunk_size: Write the mmap_scan chunk frames at a time. Memory efficient.

    :returns: Filename of the mmap file.
    :rtype: string
    """
    # Get some params
    image_height, image_width, num_frames = scan.shape
    num_pixels = image_height * image_width

    # Build filename
    filename = '{}_d1_{}_d2_{}_d3_1_order_C_frames_{}_.mmap'.format(base_name, image_height,
                                                                    image_width, num_frames)

    # Create memory mapped file
    mmap_scan = np.memmap(filename, mode='w+', shape=(num_pixels, num_frames), dtype=np.float32)
    for i in range(0, num_frames, chunk_size):
        chunk = scan[..., i: i + chunk_size].reshape((num_pixels, -1), order='F')
        mmap_scan[:, i: i + chunk_size] = chunk
    mmap_scan.flush()

    return mmap_scan


def _greedyROI(scan, num_components=200, neuron_size=(11, 11),
               num_background_components=1):
    """ Initialize components by searching for gaussian shaped, highly active squares.
    #one by one by moving a gaussian window over every pixel and
    taking the highest activation as the center of the next neuron.

    :param np.array scan: 3-dimensional scan (image_height, image_width, num_frames).
    :param int num_components: The desired number of components.
    :param (float, float) neuron_size: Expected size of the somas in pixels (y, x).
    :param int num_background_components: Number of components that model the background.
    """
    from scipy import ndimage

    # Get some params
    image_height, image_width, num_frames = scan.shape

    # Get the gaussian kernel
    gaussian_stddev = np.array(neuron_size) / 4 # entire neuron in four standard deviations
    gaussian_kernel = _gaussian2d(gaussian_stddev)

    # Create residual scan (scan minus background)
    residual_scan = scan - np.mean(scan, axis=(0, 1)) # image-wise brightness
    background = ndimage.gaussian_filter(np.mean(residual_scan, axis=-1), neuron_size)
    residual_scan -= np.expand_dims(background, -1)

    # Create components
    masks = np.zeros([image_height, image_width, num_components], dtype=np.float32)
    traces = np.zeros([num_components, num_frames], dtype=np.float32)
    mean_frame = np.mean(residual_scan, axis=-1)
    for i in range(num_components):

        # Get center of next component
        neuron_locations = ndimage.gaussian_filter(mean_frame, gaussian_stddev)
        y, x = np.unravel_index(np.argmax(neuron_locations), [image_height, image_width])

        # Compute initial trace (bit messy because of edges)
        half_kernel = np.fix(np.array(gaussian_kernel.shape) / 2).astype(np.int32)
        big_yslice = slice(max(y - half_kernel[0], 0), y + half_kernel[0] + 1)
        big_xslice = slice(max(x - half_kernel[1], 0), x + half_kernel[1] + 1)
        kernel_yslice = slice(max(0, half_kernel[0] - y),
                              None if image_height > y + half_kernel[0] else image_height - y - half_kernel[0] - 1)
        kernel_xslice = slice(max(0, half_kernel[1] - x),
                              None if image_width > x + half_kernel[1] else image_width - x - half_kernel[1] - 1)
        cropped_kernel = gaussian_kernel[kernel_yslice, kernel_xslice]
        trace = np.average(residual_scan[big_yslice, big_xslice].reshape(-1, num_frames),
                           weights=cropped_kernel.ravel(), axis=0)

        # Get mask and trace using 1-rank NMF
        half_neuron = np.fix(np.array(neuron_size) / 2).astype(np.int32)
        yslice = slice(max(y - half_neuron[0], 0), y + half_neuron[0] + 1)
        xslice = slice(max(x - half_neuron[1], 0), x + half_neuron[1] + 1)
        mask, trace = _rank1_NMF(residual_scan[yslice, xslice], trace)

        # Update residual scan
        neuron_activity = np.expand_dims(mask, -1) * trace
        residual_scan[yslice, xslice] -= neuron_activity
        mean_frame[yslice, xslice] = np.mean(residual_scan[yslice, xslice], axis=-1)

        # Store results
        masks[yslice, xslice, i] = mask
        traces[i] = trace

    # Create background components
    residual_scan += np.mean(scan, axis=(0, 1)) # add back overall brightness
    residual_scan += np.expand_dims(background, -1) # and background
    if num_background_components == 1:
        background_masks = np.expand_dims(np.mean(residual_scan, axis=-1), axis=-1)
        background_traces = np.expand_dims(np.mean(residual_scan, axis=(0, 1)), axis=0)
    else:
        from sklearn.decomposition import NMF
        print("Warning: Fitting more than one background component uses scikit-learn's "
              "NMF and may take some time.""")
        model = NMF(num_background_components, random_state=123, verbose=True)

        flat_masks = model.fit_transform(residual_scan.reshape(-1, num_frames))
        background_masks = flat_masks.reshape([image_height, image_width, -1])
        background_traces = model.components_

    return masks, traces, background_masks, background_traces


def _gaussian2d(stddev, truncate=4):
    """ Creates a 2-d gaussian kernel truncated at 4 standard deviations (8 in total).

    :param (float, float) stddev: Standard deviations in y and x.
    :param float truncate: Number of stddevs at each side of the kernel.

    ..note:: Kernel sizes will always be odd.
    """
    from matplotlib import mlab
    half_kernel = np.round(stddev * truncate) # kernel_size = 2 * half_kernel + 1
    y, x = np.meshgrid(np.arange(-half_kernel[0], half_kernel[0] + 1),
                       np.arange(-half_kernel[1], half_kernel[1] + 1))
    kernel = mlab.bivariate_normal(x, y, sigmay=stddev[0], sigmax=stddev[1])
    return kernel


# Based on caiman.source_extraction.cnmf.initialization.finetune()
def _rank1_NMF(scan, trace, num_iterations=5):
    num_frames = scan.shape[-1]
    for i in range(num_iterations):
        mask = np.maximum(np.dot(scan, trace), 0)
        mask  = mask * np.sum(mask) / np.sum(mask ** 2)
        trace = np.average(scan.reshape(-1, num_frames), weights=mask.ravel(), axis=0)
    return mask, trace


def deconvolve(trace, AR_order=2):
    """ Deconvolve traces using noise constrained deconvolution (Pnevmatikakis et al., 2016)

    :param np.array trace: 1-d array (num_frames) with the fluorescence trace.
    :param int AR_order: Order of the autoregressive process used to model the impulse
        response function, e.g., 0 = no modelling; 2 = model rise plus exponential decay.

    :returns: Deconvolved spike trace.
    :returns: AR coefficients (AR_order) that model the calcium response:
            c(t) = c(t-1) * AR_coeffs[0] + c(t-2) * AR_coeffs[1] + ...
    """
    _, _, _, AR_coeffs, _, spike_trace, _ = deconvolution.constrained_foopsi(trace,
        p=AR_order, method='cvxpy', bas_nonneg=False, fudge_factor=0.96)
        # fudge_factor is a regularization term

    return spike_trace, AR_coeffs


def deconvolve_detrended(trace, scan_fps, detrend_period=600, AR_order=2):
    """Same as the the `deconvolve` method, except that the fluorescence trace is detrended 
    before autoregressive modeling

    :param np.array trace: 1-d array (num_frames) with the fluorescence trace.
    :param float scan_fps: fps of the scan
    :param float detrend_period: number of seconds over which percentiles are computed
    :param int AR_order: Order of the autoregressive process used to model the impulse
        response function, e.g., 0 = no modelling; 2 = model rise plus exponential decay.

    :returns: Deconvolved spike trace.
    :returns: AR coefficients (AR_order) that model the calcium response:
            c(t) = c(t-1) * AR_coeffs[0] + c(t-2) * AR_coeffs[1] + ...
    """
    detrend_window = int(round(detrend_period * scan_fps))
    n_chunks = len(trace) // detrend_window
    if detrend_window > 0 and n_chunks > 0:
        chunks_len = n_chunks * detrend_window
        trace_chunks = trace[:chunks_len].reshape(-1, detrend_window)
        data_prct = df_percentile(trace_chunks, axis=1)[0].mean()
        trace = trace - percentile_filter(trace, data_prct, detrend_window)

    _, _, _, AR_coeffs, _, spike_trace, _ = deconvolution.constrained_foopsi(trace,
        p=AR_order, method='cvxpy', bas_nonneg=False, fudge_factor=0.96)

    return spike_trace, AR_coeffs


def get_centroids(masks):
    """ Calculate the centroids of each mask (calls caiman's plot_contours).

    :param np.array masks: Masks (image_height x image_width x num_components)

    :returns: Centroids (num_components x 2) in y, x pixels of each component.
    """
    # Reshape masks
    image_height, image_width, num_components = masks.shape
    masks = masks.reshape(-1, num_components, order='F')

    # Get centroids
    fake_background = np.empty([image_height, image_width]) # needed for plot contours
    coordinates = visualization.plot_contours(masks, fake_background)
    import matplotlib.pyplot as plt; plt.close()
    centroids = np.array([coordinate['CoM'] for coordinate in coordinates])

    return centroids


def classify_masks(masks, soma_diameter=(12, 12)):
    """ Uses a convolutional network to predict the probability per mask of being a soma.

    :param np.array masks: Masks (image_height x image_width x num_components)

    :returns: Soma predictions (num_components).
    """
    # Reshape masks
    image_height, image_width, num_components = masks.shape
    masks = masks.reshape(-1, num_components, order='F')

    # Prepare input
    from scipy.sparse import coo_matrix
    masks = coo_matrix(masks)
    soma_radius = np.int32(np.round(np.array(soma_diameter)/2))

    model_path = '/data/pipeline/python/pipeline/data/cnn_model'
    probs, _ = components_evaluation.evaluate_components_CNN(masks, (image_height, image_width),
                                                             soma_radius, model_name=model_path)

    return probs[:, 1]



















# Legacy: Used in preprocess.ExtractRaw
def demix_and_deconvolve_with_cnmf(scan, num_components=200, AR_order=2,
                                   merge_threshold=0.8, num_processes=20,
                                   num_pixels_per_process=5000, block_size=10000,
                                   num_background_components=4, init_method='greedy_roi',
                                   soma_radius=(5, 5), snmf_alpha=None,
                                   init_on_patches=False, patch_downsampling_factor=None,
                                   percentage_of_patch_overlap=None):
    """ Extract spike train activity from multi-photon scans using CNMF.

    Uses constrained non-negative matrix factorization to find neurons/components
    (locations) and their fluorescence traces (activity) in a timeseries of images, and
    deconvolves them using an autoregressive model of the calcium impulse response
    function. See Pnevmatikakis et al., 2016 for details.

    Default values work alright for somatic images.

    :param np.array scan: 3-dimensional scan (image_height, image_width, num_frames).
    :param int num_components: An estimate of neurons/spatial components in the scan.
    :param int AR_order: Order of the autoregressive process used to model the impulse
        response function, e.g., 0 = no modelling; 2 = model rise plus exponential decay.
    :param int merge_threshold: Maximal temporal correlation allowed between activity of
        overlapping components before merging them.
    :param int num_processes: Number of processes to run in parallel. None for as many
        processes as available cores.
    :param int num_pixels_per_process: Number of pixels that a process handles each
        iteration.
    :param int block_size: 'number of pixels to process at the same time for dot product'
    :param int num_background_components:  Number of background components to use.
    :param string init_method: Initialization method for the components.
        'greedy_roi':Look for a gaussian-shaped patch, apply rank-1 NMF, store components,
            calculate residual scan and repeat for num_components.
        'sparse_nmf': Regularized non-negative matrix factorization (as impl. in sklearn)
        'local_nmf': ...
    :param (float, float) soma_radius: Estimated neuron radius (in pixels) in y and x.
        Used in'greedy_roi' initialization to define the size of the gaussian window.
    :param int snmf_alpha: Regularization parameter (alpha) for the sparse NMF (if used).
    :param bool init_on_patches: If True, run the initialization methods on small patches
        of the scan rather than on the whole image.
    :param int patch_downsampling_factor: Division to the image dimensions to obtain patch
        dimensions, e.g., if original size is 256 and factor is 10, patches will be 26x26
    :param int percentage_of_patch_overlap: Patches are sampled in a sliding window. This
        controls how much overlap is between adjacent patches (0 for none, 0.9 for 90%)

    :returns Location matrix (image_height x image_width x num_components). Inferred
        location of each component.
    :returns Activity matrix (num_components x num_frames). Inferred fluorescence traces
         (spike train convolved with the fitted impulse response function).
    :returns: Inferred location matrix for background components (image_height x
         image_width x num_background_components).
    :returns: Inferred activity matrix for background components (image_height x
        image_width x num_background_components).
    :returns: Raw fluorescence traces (num_components x num_frames) obtained from the
        scan minus activity from background and other components.
    :returns: Spike matrix (num_components x num_frames). Deconvolved spike activity.
    :returns: Autoregressive process coefficients (num_components x AR_order) used to
        model the calcium impulse response of each component:
            c(t) = c(t-1) * AR_coeffs[0] + c(t-2) * AR_coeffs[1] + ...

    ..note:: Based on code provided by Andrea Giovanucci.
    ..note:: The produced number of components is not exactly what you ask for because
        some components will be merged or deleted.
    ..warning:: Computation- and memory-intensive for big scans.
    """
    import caiman
    from caiman.source_extraction.cnmf import cnmf

    # Save as memory mapped file in F order (that's how caiman wants it)
    mmap_filename = _save_as_memmap(scan, base_name='/tmp/caiman', order='F').filename

    # 'Load' scan
    mmap_scan, (image_height, image_width), num_frames = caiman.load_memmap(mmap_filename)
    images = np.reshape(mmap_scan.T, (num_frames, image_height, image_width), order='F')

    # Start the ipyparallel cluster
    client, direct_view, num_processes = caiman.cluster.setup_cluster(
        n_processes=num_processes)

    # Optionally, run the initialization method in small patches to initialize components
    initial_A = None
    initial_C = None
    initial_f = None
    if init_on_patches:
        # Calculate patch size (only square patches allowed)
        bigger_dimension = max(image_height, image_width)
        smaller_dimension = min(image_height, image_width)
        patch_size = bigger_dimension / patch_downsampling_factor
        patch_size = min(patch_size, smaller_dimension) # if bigger than small dimension

        # Calculate num_components_per_patch
        num_nonoverlapping_patches = (image_height/patch_size) * (image_width/patch_size)
        num_components_per_patch = num_components / num_nonoverlapping_patches
        num_components_per_patch = max(num_components_per_patch, 1) # at least 1

        # Calculate patch overlap in pixels
        overlap_in_pixels = patch_size * percentage_of_patch_overlap

        # Make sure they are integers
        patch_size = int(round(patch_size))
        num_components_per_patch = int(round(num_components_per_patch))
        overlap_in_pixels = int(round(overlap_in_pixels))

        # Run CNMF on patches (only for initialization, no impulse response modelling p=0)
        model = cnmf.CNMF(num_processes, only_init_patch=True, p=0,
                          rf=int(round(patch_size / 2)), stride=overlap_in_pixels,
                          k=num_components_per_patch, merge_thresh=merge_threshold,
                          method_init=init_method, gSig=soma_radius,
                          alpha_snmf=snmf_alpha, gnb=num_background_components,
                          n_pixels_per_process=num_pixels_per_process,
                          block_size=block_size, check_nan=False, dview=direct_view,
                          method_deconvolution='cvxpy')
        model = model.fit(images)

        # Delete log files (one per patch)
        log_files = glob.glob('caiman*_LOG_*')
        for log_file in log_files:
            os.remove(log_file)

        # Get results
        initial_A = model.A
        initial_C = model.C
        initial_f = model.f

    # Run CNMF
    model = cnmf.CNMF(num_processes, k=num_components, p=AR_order,
                      merge_thresh=merge_threshold, gnb=num_background_components,
                      method_init=init_method, gSig=soma_radius, alpha_snmf=snmf_alpha,
                      n_pixels_per_process=num_pixels_per_process, block_size=block_size,
                      check_nan=False, dview=direct_view, Ain=initial_A, Cin=initial_C,
                      f_in=initial_f, method_deconvolution='cvxpy')
    model = model.fit(images)

    # Get final results
    location_matrix = model.A  # pixels x num_components
    activity_matrix = model.C  # num_components x num_frames
    background_location_matrix = model.b  # pixels x num_background_components
    background_activity_matrix = model.f  # num_background_components x num_frames
    spikes = model.S  # num_components x num_frames, spike_ traces
    raw_traces = model.C + model.YrA  # num_components x num_frames
    AR_coefficients = model.g  # AR_order x num_components

    # Reshape spatial matrices to be image_height x image_width x num_frames
    new_shape = (image_height, image_width, -1)
    location_matrix = location_matrix.toarray().reshape(new_shape, order='F')
    background_location_matrix = background_location_matrix.reshape(new_shape, order='F')
    AR_coefficients = np.array(list(AR_coefficients))  # unwrapping it (num_components x 2)

    # Stop ipyparallel cluster
    client.close()
    caiman.stop_server()

    # Delete memory mapped scan
    os.remove(mmap_filename)

    return (location_matrix, activity_matrix, background_location_matrix,
            background_activity_matrix, raw_traces, spikes, AR_coefficients)