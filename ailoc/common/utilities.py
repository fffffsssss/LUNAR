import csv
import random
from operator import itemgetter
import numpy as np
import scipy.stats
import torch
from matplotlib import pyplot as plt
import napari
import tifffile
import os
import cv2
import sys

# import ailoc.common.local_tifffile
import ailoc.common
import ailoc.simulation


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True  # this seems affect the speed of some networks
    torch.backends.cudnn.benchmark = False


def gpu(x, data_type=torch.float32):
    """
    Transforms numpy array or torch tensor to torch.cuda.FloatTensor
    """

    if not isinstance(x, torch.Tensor):
        return torch.tensor(x, device='cuda', dtype=data_type)
    return x.to(device='cuda', dtype=data_type)


def cpu(x, data_type=np.float32):
    """
    Transforms torch tensor into numpy array
    """

    if not isinstance(x, torch.Tensor):
        return np.array(x, dtype=data_type)
    return x.cpu().detach().numpy().astype(data_type)


def softp(x):
    """
    Returns softplus(x)
    """

    return np.log(1 + np.exp(x))


def sigmoid(x):
    """
    Returns sigmoid(x)
    """

    return 1 / (1 + np.exp(-x))


def inv_softp(x):
    """
    Returns inverse softplus(x)
    """

    return np.log(np.exp(x) - 1)


def inv_sigmoid(x):
    """
    Returns inverse sigmoid(x)
    """

    return -np.log(1 / x - 1)


def torch_arctanh(x):
    """
    Returns arctanh(x) for tensor input
    """

    return 0.5 * torch.log(1 + x) - 0.5 * torch.log(1 - x)


def torch_softp(x):
    """
    Returns softplus(x) for tensor input
    """

    return torch.log(1 + torch.exp(x))


def flip_filt(filt):
    """
    Returns filter flipped over x and y dimension
    """

    return np.ascontiguousarray(filt[..., ::-1, ::-1])


def get_bg_stats_gamma(images, percentile=10, plot=False, xlim=None, floc=0):
    """Infers the parameters of a gamma distribution that fit the background of SMLM recordings.
    Identifies the darkest pixels from the averaged images as background and fits a gamma distribution to the histogram of intensity values.

    Args:
        images (np.ndarray): 3D array of recordings
        percentile (float): Percentile between 0 and 100. Sets the percentage of pixels that are assumed to only containg background activity (i.e. no fluorescent signal)
        plot (bool): If true produces a plot of the histogram and fit
        xlim (list of float): Sets xlim of the plot
        floc (float): Baseline for the gamma fit. Equal to fitting gamma to (x - floc)

    Returns:
        (float, float): (Mean, scale) parameter of the gamma fit
    """

    # ensure positive
    ind = np.where(images <= 0)
    images[ind] = 1

    # get the positions where the mean intensity is below the percentile
    map_empty = np.where(images.mean(0) < np.percentile(images.mean(0), percentile))
    pixel_vals = images[:, map_empty[0], map_empty[1]].reshape(-1)
    # fit the gamma distribution, return the alpha and scale=1/beta
    fit_alpha, fit_loc, fit_beta = scipy.stats.gamma.fit(pixel_vals, floc=floc)

    if plot:
        plt.figure(constrained_layout=True)
        if xlim is None:
            low, high = pixel_vals.min(), pixel_vals.max()
        else:
            low, high = xlim[0], xlim[1]

        _ = plt.hist(pixel_vals, bins=np.linspace(low, high), histtype='step', label='data')
        _ = plt.hist(np.random.gamma(shape=fit_alpha, scale=fit_beta, size=len(pixel_vals)) + floc,
                     bins=np.linspace(low, high), histtype='step', label='fit')
        plt.xlim(low, high)
        plt.legend()
        # plt.tight_layout()
        plt.show()
    return fit_alpha * fit_beta, fit_beta  # return the expectation and scale


def get_bg_stats_gauss(images, percentile=10, plot=False):
    """Infers the parameters of a gauss distribution that fit the background of SMLM recordings.
    Identifies the darkest pixels from the averaged images as background.

    Args:
        images (np.ndarray): 3D array of recordings
        percentile (float): Percentile between 0 and 100. Sets the percentage of pixels that are assumed to only containg background activity (i.e. no fluorescent signal)
        plot (bool): If true produces a plot of the histogram and fit

    Returns:
        (float, float): (bg_min, bg_max) background parameters
    """

    # get the positions where the mean intensity is below the percentile
    roi_idx = np.where(images.mean(0) < np.percentile(images.mean(0), percentile))
    pixel_vals = images[:, roi_idx[0], roi_idx[1]].reshape(-1)

    # fit the gauss distribution
    result = scipy.stats.norm.fit(pixel_vals)

    if plot:
        plt.figure(constrained_layout=True)
        _ = plt.hist(pixel_vals,
                     bins=np.linspace(pixel_vals.min(), pixel_vals.max(), 50),
                     # histtype='step',
                     alpha=0.5,
                     label='bg data')
        _ = plt.hist(np.random.normal(loc=result[0], scale=result[1], size=len(pixel_vals)),
                     bins=np.linspace(pixel_vals.min(), pixel_vals.max()),
                     # histtype='step',
                     alpha=0.5,
                     label='bg fit')
        plt.legend()
        plt.show()

    bg_range = tuple(np.clip([result[0] - 2 * result[1], result[0] + 2 * result[1]],
                             a_min=0, a_max=None))
    return bg_range

def get_gain_bg_empirical(images,
                          camera_params_dict,
                          adjust_gain=True,
                          percentile=50,
                          plot_show=True):
    """
    This is an empirical method to estimate gain and training background range from data with uneven background,
    use the 1% darkest pixels to estimate the variance/mean ratio as the gain, and adjust the e_per_adu to make the
    variance/mean ratio to be 1.0 for Poisson assumption, a little mismatch is allowed.
    Then use the pixels with intensity higher than the percentile to estimate the background range.
    The estimated bg range maybe a little higher than the real one on uniform background data, but it is more robust
    as the network will not predict too many false positive signals.

    Args:
        images (np.ndarray): 3D array of recordings in photon counts
        camera_params_dict (dict): dict containing the camera parameters.
        adjust_gain (bool): If true, adjust the e_per_adu to make the variance/mean ratio to be 1.0 for Poisson assumption
        percentile (float): Percentile between 0 and 100.
            Sets the percentage of pixels that are assumed to have signals,
            and the training background range is estimated from these pixels,
            the background estimated from the rest is often lower and may cause false positives.
        plot_show (bool): If true produces a plot of the histogram and fit

    Returns:
        (float, float, any): (bg_min, bg_max) background parameters
    """
    n, h, w = images.shape
    batch_size = max((1000 * 256 ** 2) // (h * w), 1)  # A good default for many systems, adjust if needed

    camera_calib = ailoc.simulation.instantiate_camera(camera_params_dict)

    # --- Step 1: Calculate Mean Image and Pixel Mask in Chunks ---
    # This is a critical step to avoid loading the full dataset at once.
    # We will compute the mean image incrementally.
    mean_image_photon = np.zeros((h, w), dtype=np.float32)

    # Process images in batches to compute the mean image
    num_batches = (n + batch_size - 1) // batch_size
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, n)
        batch = images[start_idx:end_idx]

        batch_photon = ailoc.common.cpu(camera_calib.backward(torch.tensor(batch.astype(np.float32))))
        mean_image_photon += batch_photon.mean(axis=0) * (batch.shape[0] / n)

    # --- Step 2: Adjust Gain (if required) ---
    e_per_adu_new = camera_calib.e_per_adu

    if adjust_gain:
        print('Estimating gain...')

        # Get the pixel indices for gain estimation
        min_idx = np.where(mean_image_photon < np.quantile(mean_image_photon, 0.005))

        pix_mean_list = []
        pix_var_list = []
        # divide these time traces (pixels) by time windows
        batch_size = 200
        num_batches = (n + batch_size - 1) // batch_size
        # Iterate over batches again to calculate statistics
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, n)
            batch = images[start_idx:end_idx]

            # Convert to photons and electrons for this batch only
            batch_photon = ailoc.common.cpu(camera_calib.backward(torch.tensor(batch.astype(np.float32))))
            batch_electron = camera_calib.qe * batch_photon

            # Collect pixel values for gain estimation
            pix_vals = batch_electron[:, min_idx[0], min_idx[1]]
            pix_mean_list.append(pix_vals.mean(axis=0))
            pix_var_list.append(pix_vals.var(axis=0))

        pix_mean = np.concatenate(pix_mean_list)
        pix_var = np.concatenate(pix_var_list)

        if isinstance(camera_calib, ailoc.simulation.EMCCD):
            enf_sq = 2.0
            rn_var_input = (camera_calib.read_noise_sigma / camera_calib.em_gain) ** 2
            pix_gain = (pix_var/enf_sq - rn_var_input) / pix_mean
        else:  # sCMOS or other cameras
            pix_gain = ((pix_var - camera_calib.read_noise_sigma ** 2) / pix_mean)

        used_idx = np.logical_and(pix_gain > pix_gain.mean() - 2 * pix_gain.std(),
                                  pix_gain < pix_gain.mean() + 2 * pix_gain.std())
        est_gain = pix_gain[used_idx].mean()

        print(f'The variance/mean ratio of data is estimated as {est_gain:.2f} using the provided QE and e_per_adu.')

        if est_gain > 1.1 or (est_gain < 0.9 and est_gain > 0):
            pix_mean_used = pix_mean[used_idx]
            pix_var_used = pix_var[used_idx]
            e_per_adu_new = ((pix_mean_used.mean() +
                              np.sqrt(pix_mean_used.mean() ** 2 - 4 * pix_var_used.mean() *
                                      (est_gain * pix_mean_used.mean() - pix_var_used.mean()))) / (
                                         2 * pix_var_used.mean())
                             * camera_calib.e_per_adu)
            e_per_adu_new = np.around(e_per_adu_new, decimals=2)
            print(
                f'This might be unreliable, automatically change the e_per_adu from {camera_calib.e_per_adu:.2f} to {e_per_adu_new:.2f} to make the variance/mean ratio 1.0 for Poisson noise assumption.')
            mean_image_photon *= (e_per_adu_new / camera_calib.e_per_adu)
            camera_calib.e_per_adu = e_per_adu_new

    # --- Step 3: Get Background Range ---
    # Apply the updated calibration and process in batches again
    sample_mask = np.where(mean_image_photon > np.percentile(mean_image_photon, percentile))

    pixel_vals_list = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, n)
        batch = images[start_idx:end_idx]

        # Apply the updated calibration to the current batch
        batch_photon = ailoc.common.cpu(camera_calib.backward(torch.tensor(batch.astype(np.float32))))

        # Average the batch and get pixel values for fitting
        batch_avg = batch_photon.mean(axis=0)
        pixel_vals_list.append(batch_avg[sample_mask[0], sample_mask[1]])

    pixel_vals = np.concatenate(pixel_vals_list)

    # Fit the Gauss distribution
    result = scipy.stats.norm.fit(pixel_vals)
    bg_max = max(result[0], 20.0)  # ensure at least 20
    bg_r = np.clip(max(3 * result[1], bg_max / 2), 20.0, 200.0)  # ensure between 20 and 200
    bg_min = max(bg_max - bg_r, 0.0)
    bg_range = (float(bg_min), float(bg_max))
    print(f'Estimated bg_range: {bg_range}')

    # --- Step 4: Plotting (if required) ---
    fig = plt.figure(figsize=(12, 8), constrained_layout=True, dpi=300)
    # fig = plt.figure()
    gs = fig.add_gridspec(2, 3)

    if adjust_gain:
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.imshow(mean_image_photon < np.quantile(mean_image_photon, 0.001))
        ax0.set_xlabel('Pixels')
        ax0.set_ylabel('Pixels')
        ax0.set_title('Pixels used to estimate gain(var/mean)')

        ax1 = fig.add_subplot(gs[0, 1])
        ax1.hist(pix_gain[used_idx],
                 bins=np.linspace(pix_gain[used_idx].min(), pix_gain[used_idx].max(), 50),
                 alpha=0.5,
                 label='0.5% pixels var/mean')
        ax1.plot(est_gain, 1, '*', markersize=20, label=f'Estimated gain: {est_gain:.1f}')
        ax1.legend()
        ax1.set_xlabel('Variance/mean ratio')
        ax1.set_ylabel('Counts')
    ax2 = fig.add_subplot(gs[1, 0])
    plt.colorbar(mappable=ax2.imshow(mean_image_photon, cmap='turbo'), ax=ax2, fraction=0.046, pad=0.04)
    ax2.set_title('Mean image')
    ax2.set_xlabel('Pixels')
    ax2.set_ylabel('Pixels')
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.imshow(mean_image_photon > np.percentile(mean_image_photon, percentile))
    ax3.set_title(f'Masked area for BG fit \n(mean image>{percentile}%)')
    ax3.set_xlabel('Pixels')
    ax3.set_ylabel('Pixels')
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.hist(pixel_vals,
             density=True,
             bins=20,
             alpha=0.5,
             label='Masked pixels (BG + signals)')
    ax4.hist(np.random.uniform(low=bg_range[0], high=bg_range[1], size=len(pixel_vals)),
             density=True,
             bins=20,
             alpha=0.5,
             label=f'Training BG range ({bg_range[0]:.1f}, {bg_range[1]:.1f})')
    ax4.legend()
    ax4.set_xlabel('Pixel value (photons)')
    ax4.set_ylabel('Normalized counts')
    if plot_show:
        plt.show()
        fig_dict = None
    else:
        fig_dict = {'figure': fig, 'title': 'Estimated bg_range'}

    return bg_range, e_per_adu_new, fig_dict


def get_photon_range(images, camera_params_dict, psf_size, sampler_params_dict, plot_show=True):
    """
    Estimate the training photon range from experimental data with reduced memory usage
    and improved speed using a moving average for background subtraction.
    """
    print('-' * 200)
    print('Estimating photon range from cropped molecules, '
          'this might be overestimated for high density, '
          'please visually check training data to avoid large mismatch')

    # --- Initialization ---
    camera_calib = ailoc.simulation.instantiate_camera(camera_params_dict)
    num_images, img_height, img_width = images.shape

    assert psf_size <= min(img_height, img_width), \
        f"PSF size ({psf_size}) is larger than the image size! Please check parameters."

    # --- Parameters for Peak Finding and ROI Extraction ---
    max_roi_num = 20000
    dof_range = (sampler_params_dict['z_range'][1] - sampler_params_dict['z_range'][0]) / 1000
    dog_sigma = max(2, int(dof_range * 2))
    find_max_kernel = dog_sigma + 1
    edge_dist = psf_size // 2
    window_size = 50  # For local temporal background estimation

    # --- Pre-allocate array for ROIs in digital units ---
    sparse_rois_digital = np.zeros((max_roi_num, psf_size, psf_size), dtype=np.float32)
    roi_count = 0

    print(f'Using psf_size: {psf_size}, dog_sigma: {dog_sigma}, find_max_kernel: {find_max_kernel}')

    # 1. Initialize the sum and window size for the first frame's window
    start_idx = 0
    end_idx = min(window_size + 1, num_images)
    # Use a high-precision float for the sum to avoid overflow/precision issues
    current_sum = np.sum(images[start_idx:end_idx], axis=0, dtype=np.float64)
    current_window_len = end_idx - start_idx

    for frame_idx, image_digital in enumerate(images):
        # 2. Update the moving sum efficiently for subsequent frames
        if frame_idx > 0:
            # Subtract the frame that just left the window's trailing edge
            frame_to_remove_idx = frame_idx - window_size - 1
            if frame_to_remove_idx >= 0:
                current_sum -= images[frame_to_remove_idx]
                current_window_len -= 1

            # Add the frame that just entered the window's leading edge
            frame_to_add_idx = frame_idx + window_size
            if frame_to_add_idx < num_images:
                current_sum += images[frame_to_add_idx]
                current_window_len += 1

        # 3. Calculate the local mean from the running sum
        local_mean_digital = current_sum / current_window_len
        image_nobg_digital = np.clip(image_digital - local_mean_digital, 0, None)

        # 4. Find and filter peaks (this part remains the same)
        peaks = ailoc.common.extract_smlm_peaks(
            image_nobg=image_nobg_digital,
            dog_sigma=(dog_sigma, dog_sigma),
            find_max_thre=0.3,
            find_max_kernel=(find_max_kernel, find_max_kernel),
        )

        if len(peaks) > 0:
            peaks = ailoc.common.remove_border_peaks(peaks, edge_dist + 1, image_nobg_digital.shape)

        if len(peaks) > 0:
            tmp_sparse_peaks, _ = ailoc.common.remove_close_peaks(peaks, np.hypot(psf_size, psf_size))

            # 5. Extract ROIs and fill the pre-allocated array
            for peak in tmp_sparse_peaks:
                if roi_count >= max_roi_num:
                    break
                start_row, start_col = peak[0] - edge_dist, peak[1] - edge_dist
                roi_tmp = image_nobg_digital[start_row : start_row + psf_size, start_col : start_col + psf_size]
                sparse_rois_digital[roi_count] = roi_tmp
                roi_count += 1

        if roi_count >= max_roi_num:
            print(f"\nReached max ROI count ({max_roi_num}) at frame {frame_idx}.")
            break

    # --- Post-Loop Processing ---
    if roi_count == 0:
        raise ValueError("No sparse ROIs were found. Check peak finding parameters or image data.")
    sparse_rois_digital = sparse_rois_digital[:roi_count]
    print(f'Found {roi_count} ROIs. Converting to photon counts...')

    rois_tensor = torch.from_numpy(sparse_rois_digital.astype(np.float32))
    sparse_rois = ailoc.common.cpu(camera_calib.backward(rois_tensor + camera_calib.baseline))

    # --- Distribution Fitting (Unchanged) ---
    sum_vals = np.squeeze(sparse_rois.sum(axis=(-1, -2)))
    photon_range_limit = (100, 300000)

    loc_exp, scale_exp = scipy.stats.expon.fit(sum_vals)
    mu_gauss, std_gauss = scipy.stats.norm.fit(sum_vals)
    a_gamma, loc_gamma, scale_gamma = scipy.stats.gamma.fit(sum_vals)

    ks_stat_exp, _ = scipy.stats.kstest(sum_vals, 'expon', args=(loc_exp, scale_exp))
    ks_stat_gauss, _ = scipy.stats.kstest(sum_vals, 'norm', args=(mu_gauss, std_gauss))
    ks_stat_gamma, _ = scipy.stats.kstest(sum_vals, 'gamma', args=(a_gamma, loc_gamma, scale_gamma))

    fits = {'exp': ks_stat_exp, 'gauss': ks_stat_gauss, 'gamma': ks_stat_gamma}
    best_fit_name = min(fits, key=fits.get)

    if best_fit_name == 'exp':
        photon_range_max = min(loc_exp + 2 * scale_exp, photon_range_limit[1])
    elif best_fit_name == 'gauss':
        photon_range_max = min(mu_gauss + 2 * std_gauss, photon_range_limit[1])
    else:  # gamma
        photon_range_max = min(scipy.stats.gamma.ppf(0.95, a_gamma, loc=loc_gamma, scale=scale_gamma), photon_range_limit[1])

    photon_range_min = max(photon_range_max / 20, photon_range_limit[0])
    photon_range_new = (float(photon_range_min), float(photon_range_max))
    print(f'Estimated photon_range: {photon_range_new}')

    # --- Plotting ---
    # Create the figure and GridSpec
    fig = plt.figure(figsize=(12, 6), dpi=300, constrained_layout=True)
    # fig = plt.figure()
    gs = fig.add_gridspec(1, 2)

    # plot example ROIs
    example_indices = random.sample(range(sparse_rois.shape[0]),
                                    min(25, sparse_rois.shape[0]))
    example_rois = sparse_rois[example_indices]
    gs00 = gs[0, 0].subgridspec(len(example_rois) // 5, 5)
    for i in range(len(example_rois) // 5):
        for j in range(5):
            ax = fig.add_subplot(gs00[i, j])
            ax.imshow(example_rois[i * 5 + j], cmap='turbo')
            ax.axis('off')
    # set the title
    fig.suptitle('Example ROIs', x=0.3, y=0.95)

    # plot histogram of ROI photons
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.hist(sum_vals,
             bins=np.linspace(sum_vals.min(), sum_vals.max(), 50),
             density=True,
             alpha=0.5,
             label='Summed ROI photon distribution')
    # plot photon range on it
    ax1.axvline(photon_range_new[0], color='r', linestyle='--',
                label=f'Training photon range ({photon_range_new[0]:.1f}, {photon_range_new[1]:.1f})')
    ax1.axvline(photon_range_new[1], color='r', linestyle='--')

    ax1.set_xlabel('Summed ROI photons')
    ax1.set_ylabel('Normalized counts')

    plt.legend()
    if plot_show:
        plt.show()
        fig_dict = None
    else:
        fig_dict = {'figure': fig, 'title': 'Estimated photon'}

    return photon_range_new, fig_dict


def get_mean_percentile(images, percentile=10):
    """
    Returns the mean of the pixels at where their mean values are less than the given percentile of the average image

    Args:
        images (np.ndarray): 3D array of recordings
        percentile (float): Percentile between 0 and 100. Used to calculate the mean of the percentile of the images
    """

    idx_2d = np.where(images.mean(0) < np.percentile(images.mean(0), percentile))
    pixel_vals = images[:, idx_2d[0], idx_2d[1]]

    return pixel_vals.mean()


def get_window_map(img, winsize=40, percentile=20):
    """Helper function

    Parameters
    ----------
    images: array
        3D array of recordings
    percentile: float
        Percentile between 0 and 100. Sets the percentage of pixels that are assumed to only containg background activity (i.e. no fluorescent signal)
    plot: bool
        If true produces a plot of the histogram and fit
    xlim: list of floats
        Sets xlim of the plot
    floc: float
        Baseline for the the gamma fit. Equal to fitting gamma to (x - floc)

    Returns
    -------
    binmap: array
        Mean and scale parameter of the gamma fit
    """

    img = img.mean(0)  # 按第一维求平均,得到[64 64]
    res = np.zeros([int(img.shape[0] - winsize), int(img.shape[1] - winsize)])  # [64-40，64-40]的零矩阵
    for i in range(res.shape[0]):  # 0-24
        for j in range(res.shape[1]):
            res[i, j] = img[i:i + int(winsize), j:j + int(winsize)].mean()  # 以i j出发求[40 40]区域内的平均值
    thresh = np.percentile(res, percentile)  # 从小到大，第percentile%的值，也就是还有percentile%比这个值小
    binmap = np.zeros_like(res)
    binmap[res > thresh] = 1  # 图像中intensity大于20%的都设为1，应该是表示该处有分子荧光
    return binmap


def get_pixel_truth(eval_csv, ind, field_size, pixel_size):
    """
    draw a pixel-wise binary map of the groudtruth
    """

    eval_list = []
    if isinstance(eval_csv, str):
        with open(eval_csv, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                if 'truth' not in row[0]:
                    eval_list.append([float(r) for r in row])
    else:
        for r in eval_csv:
            eval_list.append([i for i in r])
    eval_list = sorted(eval_list, key=itemgetter(1))  # csv文件按frame升序来排列

    molecule_list = []
    for i in range(len(eval_list)):
        if eval_list[i][1] == ind:
            molecule_list.append([round(eval_list[i][2] / pixel_size[0]), round(eval_list[i][3] / pixel_size[1])])
        if eval_list[i][1] > ind:
            break

    truth_map = np.zeros([round(field_size[0] / pixel_size[0]), round(field_size[1] // pixel_size[1])])
    for molecule in molecule_list:
        truth_map[molecule[0], molecule[1]] = 1

    return truth_map


def read_first_size_gb_tiff(image_path, size_gb=4):
    # with ailoc.common.local_tifffile.TiffFile(image_path, is_ome=False) as tif:
    with tifffile.TiffFile(image_path, is_ome=False) as tif:
        total_shape = tif.series[0].shape
        # occu_mem = total_shape[0] * total_shape[1] * total_shape[2] * 16 / (1024 ** 3) / 8
        occu_mem = tif.series[0].size * tif.series[0].dtype.itemsize / (1024 ** 3)  # GBytes
        if occu_mem < size_gb:
            index_img = total_shape[0]
        else:
            index_img = int(size_gb / occu_mem * total_shape[0])
        images = tif.asarray(key=range(0, index_img), series=0)
    print('-' * 200)
    print(f"read first {images.shape} images")
    return images


def find_file(name, path_list):
    for path in path_list:
        for root, dirs, files in os.walk(path):
            if name in files:
                return os.path.join(root, name)


def viewdata_napari(*args):
    viewer = napari.view_image(cpu(args[0]), colormap='turbo')
    for i in range(1, len(args)):
        viewer.add_image(cpu(args[i]), colormap='turbo')
    napari.run()


def cmpdata_napari(data1, data2):
    assert data1.shape == data2.shape, "data1 and data2 must have the same shape"
    n_dim = len(data1.shape)
    width = data1.shape[-1]
    pad_width = [(0,0) for i in range(n_dim-1)]
    pad_width.append((0, int(0.05 * width)))
    data1 = np.pad(cpu(data1), tuple(pad_width), constant_values=np.nan)
    data2 = np.pad(cpu(data2), tuple(pad_width), constant_values=np.nan)
    data3 = np.concatenate((data1, data2, data1-data2), axis=-1)
    viewer = napari.view_image(data3, colormap='turbo')
    napari.run()


def fig2data(fig):
    """
    fig = plt.figure()
    image = fig2data(fig)
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    import PIL.Image as Image
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tostring())
    image = np.asarray(image)
    return image


def print_learning_params(psf_params_dict, camera_params_dict, sampler_params_dict):
    print('-' * 70, 'learning parameters', '-' * 100)
    for params_dict in [psf_params_dict, camera_params_dict, sampler_params_dict]:
        for keys in params_dict.keys():
            if keys == 'zernike_mode':
                params = params_dict[keys].transpose()
                print(f"{keys}:\n {params}")
            elif keys == 'zernike_coef':
                params = np.around(params_dict[keys], decimals=1)
                print(f"{keys}:\n {params}")
            else:
                params = params_dict[keys]
                print(f"{keys}: {params}")


class TrainLogger(object):
    def __init__(self, filename="logfile.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


if __name__ == "__main__":
    # Generate a figure with matplotlib</font>
    figure = plt.figure()
    plot = figure.add_subplot(111)

    # draw a cardinal sine plot
    x = np.arange(1, 100, 0.1)
    y = np.sin(x) / x
    plot.plot(x, y)
    plt.show()
    ##
    image = fig2data(figure)
    cv2.imshow('image', image)
    cv2.waitKey(0)