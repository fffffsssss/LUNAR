import numpy as np
import sys
sys.path.append('../')
import logging
logger = logging.getLogger()
logger.setLevel(logging.ERROR)

import ailoc.deeploc
import ailoc.common
import ailoc.simulation


def beads_stack_calibrate():
    """
    Bead stack calibration.
    """

    # load bead stack
    beads_file_name = "../datasets/demo2-exp_npc_dmo1.2/beads_step20/1.tif"

    # set psf parameters
    zernike_aber = np.array([2, -2, 0, 2, 2, 0, 3, -1, 0, 3, 1, 0, 4, 0, 0, 3, -3, 0, 3, 3, 0,
                             4, -2, 0, 4, 2, 0, 5, -1, 0, 5, 1, 0, 6, 0, 0, 4, -4, 0, 4, 4, 0,
                             5, -3, 0, 5, 3, 0, 6, -2, 0, 6, 2, 0, 7, 1, 0, 7, -1, 0, 8, 0, 0],
                            dtype=np.float32).reshape([21, 3])
    psf_params_dict = {'na': 1.5,
                       'wavelength': 670,  # unit: nm
                       'refmed': 1.406,
                       'refcov': 1.524,
                       'refimm': 1.518,
                       'zernike_mode': zernike_aber[:, 0:2],
                       'zernike_coef': zernike_aber[:, 2],
                       'objstage0': 0,
                       'pixel_size_xy': (108, 108),
                       'otf_rescale_xy': (0.5, 0.5),  # this is an empirical value, which maybe due to the pixelation
                       'npupil': 64,
                       'psf_size': 31,
                       'focus_norm': True,
                       }

    # set camera parameters
    camera_params_dict = {'camera_type': 'scmos',
                          'qe': 0.81,
                          'spurious_charge': 0.002,
                          'read_noise_sigma': 1.61,
                          'e_per_adu': 0.47,
                          'baseline': 100.0}

    # set calibration parameters
    calib_params_dict = {'z_step': 20,
                         'filter_sigma': 3,
                         'threshold_abs': 20,
                         'fit_brightest': True,}

    beads_calib_params_dict = {'beads_file_name': beads_file_name,
                               'psf_params_dict': psf_params_dict,
                               'camera_params_dict': camera_params_dict,
                               'calib_params_dict': calib_params_dict}

    ailoc.common.beads_psf_calibrate(beads_calib_params_dict, napari_plot=False)


if __name__ == "__main__":
    beads_stack_calibrate()
