import deprecated
import torch
import numpy as np
import torch.nn as nn

import ailoc.simulation
import ailoc.common


class Simulator:
    """
    Simulator class for generating simulated images of single molecules.
    """

    def __init__(self, psf_params, camera_params, sampler_params):
        """
        Args:
            psf_params (dict): parameters for the PSF
            camera_params (dict): parameters for the camera
            sampler_params (dict): parameters for the molecule sampler
        """

        # self.psf_model = ailoc.simulation.VectorPSFCUDA(psf_params)
        self.psf_model = ailoc.simulation.VectorPSFTorch(psf_params, data_type=torch.float32)
        if camera_params['camera_type'].upper() == 'EMCCD':
            self.camera = ailoc.simulation.EMCCD(camera_params)
            self.mol_sampler = ailoc.simulation.MoleculeSampler(sampler_params,
                                                                self.psf_model.zernike_coef_map)
        elif camera_params['camera_type'].upper() == 'SCMOS':
            self.camera = ailoc.simulation.SCMOS(camera_params)
            self.mol_sampler = ailoc.simulation.MoleculeSampler(sampler_params,
                                                                self.psf_model.zernike_coef_map,
                                                                self.camera.read_noise_map)
        elif camera_params['camera_type'].upper() == 'IDEA':
            self.camera = ailoc.simulation.IdeaCamera()
            self.mol_sampler = ailoc.simulation.MoleculeSampler(sampler_params,
                                                                self.psf_model.zernike_coef_map)
        else:
            raise NotImplementedError('Camera type not supported.')

    def sample_training_data(self, batch_size, context_size, iter_train, robust_scale=False):
        """
        Sample a batch of training data. All frames in a batch unit with the context_size share the same
            background and serve as temporal context for each other.

        Args:
            batch_size (int): batch size
            context_size (int): the number of frames to be used as temporal context, the training data has
                shape (batch_size, context_size, H, W)
            iter_train (int): the number of training iterations, used for sequentially select the
                sub-fov to simulate images
            robust_scale (bool): whether to randomly scale the data to break the strict Poisson distribution
                as a data augmentation method

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple):
                a batch of simulated images and corresponding ground truth
        """

        p_map_sample, xyzph_map_sample, bg_map_sample, curr_sub_fov_xy, zernike_coefs, \
        p_map_gt, xyzph_array_gt, mask_array_gt = self.mol_sampler.sample_for_train(batch_size,
                                                                                    context_size,
                                                                                    self.psf_model,
                                                                                    iter_train,)

        data = self.gen_noiseless_data(self.psf_model, p_map_sample, xyzph_map_sample, bg_map_sample, zernike_coefs)

        data_cam = self.camera.forward(data, curr_sub_fov_xy) \
            if isinstance(self.camera, ailoc.simulation.SCMOS) else self.camera.forward(data)

        if robust_scale:
            # random scale the data to break the strict Poisson distribution
            random_scale = torch.distributions.Uniform(0.5, 1.5).sample()
            data_cam = ((data_cam - self.camera.baseline) * random_scale + self.camera.baseline)
            xyzph_array_gt[:, :, :, 3] *= random_scale
            bg_map_sample = bg_map_sample * random_scale

        return data_cam, p_map_gt, xyzph_array_gt, mask_array_gt, bg_map_sample, curr_sub_fov_xy

    @deprecated.deprecated(reason='Use sample_training_data with param batch_photophysics=True instead.')
    def transloc_sample_training_data(self, batch_size, iter_train):
        """
        Sample a batch of training data.

        Args:
            batch_size (int): batch size
            iter_train (int): the number of training iterations, used for sequentially select the
                sub-fov to simulate images

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list):
                a batch of simulated images and corresponding ground truth
        """

        # for transformer, all sampled frames in the batch share the same bg and the molecules may
        # survive in multiple frames
        p_map_sample, xyzph_map_sample, bg_map_sample, curr_sub_fov_xy, zernike_coefs, \
        p_map_gt, xyzph_array_gt, mask_array_gt = self.mol_sampler.transloc_sample_for_train(batch_size, self.psf_model,
                                                                                             iter_train)

        data = self.gen_noiseless_data(self.psf_model, p_map_sample, xyzph_map_sample, bg_map_sample, zernike_coefs)

        data_cam = self.camera.forward(data, curr_sub_fov_xy) \
            if isinstance(self.camera, ailoc.simulation.SCMOS) else self.camera.forward(data)

        return data_cam, p_map_gt, xyzph_array_gt, mask_array_gt, bg_map_sample, curr_sub_fov_xy

    def sample_evaluation_data(self, batch_size, context_size,):
        """
        sample dataset for online evaluation, the data should be generated the same as the
        training data, and each batch in the dataset has a sub-fov.

        Args:
            batch_size (int): evaluation batch size, the evaluation dataset will
                have batch_size*context_size images.
            context_size (int): the number of frames to be used as temporal context,
                each batch unit has context_size images.

        Returns:
            (torch.Tensor, list, list):
                evaluation images with shape (num_image, local context, train_size, train_size),
                ground truth molecule list (frame, x, y, z, photon) and
                sub-fov list [x_start, x_end, y_start, y_end] (each batch is from a specific sub-fov).
        """

        molecule_list_gt = []
        eval_data = []
        sub_fov_xy_list = []
        for k in range(batch_size):
            data_cam, p_map_gt, xyzph_array_gt, mask_array_gt, bg_map_sample, curr_sub_fov_xy = \
                self.sample_training_data(1, context_size, k,)
            sub_fov_xy_list.append(curr_sub_fov_xy)
            eval_data.append(data_cam)
            for i in range(context_size):
                frame_idx = k*context_size + i + 1
                for j in range(mask_array_gt.shape[-1]):
                    if mask_array_gt[0, i, j] == 1:
                        molecule_list_gt.\
                            append([frame_idx,
                                    float(ailoc.common.cpu(xyzph_array_gt[0, i, j, 0] * self.psf_model.pixel_size_xy[0])),
                                    float(ailoc.common.cpu(xyzph_array_gt[0, i, j, 1] * self.psf_model.pixel_size_xy[1])),
                                    float(ailoc.common.cpu(xyzph_array_gt[0, i, j, 2] * self.mol_sampler.z_scale)),
                                    float(ailoc.common.cpu(xyzph_array_gt[0, i, j, 3] * self.mol_sampler.photon_scale))]
                                   )
        eval_data = torch.cat(eval_data, dim=0)

        # p_map_sample, xyzph_map_sample, bg_map_sample, sub_fov_xy_list, zernike_coefs, \
        # xyzph_array_gt, mask_array_gt = self.mol_sampler.sample_for_evaluation(num_image,
        #                                                                        self.psf_model,
        #                                                                        batch_photophysics)
        #
        # molecule_list_gt = []
        # eval_data = self.gen_noiseless_data(self.psf_model, p_map_sample, xyzph_map_sample, bg_map_sample, zernike_coefs)
        #
        # for i in range(num_image):
        #     eval_data[i] = self.camera.forward(eval_data[i], sub_fov_xy_list[i]) \
        #         if isinstance(self.camera, ailoc.simulation.SCMOS) else self.camera.forward(eval_data[i])
        #
        #     for j in range(mask_array_gt.shape[1]):
        #         if mask_array_gt[i, j] == 1:
        #             molecule_list_gt.\
        #                 append([i+1,
        #                         float(ailoc.common.cpu(xyzph_array_gt[i, j, 0] * self.psf_model.pixel_size_xy[0])),
        #                         float(ailoc.common.cpu(xyzph_array_gt[i, j, 1] * self.psf_model.pixel_size_xy[1])),
        #                         float(ailoc.common.cpu(xyzph_array_gt[i, j, 2] * self.mol_sampler.z_scale)),
        #                         float(ailoc.common.cpu(xyzph_array_gt[i, j, 3] * self.mol_sampler.photon_scale))]
        #                        )

        return eval_data, molecule_list_gt, sub_fov_xy_list

    def gen_noiseless_data(self, psf_model, delta_map, xyzph_map, bg, zernike_coefs):
        """
        Generate noiseless data from the given delta map and psf parameters.
        """

        batch_size, channels, height, width = delta_map.shape[0], delta_map.shape[1], delta_map.shape[2], delta_map.shape[3]

        bg = bg * self.mol_sampler.bg_scale

        x,y,z,photons = self._translate_maps(delta_map.reshape([-1, height, width]), xyzph_map.reshape([4, -1, height, width]))

        with torch.no_grad():
            if isinstance(psf_model, ailoc.simulation.VectorPSFCUDA):
                psf_patches = psf_model.simulate(x, y, z, photons, zernike_coefs) \
                    if zernike_coefs is not None else psf_model.simulate(x, y, z, photons)
            elif isinstance(psf_model, ailoc.simulation.VectorPSFTorch):
                psf_patches = psf_model.simulate(x, y, z, photons, zernike_coefs=zernike_coefs)
            else:
                raise NotImplementedError('PSF model not supported.')

        raw_data = self.place_psfs(delta_map.reshape([-1, height, width]), psf_patches) + bg.reshape([-1, height, width])

        return torch.reshape(raw_data, [batch_size, channels, height, width])

    def reconstruct_posterior(self, psf_model, delta_map, xyzph_map, bg):
        """
        Reconstruct PSF images from the given delta map, psf model, and localization posterior.

        Args:
            psf_model (ailoc.simulation.VectorPSFTorch): psf model
            delta_map (torch.Tensor): delta map with shape (batch_size, channels, height, width) to indicate where
                the PSF images should be placed
            xyzph_map (torch.Tensor): xyzph map with shape (batch_size, 4, height, width) to indicate the xyz phtons
                of the PSF images
            bg (torch.Tensor): background maps
        """

        batch_size, channels, height, width = delta_map.shape[0], delta_map.shape[1], delta_map.shape[2], \
                                              delta_map.shape[3]

        bg = torch.reshape(bg, [batch_size, 1, height, width]).repeat(1, channels, 1, 1) * self.mol_sampler.bg_scale

        x, y, z, photons = self._translate_maps(delta_map.reshape([-1, height, width]),
                                                xyzph_map.reshape([4, -1, height, width]))

        psf_patches = psf_model.simulate(x, y, z, photons)

        raw_data = self.place_psfs(delta_map.reshape([-1, height, width]), psf_patches) + bg.reshape(
            [-1, height, width])

        return torch.reshape(raw_data, [batch_size, channels, height, width])

    def _translate_maps(self, delta, xyzph):
        pix_inds = tuple(delta.nonzero().transpose(1, 0))

        x_offsets = xyzph[0][pix_inds] * self.psf_model.pixel_size_xy[0]
        y_offsets = xyzph[1][pix_inds] * self.psf_model.pixel_size_xy[1]
        z_offsets = xyzph[2][pix_inds] * self.mol_sampler.z_scale
        photons = xyzph[3][pix_inds] * self.mol_sampler.photon_scale

        return x_offsets, y_offsets, z_offsets, photons

    @staticmethod
    def place_psfs_v1(delta, psf_patches):
        """
        Place the psf_patches on the canvas according to the delta map.
        """

        psf_size = psf_patches.shape[-1]

        canvas = torch.zeros_like(delta)
        canvas_h, canvas_w = delta.shape[1], delta.shape[2]

        delta_inds = tuple(delta.nonzero().transpose(1, 0))
        relu = nn.ReLU()

        # all row and column indices and all possible row and column indices
        row_col_inds = delta.nonzero()[:, 1:]
        unique_inds = delta.sum(0).nonzero()

        # indices on canvas, corresponding to the upper-left corner of the psf_patch on the canvas
        canvas_row_start = relu(unique_inds[:, 0] - psf_size // 2)
        canvas_col_start = relu(unique_inds[:, 1] - psf_size // 2)

        # indices on psf_patches, cut the psf_patches if it is out of the canvas
        psf_row_start = relu(psf_size // 2 - unique_inds[:, 0])
        psf_row_end = psf_size - (unique_inds[:, 0] + psf_size // 2 - canvas_h) - 1
        psf_col_start = relu(psf_size // 2 - unique_inds[:, 1])
        psf_col_end = psf_size - (unique_inds[:, 1] + psf_size // 2 - canvas_w) - 1

        # convert row and column indices to linear indices, just to give each delta a unique index
        row_col_inds_linear = canvas_w * row_col_inds[:, 0] + row_col_inds[:, 1]
        unique_inds_linear = canvas_w * unique_inds[:, 0] + unique_inds[:, 1]

        for i in range(len(unique_inds)):
            # traverse the unique indices, get all candidates that should be put the
            # same way as the current unique index
            curr_inds = torch.nonzero(ailoc.common.gpu(row_col_inds_linear == unique_inds_linear[i]))[:, 0]

            # indices on psf_patches, make sure it can be put on the canvas
            w_cut = psf_patches[curr_inds, psf_row_start[i]: psf_row_end[i], psf_col_start[i]: psf_col_end[i]]

            canvas[delta_inds[0][curr_inds],
                   canvas_row_start[i]:canvas_row_start[i] + w_cut.shape[1],
                   canvas_col_start[i]:canvas_col_start[i] + w_cut.shape[2]] += w_cut

        return canvas

    @staticmethod
    def place_psfs(delta, psf_patches):
        """
        Places psf_patches on a larger, padded canvas to avoid cropping,
        then crops the result back to the original size. The scatter_add_ function
        may behave nondeterministically when given tensors on a CUDA device
        """
        if not isinstance(delta, torch.Tensor):
            delta = torch.tensor(delta)
        if not isinstance(psf_patches, torch.Tensor):
            psf_patches = torch.tensor(psf_patches)

        if len(psf_patches) == 0:
            return torch.zeros_like(delta)

        psf_size = psf_patches.shape[-1]

        assert psf_size % 2 == 1, "PSF size must be odd"

        batch_size, canvas_h, canvas_w = delta.shape

        # Determine padding size needed for the canvas
        pad_h = psf_size // 2
        pad_w = psf_size // 2

        # Create the larger, padded canvas
        padded_canvas_h = canvas_h + 2 * pad_h
        padded_canvas_w = canvas_w + 2 * pad_w
        padded_canvas = torch.zeros(batch_size, padded_canvas_h, padded_canvas_w,
                                    dtype=delta.dtype, device=delta.device)

        # Get the coordinates of all non-zero elements in delta
        batch_inds, row_inds, col_inds = torch.nonzero(delta, as_tuple=True)

        assert len(batch_inds) == psf_patches.shape[0], "The number of PSFs must match the number of non-zero pixels"

        # Calculate the top-left corner on the padded canvas for each PSF
        padded_row_start = row_inds
        padded_col_start = col_inds

        # Use scatter_add_ to place the patches in parallel
        # The psf_patches tensor must be reshaped for scatter_add_
        psf_patches_flat = psf_patches.reshape(psf_patches.shape[0], -1)

        # Create flat indices for each patch and its corresponding position on the padded canvas
        # This part is a bit tricky and requires careful index calculation.
        # We will build a set of indices for each pixel in each patch.
        psf_area = psf_size * psf_size

        # Repeat the indices for each pixel in the patch
        batch_indices_scatter = batch_inds.unsqueeze(1).repeat(1, psf_area).flatten()

        # Create the relative row and column indices for each pixel within a patch
        rel_rows = torch.arange(psf_size, device=delta.device).unsqueeze(1).repeat(1, psf_size)
        rel_cols = torch.arange(psf_size, device=delta.device).unsqueeze(0).repeat(psf_size, 1)
        rel_rows_flat = rel_rows.flatten()
        rel_cols_flat = rel_cols.flatten()

        # Calculate the absolute row and column indices on the padded canvas
        target_rows = padded_row_start.unsqueeze(1) + rel_rows_flat
        target_cols = padded_col_start.unsqueeze(1) + rel_cols_flat

        # Flatten the target indices for scatter_add_
        padded_canvas_rows = target_rows.flatten()
        padded_canvas_cols = target_cols.flatten()

        # Convert 2D indices to 1D linear indices for scatter_add_
        linear_indices = (batch_indices_scatter * padded_canvas_h * padded_canvas_w +
                          padded_canvas_rows * padded_canvas_w +
                          padded_canvas_cols)

        # Use scatter_add_ to place all patches at once
        padded_canvas.view(-1).scatter_add_(0, linear_indices, psf_patches_flat.flatten())

        # Crop the padded canvas back to the original size
        cropped_canvas = padded_canvas[:, pad_h: pad_h + canvas_h, pad_w: pad_w + canvas_w]

        return cropped_canvas

