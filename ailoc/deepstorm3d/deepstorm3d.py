import torch
import numpy as np
import time
import collections
import matplotlib.pyplot as plt
import datetime

import ailoc.common
import ailoc.simulation
import ailoc.deepstorm3d


class DeepSTORM3D(ailoc.common.XXLoc):
    """
    DeepSTORM3D class, the core code is adopted from the DeepSTORM3D project
    (Nehme, E. et al. DeepSTORM3D: dense 3D localization microscopy and PSF design by deep learning.
    Nat Methods 17, 734–740 (2020)).
    """

    def __init__(self, psf_params_dict, camera_params_dict, sampler_params_dict):
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.dict_psf_params, self.dict_camera_params, self.dict_sampler_params = \
            psf_params_dict, camera_params_dict, sampler_params_dict

        self._data_simulator = ailoc.simulation.Simulator(psf_params_dict, camera_params_dict, sampler_params_dict)
        self.scale_ph_offset = np.mean(self.dict_sampler_params['bg_range'])
        self.scale_ph_factor = self.dict_sampler_params['photon_range'][1]/50

        self.local_context = False  # DeepSTORM3D cannot process temporal context
        # attn_length only useful when local_context=True, should be odd,
        # using the same number of frames before and after the target frame
        self.attn_length = 1
        assert self.attn_length % 2 == 1, 'attn_length should be odd'
        # add frames at the beginning and end to provide context
        self.context_size = sampler_params_dict['context_size'] + 2*(self.attn_length//2) if self.local_context else sampler_params_dict['context_size']

        # upsampling factor for xy
        self.upsampling_factor = 4  # the xy resolution of the output is 1/4 pixel size (nm)
        #  discretization in z, # in [voxels] spanning the axial range (zmax - zmin)
        self.discret_z = 120  # for astigmatism (+-700 nm) the z size is 11.67 nm and for Tetrapod (+-3 μm) is 50 nm
        # scaling factor for the loss function to balance the vacant and occupied voxels
        self.scaling_factor = 800.0

        self._network = ailoc.deepstorm3d.LocalizationCNN(self.local_context,
                                                          self.attn_length,
                                                          self.context_size,
                                                          self.discret_z,
                                                          self.scaling_factor)
        self.network.to(self._device)
        self.network.get_parameter_number()

        self.evaluation_dataset = {}
        self.evaluation_recorder = self._init_recorder()

        self._iter_train = 0

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=5e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                    mode='min',
                                                                    factor=0.1,
                                                                    patience=2,
                                                                    verbose=True,
                                                                    min_lr=1e-6)

        self.postprocessing_module = ailoc.deepstorm3d.Postprocess(device=self._device,
                                                                   pixel_size_xy=np.array(self.dict_psf_params['pixel_size_xy'])/self.upsampling_factor,
                                                                   pixel_size_z=2/self.discret_z*self.data_simulator.mol_sampler.z_scale,
                                                                   z_min=-self.data_simulator.mol_sampler.z_scale,
                                                                   thresh=20,
                                                                   radius=6,
                                                                   )

        self.loss_func = ailoc.deepstorm3d.KDE_loss3D(self.scaling_factor, self._device)

    @staticmethod
    def _init_recorder():
        recorder = {'loss': collections.OrderedDict(),  # loss function value
                    'iter_time': collections.OrderedDict(),  # time cost for each iteration
                    'n_per_img': collections.OrderedDict(),  # average summed probability channel per image
                    'recall': collections.OrderedDict(),  # TP/(TP+FN)
                    'precision': collections.OrderedDict(),  # TP/(TP+FP)
                    'jaccard': collections.OrderedDict(),  # TP/(TP+FP+FN)
                    'rmse_lat': collections.OrderedDict(),  # root of mean squared error
                    'rmse_ax': collections.OrderedDict(),
                    'rmse_vol': collections.OrderedDict(),
                    'jor': collections.OrderedDict(),  # 100*jaccard/rmse_lat
                    'eff_lat': collections.OrderedDict(),  # 100-np.sqrt((100-100*jaccard)**2+1**2*rmse_lat**2)
                    'eff_ax': collections.OrderedDict(),  # 100-np.sqrt((100-100*jaccard)**2+0.5**2*rmse_ax**2)
                    'eff_3d': collections.OrderedDict()  # (eff_lat+eff_ax)/2
                    }

        return recorder

    @property
    def network(self):
        return self._network

    @property
    def data_simulator(self):
        return self._data_simulator

    def compute_loss(self, pred, target):
        """
        Loss function.
        """
        total_loss = self.loss_func(pred, target)

        return total_loss

    def online_train(self, batch_size=1, max_iterations=50000, eval_freq=500, file_name=None):
        """
        Train the network.

        Args:
            batch_size (int): batch size
            max_iterations (int): maximum number of iterations
            eval_freq (int): every eval_freq iterations the network will be saved
                and evaluated on the evaluation dataset to check the current performance
            file_name (str): the name of the file to save the network
        """

        file_name = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M') + 'DeepSTORM3D.pt' if file_name is None else file_name

        print('Start training...')

        if self._iter_train > 0:
            print('training from checkpoint, the recent performance is:')
            self.print_recorder(max_iterations)

        while self._iter_train < max_iterations:
            t0 = time.time()
            total_loss = []
            for i in range(eval_freq):
                train_data, p_map_gt, xyzph_array_gt, mask_array_gt, bg_map_gt, _ = \
                    self.data_simulator.sample_training_data(batch_size=batch_size,
                                                             context_size=self.context_size,
                                                             iter_train=self._iter_train)
                grid_gt = self.transform_target(p_map_gt, xyzph_array_gt, mask_array_gt)
                pred_volume = self.inference(train_data, self.data_simulator.camera)

                loss = self.compute_loss(pred_volume, grid_gt)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self._iter_train += 1

                total_loss.append(ailoc.common.cpu(loss))

                # print(f'iter: {self._iter_train}, loss: {ailoc.common.cpu(loss)}, {pred_volume.max()}')

            avg_iter_time = 1000 * (time.time() - t0) / eval_freq
            avg_loss = np.mean(total_loss)
            self.evaluation_recorder['loss'][self._iter_train] = avg_loss
            self.evaluation_recorder['iter_time'][self._iter_train] = avg_iter_time

            # reduce learning rate if loss stagnates
            self.scheduler.step(avg_loss)

            if self._iter_train > 1000:
                print('-' * 200)
                self.online_evaluate(batch_size=batch_size*self.context_size)

            self.print_recorder(max_iterations)
            self.save(file_name)

        print('training finished!')

        self.determine_post_process_param(batch_size=batch_size*self.context_size)
        self.save(file_name)

    def transform_target(self, p_map_gt, xyzph_array_gt, mask_array_gt):
        '''
        Transform the target data to the format that can be used for DeepSTORM3D loss computation.
        '''

        n, c, h, w = p_map_gt.shape
        xyzph_array_gt = xyzph_array_gt.reshape(n*c, -1, 4)
        mask_array_gt = mask_array_gt.reshape(n*c, -1)

        # extract xyz list for each image
        xyz_list_np = []
        for i in range(n*c):
            xyz_tmp = []
            for j in range(mask_array_gt.shape[1]):
                if mask_array_gt[i,j] == 0:
                    continue
                mol_xyz_tmp = xyzph_array_gt[i, j][:-1]
                xyz_tmp.append(mol_xyz_tmp)
            xyz_list_np.append(ailoc.common.cpu(torch.stack(xyz_tmp))) if xyz_tmp else xyz_list_np.append(np.zeros((0, 3)))

        # calculate upsampling factor
        upsampling_factor = self.upsampling_factor

        # discrete axial size
        pixel_size_axial = 2/self.discret_z

        boolean_grid_list = []
        for i in range(n*c):
            xyz_np = xyz_list_np[i][None]
            # if no localization, add a zero tensor
            if xyz_np.shape[1] == 0:
                boolean_grid_list.append(torch.zeros((self.discret_z, int(h * upsampling_factor), int(w * upsampling_factor))))
                continue

            # shift the z axis back to 0
            zshift = xyz_np[:, :, 2] - (-1)
            # zshift = 1 - xyz_np[:, :, 2]

            # number of particles
            batch_size, num_particles = zshift.shape

            # project xyz locations on the grid and shift xy to the upper left corner
            xg = (np.floor(xyz_np[:, :, 0] / (1/self.upsampling_factor) ) ).astype('int')
            yg = (np.floor(xyz_np[:, :, 1] / (1/self.upsampling_factor) ) ).astype('int')
            zg = (np.floor(zshift / pixel_size_axial)).astype('int')

            # indices for sparse tensor
            indX, indY, indZ = (xg.flatten('F')).tolist(), (yg.flatten('F')).tolist(), (zg.flatten('F')).tolist()

            # update dimensions
            new_h, new_w = int(h * upsampling_factor), int(w * upsampling_factor)

            # if sampling a batch add a sample index
            if batch_size > 1:
                indS = (np.kron(np.ones(num_particles), np.arange(0, batch_size, 1)).astype('int')).tolist()
                ibool = torch.LongTensor([indS, indZ, indY, indX])
            else:
                ibool = torch.LongTensor([indZ, indY, indX])

            # spikes for sparse tensor
            vals = torch.ones(batch_size * num_particles)

            # resulting 3D boolean tensor
            if batch_size > 1:
                boolean_grid = torch.sparse_coo_tensor(ibool, vals, torch.Size([batch_size, self.discret_z, new_h, new_w]),
                                                       dtype=torch.float32).to_dense()
            else:
                boolean_grid = torch.sparse_coo_tensor(ibool, vals, torch.Size([self.discret_z, new_h, new_w]), dtype=torch.float32).to_dense()

            boolean_grid_list.append(boolean_grid)

        return ailoc.common.gpu(torch.stack(boolean_grid_list))

    def inference(self, data, camera):
        """
        Inference with the network, the input data should be transformed into photon unit,
        output are prediction maps that can be directly used for loss computation.

        Args:
            data (torch.Tensor): input data, shape (batch_size, optional local context, H, W)
            camera (ailoc.simulation.Camera): camera object used to transform the adu data to photon data
        """

        data_photon = camera.backward(data)
        # data_scaled = (data_photon - self.scale_ph_offset)/self.scale_ph_factor
        # pred_volume = self.network(data_scaled)

        data_scaled = ((data_photon - data_photon.min(-1, True)[0].min(-2, True)[0]) /
                       (data_photon.max(-1, True)[0].max(-2, True)[0] -
                        data_photon.min(-1, True)[0].min(-2, True)[0]))
        pred_volume = self.network(data_scaled)

        return pred_volume

    def post_process(self, p_pred, xyzph_pred, xyzph_sig_pred, bg_pred, return_infer_map=False):
        """
        Postprocess a batch of inference output map, output is GMM maps and molecule array
        [frame, x, y, z, photon, integrated prob, x uncertainty, y uncertainty, z uncertainty,
        photon uncertainty, x_offset, y_offset].
        """

        raise NotImplementedError('DeepSTORM3D uses its own analyze function for postprocessing!')

    def determine_post_process_param(self, batch_size):
        """
        Determine the best postprocess parameters on evaluation dataset.
        """
        print('find optimal combinations of postprocessing parameters for inference')

        efficiency_record = []

        thresh_to_test = [5, 10, 20, 30, 40, 80]
        radius_to_test = [2, 4, 5, 6, 8, 10]

        for thresh in thresh_to_test:
            for radius in radius_to_test:
                self.postprocessing_module = ailoc.deepstorm3d.Postprocess(device=self._device,
                                                                           pixel_size_xy=np.array(self.dict_psf_params[
                                                                                                      'pixel_size_xy']) / self.upsampling_factor,
                                                                           pixel_size_z=2/self.discret_z*self.data_simulator.mol_sampler.z_scale,
                                                                           z_min=-self.data_simulator.mol_sampler.z_scale,
                                                                           thresh=thresh,
                                                                           radius=radius,
                                                                           )
                self.online_evaluate(batch_size=batch_size)
                efficiency_record.append([thresh,
                                          radius,
                                          self.evaluation_recorder['eff_3d'][self._iter_train],
                                          self.evaluation_recorder['rmse_lat'][self._iter_train],
                                          self.evaluation_recorder['rmse_ax'][self._iter_train],
                                          self.evaluation_recorder['recall'][self._iter_train],
                                          self.evaluation_recorder['precision'][self._iter_train],
                                          self.evaluation_recorder['jaccard'][self._iter_train],
                                          ])

        print(f'below are: threshold; radius; efficiency 3d; lateral RMSE; axial RMSE; Recall; Precision; Jaccard')
        for record in efficiency_record:
            print(record)

        # replace the nan values to -inf, choose the threshold and radius with the best efficiency
        efficiency_record = np.array(efficiency_record)
        efficiency_record[np.isnan(efficiency_record)] = -np.inf
        best_record = np.array(efficiency_record)[np.argmax(efficiency_record[:, 2])]
        thresh_best = best_record[0]
        radius_best = best_record[1]
        print(f'optimal postprocess parameters: threshold={thresh_best}, radius={radius_best}')
        self.postprocessing_module = ailoc.deepstorm3d.Postprocess(device=self._device,
                                                                   pixel_size_xy=np.array(self.dict_psf_params[
                                                                                              'pixel_size_xy']) / self.upsampling_factor,
                                                                   pixel_size_z=2/self.discret_z*self.data_simulator.mol_sampler.z_scale,
                                                                   z_min=-self.data_simulator.mol_sampler.z_scale,
                                                                   thresh=int(thresh_best),
                                                                   radius=int(radius_best),
                                                                   )

    def analyze(self, data, camera, sub_fov_xy=None, return_infer_map=False):
        """
        the official implementation of DeepSTORM3D can only postprocess 1 frame at a time,
        so we need to traverse the batch data and postprocess each frame separately.
        """

        h, w = data.shape[-2:]
        data = data.reshape(-1, 1, h, w)
        self.network.eval()
        with torch.no_grad():
            mol_list = []
            for i in range(data.shape[0]):
                data_tmp = data[i:i + 1]
                pred_volume = self.inference(data_tmp, camera)
                xyz_rec, conf_rec = self.postprocessing_module(pred_volume)
                if xyz_rec is not None:
                    frame_idx = np.full((xyz_rec.shape[0], 1), i+1)
                    mol_list_tmp = np.column_stack((frame_idx, xyz_rec, conf_rec))
                    mol_list.append(mol_list_tmp)

        if len(mol_list) == 0:
            mol_array = np.zeros((0, 5))
        else:
            mol_array = np.vstack(mol_list)

        return mol_array, None

        # h, w = data.shape[-2:]
        # data = data.reshape(-1, 1, h, w)
        # self.network.eval()
        # with torch.no_grad():
        #     mol_list = []
        #     pred_volume = self.inference(data, camera)
        #     for i in range(pred_volume.shape[0]):
        #         xyz_rec, conf_rec = self.postprocessing_module(pred_volume[i:i+1])
        #         if xyz_rec is not None:
        #             frame_idx = np.full((xyz_rec.shape[0], 1), i+1)
        #             mol_list_tmp = np.column_stack((frame_idx, xyz_rec, conf_rec))
        #             mol_list.append(mol_list_tmp)
        #
        # if len(mol_list) == 0:
        #     mol_array = np.zeros((0, 5))
        # else:
        #     mol_array = np.vstack(mol_list)
        #
        # return mol_array, None

    def online_evaluate(self, batch_size):
        """
        Evaluate the network during training using the validation dataset.
        """

        self.network.eval()
        with torch.no_grad():
            print('evaluating...')
            t0 = time.time()

            n_per_img = []
            molecule_list_pred = []
            for i in range(int(np.ceil(self.evaluation_dataset['data'].shape[0]/batch_size))):
                molecule_array_tmp, inference_dict_tmp = \
                    self.analyze(
                        ailoc.common.gpu(self.evaluation_dataset['data'][i * batch_size: (i + 1) * batch_size]),
                        self.data_simulator.camera,
                        return_infer_map=True)

                n_per_img.append(np.nan)  # no need for this value in DeepSTORM3D

                if len(molecule_array_tmp) > 0:
                    molecule_array_tmp[:, 0] += i * batch_size * (self.context_size-2*(self.attn_length//2) if self.local_context else self.context_size)
                    molecule_list_pred += molecule_array_tmp.tolist()

            metric_dict, paired_array = ailoc.common.pair_localizations(prediction=np.array(molecule_list_pred),
                                                                        ground_truth=self.evaluation_dataset['molecule_list_gt'],
                                                                        frame_num=self.evaluation_dataset['data'].shape[0] * (self.context_size-2*(self.attn_length//2) if self.local_context else self.context_size),
                                                                        fov_xy_nm=(0, self.evaluation_dataset['data'].shape[-1]*self.data_simulator.psf_model.pixel_size_xy[0],
                                                                                   0, self.evaluation_dataset['data'].shape[-2]*self.data_simulator.psf_model.pixel_size_xy[1]))

            for k in self.evaluation_recorder.keys():
                if k in metric_dict.keys():
                    self.evaluation_recorder[k][self._iter_train] = metric_dict[k]

            self.evaluation_recorder['n_per_img'][self._iter_train] = np.mean(n_per_img)

            print(f'evaluating done! time cost: {time.time() - t0:.2f}s')

        self.network.train()

    def build_evaluation_dataset(self, napari_plot=False):
        """
        Build the evaluation dataset, sampled by the same way as training data.
        """

        print("building evaluation dataset, this may take a while...")
        t0 = time.time()
        eval_data, molecule_list_gt, sub_fov_xy_list = \
            self.data_simulator.sample_evaluation_data(batch_size=self.dict_sampler_params['eval_batch_size'],
                                                       context_size=self.context_size,)
        self.evaluation_dataset['data'] = ailoc.common.cpu(eval_data)

        molecule_list_gt = np.array(molecule_list_gt)
        if self.local_context:
            molecule_list_gt_corrected = []
            for i in range(self.dict_sampler_params['eval_batch_size']):
                curr_context_idx = np.where((molecule_list_gt[:, 0]>i*self.context_size+(self.attn_length//2))&
                                            (molecule_list_gt[:, 0]<(i+1)*self.context_size-(self.attn_length//2)+1))
                curr_molecule_list_gt = molecule_list_gt[curr_context_idx]
                curr_molecule_list_gt[:, 0] -= ((2*i+1) * (self.attn_length//2))
                molecule_list_gt_corrected.append(curr_molecule_list_gt)
            molecule_list_gt = np.concatenate(molecule_list_gt_corrected, axis=0)

        self.evaluation_dataset['molecule_list_gt'] = molecule_list_gt
        self.evaluation_dataset['sub_fov_xy_list'] = sub_fov_xy_list
        print(f"evaluation dataset with shape {eval_data.shape} building done! "
              f"contain {len(molecule_list_gt)} target molecules, "
              f"time cost: {time.time() - t0:.2f}s")

        if napari_plot:
            print('visually checking evaluation data...')
            ailoc.common.viewdata_napari(eval_data)

    def save(self, file_name):
        """
        Save the whole DeepLoc instance, including the network, optimizer, recorder, etc.
        """

        with open(file_name, 'wb') as f:
            torch.save(self, f)
        print(f"DeepSTORM3D instance saved to {file_name}")

    def check_training_psf(self, num_z_step=21):
        """
        Check the PSF.
        """

        print(f"checking PSF...")
        x = ailoc.common.gpu(torch.zeros(num_z_step))
        y = ailoc.common.gpu(torch.zeros(num_z_step))
        z = ailoc.common.gpu(torch.linspace(*self.data_simulator.mol_sampler.z_range, num_z_step))
        photons = ailoc.common.gpu(torch.ones(num_z_step))

        psf = ailoc.common.cpu(self.data_simulator.psf_model.simulate(x, y, z, photons))

        plt.figure(constrained_layout=True)
        for j in range(num_z_step):
            plt.subplot(int(np.ceil(num_z_step/7)), 7, j + 1)
            plt.imshow(psf[j], cmap='gray')
            plt.title(f"{ailoc.common.cpu(z[j]):.0f} nm")
        plt.show()

    def check_training_data(self):
        """
        Check the training data ,randomly sample a batch of training data and visualize it.
        """

        print(f"checking training data...")
        data_cam, p_map_gt, xyzph_array_gt, mask_array_gt, bg_map_sample, curr_sub_fov_xy = \
            self.data_simulator.sample_training_data(batch_size=1,
                                                     context_size=self.context_size,
                                                     iter_train=0, )

        cmap = 'gray'

        im_num = self.context_size * 2
        n_col = 4
        n_row = int(np.ceil(im_num / n_col))

        fig, ax_arr = plt.subplots(n_row,
                                   n_col,
                                   figsize=(n_col * 3,
                                            n_row * 2),
                                   constrained_layout=True)
        ax = []
        plts = []
        for i in ax_arr:
            try:
                for j in i:
                    ax.append(j)
            except:
                ax.append(i)

        for i in range(self.context_size):
            plts.append(ax[i].imshow(ailoc.common.cpu(data_cam)[0, i], cmap=cmap))
            ax[i].set_title(f"frame {i}")
            plt.colorbar(mappable=plts[-1], ax=ax[i], fraction=0.046, pad=0.04)

        for i in range(self.context_size):
            ax_num = i + self.context_size
            plts.append(ax[ax_num].imshow(ailoc.common.cpu(data_cam)[0, i], cmap=cmap))
            ax[ax_num].set_title(f"GT frame {i}")
            pix_gt = ailoc.common.cpu(p_map_gt[0, i].nonzero())
            ax[ax_num].scatter(pix_gt[:, 1], pix_gt[:, 0], s=5, c='m', marker='x')
            plt.colorbar(mappable=plts[-1], ax=ax[ax_num], fraction=0.046, pad=0.04)

        plt.show()

    def print_recorder(self, max_iterations):
        try:
            print(f"Iterations: {self._iter_train}/{max_iterations} || "
                  f"Loss: {self.evaluation_recorder['loss'][self._iter_train]:.2f} || "
                  f"IterTime: {self.evaluation_recorder['iter_time'][self._iter_train]:.2f} ms || "
                  f"ETA: {self.evaluation_recorder['iter_time'][self._iter_train] * (max_iterations - self._iter_train) / 3600000:.2f} h || ",
                  end='')

            print(f"SumProb: {self.evaluation_recorder['n_per_img'][self._iter_train]:.2f} || "
                  f"Eff_3D: {self.evaluation_recorder['eff_3d'][self._iter_train]:.2f} || "
                  f"Jaccard: {self.evaluation_recorder['jaccard'][self._iter_train]:.2f} || "
                  f"Recall: {self.evaluation_recorder['recall'][self._iter_train]:.2f} || "
                  f"Precision: {self.evaluation_recorder['precision'][self._iter_train]:.2f} || "
                  f"RMSE_lat: {self.evaluation_recorder['rmse_lat'][self._iter_train]:.2f} || "
                  f"RMSE_ax: {self.evaluation_recorder['rmse_ax'][self._iter_train]:.2f}")

        except KeyError:
            print('No recent performance record found')

    def remove_gpu_attribute(self):
        """
        Remove the gpu attribute of the loc model, so that can be shared between processes.
        """

        self._network.to('cpu')
        self.optimizer = None
        self.scheduler = None
        self.evaluation_dataset = None
        self._data_simulator = None
