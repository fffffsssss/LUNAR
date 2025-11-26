import warnings
import deprecated
import torch
import numpy as np
import time
import collections
import matplotlib.pyplot as plt
import datetime
import random
from tqdm import tqdm
import copy
import collections
import scipy
import os

import ailoc.common
import ailoc.simulation
import ailoc.lunar


class Lunar_LocLearning(ailoc.common.XXLoc):
    """
    LUNAR class, Localization Using Neural-physics Adaptive Reconstruction
    """

    def __init__(self, psf_params_dict, camera_params_dict, sampler_params_dict, attn_length=7):
        self.dict_psf_params, self.dict_camera_params, self.dict_sampler_params = \
            psf_params_dict, camera_params_dict, sampler_params_dict

        self._data_simulator = ailoc.simulation.Simulator(psf_params_dict, camera_params_dict, sampler_params_dict)
        self.scale_ph_offset = np.mean(self.dict_sampler_params['bg_range'])
        self.scale_ph_factor = self.dict_sampler_params['photon_range'][1]/50

        self.temporal_attn = (self.dict_sampler_params.get('temporal_attn', False) or self.dict_sampler_params.get(
            'local_context', False))
        self.local_context = self.temporal_attn  # for possible use in somewhere
        # should be odd, using the same number of frames before and after the target frame
        self.attn_length = attn_length
        assert self.attn_length % 2 == 1, 'attn_length should be odd'
        # add frames at the beginning and end to provide context
        self.context_size = sampler_params_dict['context_size'] + 2*(self.attn_length//2) if self.temporal_attn else sampler_params_dict['context_size']
        self._network = ailoc.lunar.LunarNet(
            self.temporal_attn,
            self.attn_length,
            self.context_size,
        )

        self.evaluation_dataset = {}
        self.evaluation_recorder = self._init_recorder()

        self._iter_train = 0

        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.optimizer = torch.optim.AdamW(self.network.parameters(), lr=6e-4, weight_decay=0.05)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=30000)

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

    def compute_loss(self, p_pred, xyzph_pred, xyzph_sig_pred, bg_pred, p_gt, xyzph_array_gt, mask_array_gt, bg_gt):
        """
        Loss function.
        """

        count_loss = torch.mean(ailoc.lunar.count_loss(p_pred, p_gt))
        loc_loss = torch.mean(ailoc.lunar.loc_loss(p_pred, xyzph_pred, xyzph_sig_pred, xyzph_array_gt, mask_array_gt))
        sample_loss = torch.mean(ailoc.lunar.sample_loss(p_pred, p_gt))
        bg_loss = torch.mean(ailoc.lunar.bg_loss(bg_pred, bg_gt))

        total_loss = count_loss + loc_loss + sample_loss + bg_loss

        return total_loss

    def online_train(self,
                     batch_size=1,
                     max_iterations=50000,
                     eval_freq=500,
                     file_name=None,
                     robust_scale=False):
        """
        Train the network.

        Args:
            batch_size (int): batch size
            max_iterations (int): maximum number of iterations
            eval_freq (int): every eval_freq iterations the network will be saved
                and evaluated on the evaluation dataset to check the current performance
            file_name (str): the name of the file to save the network
        """

        file_name = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M') + 'LUNAR_LL.pt' if file_name is None else file_name
        if not os.path.exists(os.path.dirname(file_name)):
            os.makedirs(os.path.dirname(file_name))

        self.scheduler.T_max = max_iterations
        print('Start training...')

        if self._iter_train > 0:
            print('training from checkpoint, the recent performance is:')
            self.print_recorder(max_iterations)

            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                        T_max=max_iterations,
                                                                        last_epoch=self._iter_train)

        while self._iter_train < max_iterations:
            t0 = time.time()
            total_loss = []
            for i in range(eval_freq):
                train_data, p_map_gt, xyzph_array_gt, mask_array_gt, bg_map_gt, _ = \
                    self.data_simulator.sample_training_data(batch_size=batch_size,
                                                             context_size=self.context_size,
                                                             iter_train=self._iter_train,
                                                             robust_scale=robust_scale)
                p_map_gt, xyzph_array_gt, mask_array_gt, bg_map_gt = self.unfold_target(p_map_gt,
                                                                                        xyzph_array_gt,
                                                                                        mask_array_gt,
                                                                                        bg_map_gt)
                p_pred, xyzph_pred, xyzph_sig_pred, bg_pred = self.inference(train_data,
                                                                             self.data_simulator.camera,)
                loss = self.compute_loss(p_pred, xyzph_pred, xyzph_sig_pred, bg_pred,
                                         p_map_gt, xyzph_array_gt, mask_array_gt, bg_map_gt)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self._iter_train += 1

                total_loss.append(ailoc.common.cpu(loss))

            avg_iter_time = 1000 * (time.time() - t0) / eval_freq
            avg_loss = np.mean(total_loss)
            self.evaluation_recorder['loss'][self._iter_train] = avg_loss
            self.evaluation_recorder['iter_time'][self._iter_train] = avg_iter_time

            if self._iter_train > 1000:
                print('-' * 200)
                self.online_evaluate(batch_size=batch_size)

            self.print_recorder(max_iterations)
            self.save(file_name)

        print('training finished!')

    def unfold_target(self, p_map_gt, xyzph_array_gt, mask_array_gt, bg_map_gt):
        if self.temporal_attn:
            extra_length = self.attn_length // 2
            if extra_length > 0:
                p_map_gt = p_map_gt[:, extra_length:-extra_length]
                xyzph_array_gt = xyzph_array_gt[:, extra_length:-extra_length]
                mask_array_gt = mask_array_gt[:, extra_length:-extra_length]
                bg_map_gt = bg_map_gt[:, extra_length:-extra_length]
        else:
            pass
        return p_map_gt.flatten(start_dim=0, end_dim=1), \
               xyzph_array_gt.flatten(start_dim=0, end_dim=1), \
               mask_array_gt.flatten(start_dim=0, end_dim=1), \
               bg_map_gt.flatten(start_dim=0, end_dim=1)

    def inference(self, data, camera):
        """
        Inference with the network, the input data should be transformed into photon unit,
        output are prediction maps that can be directly used for loss computation.

        Args:
            data (torch.Tensor): input data, shape (batch_size, optional local context, H, W)
            camera (ailoc.simulation.Camera): camera object used to transform the adu data to photon data
        """

        data_photon = camera.backward(data)
        data_scaled = (data_photon - self.scale_ph_offset)/self.scale_ph_factor
        p_pred, xyzph_pred, xyzph_sig_pred, bg_pred = self.network(data_scaled)

        return p_pred, xyzph_pred, xyzph_sig_pred, bg_pred

    def post_process(self, p_pred, xyzph_pred, xyzph_sig_pred, bg_pred, return_infer_map=False):
        """
        Postprocess a batch of inference output map, output is GMM maps and molecule array
        [frame, x, y, z, photon, integrated prob, x uncertainty, y uncertainty, z uncertainty,
        photon uncertainty, x_offset_pixel, y_offset_pixel].
        """

        molecule_array, inference_dict = ailoc.common.gmm_to_localizations(p_pred=p_pred,
                                                                           xyzph_pred=xyzph_pred,
                                                                           xyzph_sig_pred=xyzph_sig_pred,
                                                                           bg_pred=bg_pred,
                                                                           thre_integrated=0.7,
                                                                           pixel_size_xy=self.data_simulator.psf_model.pixel_size_xy,
                                                                           z_scale=self.data_simulator.mol_sampler.z_scale,
                                                                           photon_scale=self.data_simulator.mol_sampler.photon_scale,
                                                                           bg_scale=self.data_simulator.mol_sampler.bg_scale,
                                                                           batch_size=p_pred.shape[0],
                                                                           return_infer_map=return_infer_map)

        return molecule_array, inference_dict

    def analyze(self, data, camera, sub_fov_xy=None, return_infer_map=False):
        """
        Wrap the inference and post_process function, receive a batch of data and return the molecule list.

        Args:
            data (torch.Tensor): a batch of data to be analyzed.
            camera (ailoc.simulation.Camera): camera object used to transform the data to photon unit.
            sub_fov_xy (tuple of int): (x_start, x_end, y_start, y_end), start from 0, in pixel unit,
                the FOV indicator for these images
            return_infer_map (bool): whether to return the prediction maps, which may occupy some memory
                and take more time during data analysis

        Returns:
            (np.ndarray, dict): molecule array, [frame, x, y, z, photon, integrated prob, x uncertainty,
                y uncertainty, z uncertainty, photon uncertainty...], the xy position are relative
                to the current image size, may need to be translated outside this function, the second output
                is a dict that contains the inferred multichannel maps from the network.
        """

        self.network.eval()
        with torch.no_grad():
            p_pred, xyzph_pred, xyzph_sig_pred, bg_pred = self.inference(data, camera)
            molecule_array, inference_dict = self.post_process(p_pred,
                                                               xyzph_pred,
                                                               xyzph_sig_pred,
                                                               bg_pred,
                                                               return_infer_map)

        return molecule_array, inference_dict

    def online_evaluate(self, batch_size, print_info=False):
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

                n_per_img.append(inference_dict_tmp['prob'].sum((-2, -1)).mean())

                if len(molecule_array_tmp) > 0:
                    molecule_array_tmp[:, 0] += i * batch_size * (self.context_size-2*(self.attn_length//2) if self.temporal_attn else self.context_size)
                    molecule_list_pred += molecule_array_tmp.tolist()

            metric_dict, paired_array = ailoc.common.pair_localizations(prediction=np.array(molecule_list_pred),
                                                                        ground_truth=self.evaluation_dataset['molecule_list_gt'],
                                                                        frame_num=self.evaluation_dataset['data'].shape[0]*(self.context_size-2*(self.attn_length//2) if self.temporal_attn else self.context_size),
                                                                        fov_xy_nm=(0, self.evaluation_dataset['data'].shape[-1]*self.data_simulator.psf_model.pixel_size_xy[0],
                                                                                   0, self.evaluation_dataset['data'].shape[-2]*self.data_simulator.psf_model.pixel_size_xy[1]),
                                                                        print_info=print_info,)

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

        # there are attn_length//2 more frames at the beginning and end
        molecule_list_gt = np.array(molecule_list_gt)
        if self.temporal_attn:
            molecule_list_gt_corrected = []
            for i in range(self.dict_sampler_params['eval_batch_size']):
                curr_context_idx = np.where((molecule_list_gt[:, 0] > i * self.context_size + (self.attn_length//2)) &
                                            (molecule_list_gt[:, 0] < (i + 1) * self.context_size - (self.attn_length//2)+1))
                curr_molecule_list_gt = molecule_list_gt[curr_context_idx]
                curr_molecule_list_gt[:, 0] -= (2*(self.attn_length//2) * i + (self.attn_length//2))
                molecule_list_gt_corrected.append(curr_molecule_list_gt)
            molecule_list_gt = np.concatenate(molecule_list_gt_corrected, axis=0)

        self.evaluation_dataset['molecule_list_gt'] = np.array(molecule_list_gt)
        self.evaluation_dataset['sub_fov_xy_list'] = sub_fov_xy_list
        print(f"evaluation dataset with shape {eval_data.shape} building done! "
              f"contain {len(molecule_list_gt)} target molecules, "
              f"time cost: {time.time() - t0:.2f}s")

        if napari_plot:
            print('visually checking evaluation data...')
            ailoc.common.viewdata_napari(eval_data)

    def save(self, file_name):
        """
        Save the whole LUNAR instance, including the network, optimizer, recorder, etc.
        """

        with open(file_name, 'wb') as f:
            torch.save(self, f)
        print(f"LUNAR instance saved to {file_name}")

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
        self.evaluation_dataset = None
        self._network.tam.embedding_layer.attn_mask = None
        self.optimizer = None
        self.scheduler = None
        self._data_simulator = None


class Lunar_SyncLearning(Lunar_LocLearning):
    """
    LUNAR class, simultaneously learning the localization network and the PSF model.
    """

    def __init__(self, psf_params_dict, camera_params_dict, sampler_params_dict, zernike_idx_learn=None):
        super().__init__(psf_params_dict, camera_params_dict, sampler_params_dict)

        self.zernike_idx_learn = zernike_idx_learn
        self.learned_psf = ailoc.simulation.VectorPSFTorch(psf_params_dict,
                                                           req_grad=True,
                                                           data_type=torch.float32,
                                                           zernike_idx_learn=self.zernike_idx_learn)

        self.warmup = 5000

        self.z_bins = 10
        self.real_data_z_weight = [np.ones(self.z_bins) * (1/self.z_bins),
                                   np.linspace(-1, 1, self.z_bins+1)]
        self.intervals = np.linspace(sampler_params_dict['z_range'][0],
                                     sampler_params_dict['z_range'][1],
                                     self.z_bins+1)
        self.target_z_counts = collections.Counter()
        for i in range(self.z_bins):
            self.target_z_counts[(self.intervals[i], self.intervals[i+1])] = 1000
        self.roilib_sparse_first = True
        self.over_cut = 8

        self.roi_lib = {'sparse': np.array([]),
                        'dense': np.array([]),
                        'sparse_z_counts': collections.Counter(),
                        'dense_z_counts': collections.Counter(),
                        'total_z_counts': collections.Counter()}
        self.roi_lib_test = None

        self.use_threshold = False
        self.photon_threshold = 3000/self.data_simulator.mol_sampler.photon_scale
        self.p_var_threshold = 0.00125  # p=0.5, var=0.25,  density=0.005, 0.25*0.005

        # self.optimizer_psf = torch.optim.Adam([self.learned_psf.zernike_coef], lr=0.01*self.z_bins)
        self.optimizer_psf = torch.optim.Adam([self.learned_psf.zernike_coef], lr=0.025)
        self.scheduler_psf = torch.optim.lr_scheduler.StepLR(self.optimizer_psf, step_size=1000, gamma=0.9)

    @staticmethod
    def _init_recorder():
        recorder = {'loss_sleep': collections.OrderedDict(),  # loss function value
                    'loss_wake': collections.OrderedDict(),
                    'loss_recon': collections.OrderedDict(),  # reconstruction loss of the test roi lib
                    'avg_3Dsig_recon': collections.OrderedDict(),  # average 3D sigma on the test roi lib
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
                    'eff_3d': collections.OrderedDict(),  # (eff_lat+eff_ax)/2
                    'learned_psf_zernike': collections.OrderedDict()  # learned PSF parameters
                    }

        return recorder

    def online_train(self,
                     batch_size=2,
                     max_iterations=50000,
                     eval_freq=1000,
                     file_name=None,
                     real_data=None,
                     num_sample=100,
                     wake_interval=1,
                     max_recon_psfs=5000,
                     online_build_eval_set=True,
                     robust_scale=False):
        """
        Train the network.

        Args:
            batch_size (int): batch size
            max_iterations (int): maximum number of iterations in sleep phase
            eval_freq (int): every eval_freq iterations the network will be saved
                and evaluated on the evaluation dataset to check the current performance
            file_name (str): the name of the file to save the network
            real_data (np.ndarray): real data to be used in wake phase
            num_sample (int): number of samples for posterior based expectation estimation
            wake_interval (int): do one wake iteration every wake_interval sleep iterations
            max_recon_psfs (int): maximum number of reconstructed psfs, considering the GPU memory usage
            online_build_eval_set (bool): whether to build the evaluation set online using the current learned psf,
                if False, the evaluation set should be manually built before training
        """

        file_name = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M') + 'LUNAR_SL.pt' if file_name is None else file_name
        self.scheduler.T_max = max_iterations

        assert real_data is not None, 'real data is not provided'

        # self.prepare_sample_real_data(real_data, batch_size)

        # use a flag to early stop the physics learning if zernike coefficients are converged
        zernike_converged = False

        print('-' * 200)
        print('Start training...')

        if self._iter_train > 0:
            print('training from checkpoint, the recent performance is:')
            self.print_recorder(max_iterations)

            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                        T_max=max_iterations,
                                                                        last_epoch=self._iter_train)

            if self._iter_train > self.warmup:
                # need to initialize the roi_lib as the model file does not save the roi_lib
                self.update_roilib2learn(real_data, self.context_size * batch_size)

        # Forcibly set robust_training to True in warmup phase for better generalization
        if self._iter_train < self.warmup:
            self.data_simulator.mol_sampler.robust_training = True

        while self._iter_train < max_iterations:
            t0 = time.time()
            total_loss_sleep = []
            total_loss_wake = []
            for i in range(eval_freq):
                loss_sleep = self.sleep_train(batch_size=batch_size, robust_scale=robust_scale)
                total_loss_sleep.append(loss_sleep)
                if self._iter_train > self.warmup and not zernike_converged:
                    if (self._iter_train-1) % self.warmup == 0:
                        # calculate the z weight due to non-uniform z distribution
                        # self.get_real_data_z_prior(real_data, self.sample_batch_size)
                        self.update_roilib2learn(real_data, self.context_size*batch_size)

                    if (self._iter_train-1) % wake_interval == 0:
                        # loss_wake = self.wake_train(real_data,
                        #                             self.sample_batch_size,
                        #                             num_sample,
                        #                             max_recon_psfs)
                        loss_wake = self.physics_learning(batch_size * self.context_size,
                                                          num_sample,
                                                          max_recon_psfs)
                        total_loss_wake.append(loss_wake) if loss_wake is not np.nan else None

                    # reset the robust training flag to user defined
                    if self._iter_train == self.warmup + 1:
                        self.data_simulator.mol_sampler.robust_training = self.dict_sampler_params['robust_training']

                if self._iter_train % 100 == 0:
                    self.evaluation_recorder['learned_psf_zernike'][
                        self._iter_train] = self.learned_psf.zernike_coef.detach().cpu().numpy()
                    if not zernike_converged:
                        # change the flag if zernike coefficients are converged
                        zernike_converged = self.check_zernike_convergence()

            torch.cuda.empty_cache()

            avg_iter_time = 1000 * (time.time() - t0) / eval_freq
            avg_loss_sleep = np.mean(total_loss_sleep)
            avg_loss_wake = np.mean(total_loss_wake) if len(total_loss_wake) > 0 else np.nan
            self.evaluation_recorder['loss_sleep'][self._iter_train] = avg_loss_sleep
            self.evaluation_recorder['loss_wake'][self._iter_train] = avg_loss_wake
            (self.evaluation_recorder['loss_recon'][self._iter_train],
             self.evaluation_recorder['avg_3Dsig_recon'][self._iter_train]) = self.compute_roilibtest_loss(batch_size *self.context_size)
            self.evaluation_recorder['iter_time'][self._iter_train] = avg_iter_time

            if self._iter_train > 1000:
                print('-' * 200)
                self.build_evaluation_dataset(napari_plot=False) if online_build_eval_set else None
                self.online_evaluate(batch_size=batch_size)

            self.print_recorder(max_iterations)
            self.save(file_name)

        self.plot_roilib_recon()

        print('training finished!')

    def sleep_train(self, batch_size, robust_scale):
        train_data, p_map_gt, xyzph_array_gt, mask_array_gt, bg_map_gt, _ = \
            self.data_simulator.sample_training_data(batch_size=batch_size,
                                                     context_size=self.context_size,
                                                     iter_train=self._iter_train,
                                                     robust_scale=robust_scale)
        p_map_gt, xyzph_array_gt, mask_array_gt, bg_map_gt = self.unfold_target(p_map_gt,
                                                                                xyzph_array_gt,
                                                                                mask_array_gt,
                                                                                bg_map_gt)
        p_pred, xyzph_pred, xyzph_sig_pred, bg_pred = self.inference(train_data,
                                                                     self.data_simulator.camera)
        loss = self.compute_loss(p_pred, xyzph_pred, xyzph_sig_pred, bg_pred,
                                 p_map_gt, xyzph_array_gt, mask_array_gt, bg_map_gt)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self._iter_train += 1

        return loss.detach().cpu().numpy()

    @deprecated.deprecated(reason="This function is deprecated, use physics_learning instead")
    def wake_train(self, real_data, batch_size, num_sample=50, max_recon_psfs=5000):
        """
        for each wake training iteration, using the current q network to localization enough signals, and then
        use this signals and posterior samples to update the PSF model
        """

        real_data_sampled_crop_list = []
        delta_map_sample_crop_list = []
        xyzph_map_sample_crop_list = []
        bg_sample_crop_list = []
        xyzph_pred_crop_list = []
        xyzph_sig_pred_crop_list = []
        z_weight_crop_list = []
        num_psfs = 0
        infer_round = 0
        patience = 10

        self.network.eval()
        with torch.no_grad():
            while num_psfs < max_recon_psfs and infer_round < patience:
                real_data_sampled = ailoc.common.gpu(self.sample_real_data(real_data, batch_size))

                p_pred, xyzph_pred, xyzph_sig_pred, bg_pred = self.inference(real_data_sampled,
                                                                             self.data_simulator.camera)
                delta_map_sample, xyzph_map_sample, bg_sample = self.sample_posterior(p_pred,
                                                                                      xyzph_pred,
                                                                                      xyzph_sig_pred,
                                                                                      bg_pred,
                                                                                      num_sample)
                if len(delta_map_sample.nonzero()) > 0.0075 * p_pred.shape[0] * p_pred.shape[1] * p_pred.shape[
                    2] * num_sample:
                    print('too many non-zero elements in delta_map_sample, the network probably diverges, '
                          'consider decreasing the PSF learning rate to make the network more stable')
                    return np.nan

                real_data_sampled = self.data_simulator.camera.backward(real_data_sampled)
                if self.temporal_attn and self.attn_length//2 > 0:
                    real_data_sampled = real_data_sampled[(self.attn_length//2): -(self.attn_length//2)]

                real_data_sampled_crop, \
                delta_map_sample_crop, \
                xyzph_map_sample_crop, \
                bg_sample_crop, \
                xyzph_pred_crop, \
                xyzph_sig_pred_crop, \
                z_weight_crop = self.crop_patches(delta_map_sample,
                                                  real_data_sampled,
                                                  xyzph_map_sample,
                                                  bg_sample,
                                                  p_pred,
                                                  xyzph_pred,
                                                  xyzph_sig_pred,
                                                  self.use_threshold,
                                                  self.photon_threshold,
                                                  self.p_var_threshold,
                                                  crop_size=self.data_simulator.psf_model.psf_size*2,
                                                  max_psfs=max_recon_psfs,
                                                  curr_num_psfs=num_psfs,
                                                  z_weight=self.real_data_z_weight)
                if real_data_sampled_crop is None:
                    infer_round += 1
                    continue

                real_data_sampled_crop_list.append(real_data_sampled_crop)
                delta_map_sample_crop_list.append(delta_map_sample_crop)
                xyzph_map_sample_crop_list.append(xyzph_map_sample_crop)
                bg_sample_crop_list.append(bg_sample_crop)
                xyzph_pred_crop_list.append(xyzph_pred_crop)
                xyzph_sig_pred_crop_list.append(xyzph_sig_pred_crop)
                z_weight_crop_list.append(z_weight_crop)

                infer_round += 1
                num_psfs += len(delta_map_sample_crop.nonzero())
        self.network.train()

        if len(real_data_sampled_crop_list) == 0:
            warnings.warn('No valid ROI was extracted; '
                          'the quality of the network or the raw data may be poor, '
                          'skip the current wake iteration')
            return np.nan

        real_data_sampled_crop_list = torch.cat(real_data_sampled_crop_list, dim=0)
        delta_map_sample_crop_list = torch.cat(delta_map_sample_crop_list, dim=0)
        xyzph_map_sample_crop_list = torch.cat(xyzph_map_sample_crop_list, dim=1)
        bg_sample_crop_list = torch.cat(bg_sample_crop_list, dim=0)
        xyzph_pred_crop_list = torch.cat(xyzph_pred_crop_list, dim=0)
        xyzph_sig_pred_crop_list = torch.cat(xyzph_sig_pred_crop_list, dim=0)
        z_weight_crop_list = torch.cat(z_weight_crop_list, dim=0)

        # calculate the p_theta(x|h), q_phi(h|x) to compute the loss and optimize the psf using Adam
        reconstruction = self.data_simulator.reconstruct_posterior(self.learned_psf,
                                                                   delta_map_sample_crop_list,
                                                                   xyzph_map_sample_crop_list,
                                                                   bg_sample_crop_list, )

        loss, elbo_record = self.wake_loss(real_data_sampled_crop_list,
                                           reconstruction,
                                           delta_map_sample_crop_list,
                                           xyzph_map_sample_crop_list,
                                           xyzph_pred_crop_list,
                                           xyzph_sig_pred_crop_list,
                                           z_weight_crop_list)
        self.optimizer_psf.zero_grad()
        loss.backward()
        self.optimizer_psf.step()
        self.scheduler_psf.step()

        # update the psf simulator
        if isinstance(self.data_simulator.psf_model, ailoc.simulation.VectorPSFTorch):
            self.data_simulator.psf_model.zernike_coef = self.learned_psf.zernike_coef.detach()
        elif isinstance(self.data_simulator.psf_model, ailoc.simulation.VectorPSFCUDA):
            self.data_simulator.psf_model.zernike_coef = self.learned_psf.zernike_coef.detach().cpu().numpy()

        return elbo_record.detach().cpu().numpy()

    def wake_loss(self, real_data,
                  reconstruction,
                  delta_map_sample,
                  xyzph_map_sample,
                  xyzph_pred,
                  xyzph_sig_pred,
                  weight_per_img):

        num_sample = reconstruction.shape[1]
        log_p_x_given_h = ailoc.lunar.compute_log_p_x_given_h(data=real_data[:, None].expand(-1, num_sample, -1, -1),
                                                                model=reconstruction)
        # log_q_h_given_x = ailoc.lunar.compute_log_q_h_given_x(xyzph_pred,
        #                                                         xyzph_sig_pred,
        #                                                         delta_map_sample,
        #                                                         xyzph_map_sample)
        with torch.no_grad():
            log_q_h_given_x = ailoc.lunar.compute_log_q_h_given_x(xyzph_pred,
                                                                    xyzph_sig_pred,
                                                                    delta_map_sample,
                                                                    xyzph_map_sample)
            importance = log_p_x_given_h - log_q_h_given_x
            importance_norm = torch.exp(importance - importance.logsumexp(dim=-1, keepdim=True))
            elbo_record = - torch.mean(torch.sum(importance_norm * log_p_x_given_h, dim=-1, keepdim=True))
            # elbo_record = - torch.mean(torch.sum(importance_norm * (log_p_x_given_h + log_q_h_given_x), dim=-1, keepdim=True))

        # elbo = torch.sum(importance_norm * (log_p_x_given_h + log_q_h_given_x), dim=-1, keepdim=True)
        elbo = torch.sum(importance_norm * log_p_x_given_h, dim=-1, keepdim=True)
        if weight_per_img is not None:
            total_loss = - torch.mean(elbo * weight_per_img[:, None])
        else:
            total_loss = - torch.mean(elbo)

        return total_loss, elbo_record

    def sample_posterior(self, p_pred, xyzph_pred, xyzph_sig_pred, bg_pred, num_sample):
        with torch.no_grad():
            batch_size, h, w = p_pred.shape[0], p_pred.shape[-2], p_pred.shape[-1]
            delta = ailoc.common.gpu(ailoc.common.sample_prob(p_pred,
                                                              batch_size))[:, None].expand(-1, num_sample, -1, -1)
            xyzph_sample = torch.distributions.Normal(
                loc=(xyzph_pred.permute([1, 0, 2, 3])[:, :, None]).expand(-1, -1, num_sample, -1, -1),
                scale=(xyzph_sig_pred.permute([1, 0, 2, 3])[:, :, None]).expand(-1, -1, num_sample, -1, -1)).sample()
            xyzph_sample[0] = torch.clamp(xyzph_sample[0], min=-self.learned_psf.psf_size//2, max=self.learned_psf.psf_size//2)
            xyzph_sample[1] = torch.clamp(xyzph_sample[1], min=-self.learned_psf.psf_size//2, max=self.learned_psf.psf_size//2)
            xyzph_sample[2] = torch.clamp(xyzph_sample[2], min=-1.5, max=1.5)
            xyzph_sample[3] = torch.clamp(xyzph_sample[3], min=0.0, max=3.0)
            bg_sample = bg_pred.detach()

        return delta, xyzph_sample, bg_sample

    @deprecated.deprecated('This function is deprecated')
    def prepare_sample_real_data(self, real_data, batch_size):
        self.sample_batch_size = self.context_size * batch_size

        n, h, w = real_data.shape
        assert n >= self.sample_batch_size and h >= self.data_simulator.mol_sampler.train_size \
               and w >= self.data_simulator.mol_sampler.train_size, 'real data is too small'

        self.sample_window_size = min(min(h // 4 * 4, w // 4 * 4), 256)

        self.h_sample_prob = real_data[:, :h - self.sample_window_size + 1, :].mean(axis=(0, 2)) / \
                             np.sum(real_data[:, :h - self.sample_window_size + 1, :].mean(axis=(0, 2)))

        self.w_sample_prob = real_data[:, :, :w - self.sample_window_size + 1].mean(axis=(0, 1)) / \
                             np.sum(real_data[:, :, :w - self.sample_window_size + 1].mean(axis=(0, 1)))

    @deprecated.deprecated('This function is deprecated')
    def sample_real_data(self, real_data, n_img):
        n, h, w = real_data.shape

        n_start = np.random.randint(0, n-n_img+1)
        h_start = np.random.choice(np.arange(h-self.sample_window_size+1), size=1, p=self.h_sample_prob)[0]
        w_start = np.random.choice(np.arange(w-self.sample_window_size+1), size=1, p=self.w_sample_prob)[0]
        real_data_sample = real_data[n_start: n_start+n_img,
                                     h_start: h_start + self.sample_window_size,
                                     w_start: w_start + self.sample_window_size]
        return real_data_sample.astype(np.float32)

    @deprecated.deprecated('This function is deprecated')
    def get_real_data_z_prior(self, real_data, batch_size):
        print('-' * 200)
        print(f'Using the current network to estimate the z prior of the real data {real_data.shape}...')

        self.network.eval()
        with torch.no_grad():
            n, h, w = real_data.shape
            fov_xy = (0, w-1, 0, h-1)
            molecule_list = []
            for i in tqdm(range(int(np.ceil(n / batch_size)))):
                sub_fov_data_list, \
                sub_fov_xy_list, \
                original_sub_fov_xy_list = ailoc.common.split_fov(data=real_data[i * batch_size: (i + 1) * batch_size],
                                                                  fov_xy=fov_xy,
                                                                  sub_fov_size=256,
                                                                  over_cut=8)
                sub_fov_molecule_list = []
                for i_fov in range(len(sub_fov_xy_list)):
                    with torch.cuda.amp.autocast():
                        molecule_list_tmp, inference_dict_tmp = ailoc.common.data_analyze(loc_model=self,
                                                                                          data=sub_fov_data_list[i_fov],
                                                                                          sub_fov_xy=sub_fov_xy_list[i_fov],
                                                                                          camera=self.data_simulator.camera,
                                                                                          batch_size=batch_size,
                                                                                          retain_infer_map=False)
                    sub_fov_molecule_list.append(molecule_list_tmp)

                # merge the localizations in each sub-FOV to whole FOV, filter repeated localizations in over cut region
                molecule_list_block = ailoc.common.SmlmDataAnalyzer.filter_over_cut(sub_fov_molecule_list,
                                                                                    sub_fov_xy_list,
                                                                                    original_sub_fov_xy_list,
                                                                                    ailoc.common.cpu(self.data_simulator.psf_model.pixel_size_xy))
                molecule_list += molecule_list_block

            molecule_list = np.array(molecule_list)

            try:
                z_pdf = list(np.histogram(molecule_list[:, 3],
                                          range=self.data_simulator.mol_sampler.z_range,
                                          bins=self.z_bins, density=True))
                z_pdf[0] *= (self.data_simulator.mol_sampler.z_range[1]-self.data_simulator.mol_sampler.z_range[0])/self.z_bins
                z_pdf[0] = np.clip(z_pdf[0], a_min=1e-3, a_max=None)
                z_weight = copy.deepcopy(z_pdf)
                z_weight[0] = (1/z_weight[0])/np.sum(1/z_weight[0])
                z_weight[1] = z_weight[1]/self.data_simulator.mol_sampler.z_scale
                z_weight[1][0] = -np.inf
                z_weight[1][-1] = np.inf
                print(f'\nEstimation done, the z distribution of {molecule_list.shape[0]} '
                      f'molecules in real data is:')
                print(z_pdf[1])
                print(z_pdf[0])
                print('The corresponding z weight is:')
                print(z_weight[0])
            except IndexError:
                print('No molecule detected in the real data, please check the data or the network.')
                z_weight = self.real_data_z_weight
                print('Using default z weight:')
                print(z_weight[0])

        self.network.train()
        self.real_data_z_weight = z_weight

    @deprecated.deprecated('This function is deprecated')
    # @staticmethod
    def crop_patches(delta_map_sample,
                     real_data,
                     xyzph_map_sample,
                     bg_sample,
                     p_pred,
                     xyzph_pred,
                     xyzph_sig_pred,
                     use_threshold,
                     photon_threshold,
                     p_var_threshold,
                     crop_size,
                     max_psfs,
                     curr_num_psfs,
                     z_weight=None):
        """
        Crop the psf_patches on the canvas according to the delta map,
        the max_num is the maximum number of psfs to be used for wake training.
        """

        delta_inds = delta_map_sample[:, 0].nonzero().transpose(1, 0)

        if len(delta_inds[0]) == 0:
            return None, None, None, None, None, None, None

        if crop_size > delta_map_sample.shape[-1]:
            crop_size = delta_map_sample.shape[-1]

        # crop the psfs using the delta_inds, align the center pixel of the crop_size to the delta,
        # if the delta is in the margin area, shift the delta and crop the psf
        real_data_crop = []
        delta_map_sample_crop = []
        xyzph_map_sample_crop = []
        bg_sample_crop = []
        xyzph_pred_crop = []
        xyzph_sig_pred_crop = []
        z_weight_crop = []

        delta_idx_list = list(range(len(delta_inds[0])))
        photon_delta_list = xyzph_pred[:, 3][tuple(delta_inds)]
        sorted_numbers, indices = torch.sort(photon_delta_list, descending=False)
        delta_idx_list = [delta_idx_list[idx] for idx in indices]

        num_psfs = 0
        while len(delta_idx_list) > 0:
            if num_psfs+curr_num_psfs >= max_psfs:
                break

            # random select a delta
            random_idx = random.sample(delta_idx_list, 1)[0]
            delta_idx_list.remove(random_idx)

            # # pop the brightest delta
            # random_idx = delta_idx_list.pop()

            frame_num, center_h, center_w = delta_inds[:, random_idx]

            # set the crop center, considering the margin area
            if center_h < crop_size // 2:
                center_h = crop_size // 2
            elif center_h > real_data.shape[1] - 1 - (crop_size-crop_size // 2-1):
                center_h = real_data.shape[1] - 1 - (crop_size-crop_size // 2-1)
            if center_w < crop_size // 2:
                center_w = crop_size // 2
            elif center_w > real_data.shape[2] - 1 - (crop_size-crop_size // 2-1):
                center_w = real_data.shape[2] - 1 - (crop_size-crop_size // 2-1)
            # set the crop range,
            h_range_tmp = (center_h - crop_size // 2, center_h - crop_size // 2 + crop_size)
            w_range_tmp = (center_w - crop_size // 2, center_w - crop_size // 2 + crop_size)

            # remove all delta in the crop area
            curr_frame_delta = torch.where(delta_inds[0, :] == frame_num, delta_inds[1:, :], -1)
            delta_inds_in_crop = (torch.eq(h_range_tmp[0] <= curr_frame_delta[0, :],
                                           curr_frame_delta[0, :] < h_range_tmp[1]) *
                                  torch.eq(w_range_tmp[0] <= curr_frame_delta[1, :],
                                           curr_frame_delta[1, :] < w_range_tmp[1])).nonzero().tolist()

            for j in delta_inds_in_crop:
                try:
                    delta_idx_list.remove(j[0])
                except ValueError:
                    pass

            tmp_p_pred_crop = p_pred[frame_num,
                              h_range_tmp[0]: h_range_tmp[1], w_range_tmp[0]: w_range_tmp[1]]
            tmp_real_data_crop = real_data[frame_num, h_range_tmp[0]: h_range_tmp[1], w_range_tmp[0]: w_range_tmp[1]]
            tmp_delta_map_sample_crop = delta_map_sample[frame_num, :,
                                         h_range_tmp[0]: h_range_tmp[1], w_range_tmp[0]: w_range_tmp[1]]
            tmp_xyzph_map_sample_crop = xyzph_map_sample[:, frame_num, :,
                                         h_range_tmp[0]: h_range_tmp[1], w_range_tmp[0]: w_range_tmp[1]]
            tmp_bg_sample_crop = bg_sample[frame_num,
                                  h_range_tmp[0]: h_range_tmp[1], w_range_tmp[0]: w_range_tmp[1]]
            tmp_xyzph_pred_crop = xyzph_pred[frame_num, :,
                                   h_range_tmp[0]: h_range_tmp[1], w_range_tmp[0]: w_range_tmp[1]]
            tmp_xyzph_sig_pred_crop = xyzph_sig_pred[frame_num, :,
                                       h_range_tmp[0]: h_range_tmp[1], w_range_tmp[0]: w_range_tmp[1]]

            # check the crop area quality to ensure a better PSF learning
            if use_threshold:
                tmp_delta_inds = tuple(tmp_delta_map_sample_crop[0].nonzero().transpose(1, 0))
                tmp_p_pred_var = (tmp_p_pred_crop - tmp_p_pred_crop**2).mean()
                tmp_photons = tmp_xyzph_pred_crop[3][tmp_delta_inds]
                # flag_1 filters the average photons
                # flag_2 filters the detection probability
                flag_1 = (tmp_photons.mean() >= photon_threshold)
                flag_2 = (tmp_p_pred_var < p_var_threshold)
                if not (flag_1 & flag_2):
                    continue
                    # ailoc.common.plot_image_stack(torch.concatenate([tmp_real_data_crop[None], tmp_p_pred_crop[None], tmp_delta_map_sample_crop[0:1]]))

            real_data_crop.append(tmp_real_data_crop)
            delta_map_sample_crop.append(tmp_delta_map_sample_crop)
            xyzph_map_sample_crop.append(tmp_xyzph_map_sample_crop)
            bg_sample_crop.append(tmp_bg_sample_crop)
            xyzph_pred_crop.append(tmp_xyzph_pred_crop)
            xyzph_sig_pred_crop.append(tmp_xyzph_sig_pred_crop)

            num_psfs += len(delta_map_sample_crop[-1].nonzero())

            if z_weight is not None:
                with torch.no_grad():
                    # # version 1: use the average z to index the z weight
                    # delta_inds_for_z = tuple(delta_map_sample_crop[-1][0].nonzero().transpose(1, 0))
                    # z_avg_crop = xyzph_pred_crop[-1][2][delta_inds_for_z].mean()
                    # def find_weight(intervals, number):
                    #     for i in range(len(intervals) - 1):
                    #         if intervals[i] <= number < intervals[i + 1]:
                    #             return i  # returning the interval index (0-based)
                    # z_weight_tmp = z_weight[0][find_weight(z_weight[1], z_avg_crop)]
                    # z_weight_crop.append(ailoc.common.gpu(z_weight_tmp))

                    # version 2: use the average z weight
                    delta_inds_for_z = tuple(delta_map_sample_crop[-1][0].nonzero().transpose(1, 0))
                    z_crop = ailoc.common.cpu(xyzph_pred_crop[-1][2][delta_inds_for_z])
                    z_weight_tmp = np.average(z_weight[0][np.digitize(z_crop, z_weight[1])-1])
                    z_weight_crop.append(ailoc.common.gpu(z_weight_tmp))

        if len(real_data_crop) == 0:
            return None, None, None, None, None, None, None

        return torch.stack(real_data_crop), \
               torch.stack(delta_map_sample_crop), \
               torch.permute(torch.stack(xyzph_map_sample_crop), dims=(1, 0, 2, 3, 4)), \
               torch.stack(bg_sample_crop), \
               torch.stack(xyzph_pred_crop), \
               torch.stack(xyzph_sig_pred_crop), \
               torch.stack(z_weight_crop) if z_weight is not None else None

    @staticmethod
    def count_intervals(mol_list, intervals):
        counts = collections.Counter()
        for mol in mol_list:
            z = mol[3]
            for i in range(len(intervals) - 1):
                if intervals[i] <= z < intervals[i + 1]:
                    counts[(intervals[i], intervals[i + 1])] += 1
                    break
        return counts

    def z_partition(self, chunk_list, current_counts, target_counts, intervals, replacement=True):
        """
        Partition the input roi_list,
        select ROIs from the input to construct a new_roi_list to approach the target z interval counts
        """

        # vectorized for better computation
        target_array = np.array([target_counts[interval] for interval in target_counts])

        # calculate all chunk's z counts for quick selection
        chunk_z_counts_list = []
        for i, chunk in enumerate(chunk_list):
            mol_array_tmp = chunk[0]
            tmp_z_counts = self.count_intervals(mol_array_tmp, intervals)
            tmp_z_counts = [tmp_z_counts[interval] for interval in target_counts]
            tmp_z_counts.append(i)  # add the chunk number at the end
            chunk_z_counts_list.append(np.array(tmp_z_counts))  # transform the list into array
        chunk_z_counts_array = np.array(chunk_z_counts_list)  # collect all z counts array for fast computation

        # Create the new list
        new_chunk_list = []
        new_chunk_z_counts = collections.Counter()
        for interval in target_counts:
            new_chunk_z_counts[interval] = 0
        current_z_counts = collections.Counter() + current_counts

        if len(chunk_z_counts_array) == 0:
            return new_chunk_list, new_chunk_z_counts, current_z_counts

        while True:
            # calculate the current state and the score
            current_z_array = np.array([current_z_counts[interval] for interval in target_counts])
            best_score = np.sum((target_array - current_z_array) ** 2)
            # calculate current roi list plus all candidate roi, and the score
            tmp_chunk_z_array = current_z_array + chunk_z_counts_array[:, :-1]
            tmp_chunk_score = np.sum((target_array - tmp_chunk_z_array) ** 2, axis=1)

            # Check if any improvement,
            if any(tmp_chunk_score < best_score):
                # get the best chunk
                tmp_indices = np.where(tmp_chunk_score == tmp_chunk_score.min())[0]
                best_chunk_array_idx = tmp_indices[np.random.randint(0, tmp_indices.size)]
                best_chunk_list_idx = chunk_z_counts_array[best_chunk_array_idx][-1]
                best_chunk = chunk_list[best_chunk_list_idx]

                # update the new roi list and counts
                tmp_z_counts = self.count_intervals(best_chunk[0], intervals)
                current_z_counts.update(tmp_z_counts)
                new_chunk_z_counts.update(tmp_z_counts)
                new_chunk_list.append(best_chunk)

                # remove the selected roi from the chunk z counts array
                if not replacement:
                    chunk_z_counts_array = np.delete(chunk_z_counts_array, best_chunk_array_idx, axis=0)
            else:
                # print('no better roi found')
                break

            if sum(current_z_counts.values()) >= sum(target_counts.values()):
                # print('found number satisfies')
                break

        return new_chunk_list, new_chunk_z_counts, current_z_counts

    def update_roilib2learn(self, real_data, batch_size):
        """
        Use current network to analyze experimental data, build ROI library for physics learning.
        This creates a more balanced z distributed roi library from the experimental data.
        """

        t0 = time.time()
        print('-' * 200)
        print(f'Using the current network to build roilib2learn from the experimental data {real_data.shape}...')

        if self.temporal_attn:
            extra_length = self.attn_length//2
        else:
            extra_length = 0

        # first split the real data into sub-FOVs
        n, h, w = real_data.shape
        fov_xy = (0, w - 1, 0, h - 1)

        sub_fov_data_list, \
            sub_fov_xy_list, \
            original_sub_fov_xy_list = ailoc.common.split_fov(data=real_data,
                                                              fov_xy=fov_xy,
                                                              sub_fov_size=256,
                                                              over_cut=self.over_cut)

        # use current network to analyze the sub-fov data block
        sub_fov_molecule_list = []
        self.network.eval()
        with torch.no_grad():
            for i_fov in range(len(sub_fov_xy_list)):
                with torch.cuda.amp.autocast():
                    molecule_list_tmp, inference_dict_tmp = ailoc.common.data_analyze(loc_model=self,
                                                                                      data=sub_fov_data_list[i_fov],
                                                                                      sub_fov_xy=sub_fov_xy_list[i_fov],
                                                                                      camera=self.data_simulator.camera,
                                                                                      batch_size=batch_size,
                                                                                      retain_infer_map=False)
                sub_fov_molecule_list.append(molecule_list_tmp)
            # merge the localizations in each sub-FOV to whole FOV, filter repeated localizations in over cut region
            molecule_array = np.array(
                ailoc.common.SmlmDataAnalyzer.filter_over_cut(sub_fov_molecule_list,
                                                              sub_fov_xy_list,
                                                              original_sub_fov_xy_list,
                                                              ailoc.common.cpu(self.data_simulator.psf_model.pixel_size_xy)))

        assert len(molecule_array) > 0, ('No molecule detected in the real data, please check the data or the network.\n'
                                         'If data is OK, the network is probably far away from the true posterior, consider:\n'
                                         '1. Increase the model.warmup;\n'
                                         '2. Turn off robust training;\n'
                                         '3. Turn on the temporal attn for more powerful network;\n')

        # print the z and photon distribution
        z_hist_curr = np.histogram(molecule_array[:, 3], bins=10)
        photon_hist_curr = np.histogram(molecule_array[:, 4], bins=10)
        print(f'Current estimated z distribution of {molecule_array.shape[0]} molecules:')
        for i in range(len(z_hist_curr[0])):
            print(f'({z_hist_curr[1][i]:.0f}, {z_hist_curr[1][i + 1]:.0f}): {z_hist_curr[0][i]}')
        print(f'Current estimated photon distribution of {molecule_array.shape[0]} molecules:')
        for i in range(len(photon_hist_curr[0])):
            print(f'({photon_hist_curr[1][i]:.0f}, {photon_hist_curr[1][i + 1]:.0f}): {photon_hist_curr[0][i]}')

        # calculate the threshold to filter ROIs
        photon_low_thre = max(self.dict_sampler_params['photon_range'][0]*1.1,
                              np.quantile(molecule_array[:, 4], 0.2))
        photon_up_thre = self.dict_sampler_params['photon_range'][1]*0.95
        z_up_thre = self.dict_sampler_params['z_range'][1]-50
        z_low_thre = self.dict_sampler_params['z_range'][0]+50
        p_low_thre = 0.5

        self.roi_lib['sparse'] = np.array([])
        self.roi_lib['sparse_z'] = np.array([])
        self.roi_lib['sparse_z_counts'] = collections.Counter()
        self.roi_lib['dense'] = np.array([])
        self.roi_lib['dense_z_counts'] = collections.Counter()
        self.roi_lib['total_z_counts'] = collections.Counter()
        for i in range(self.z_bins):
            self.roi_lib['sparse_z_counts'][(self.intervals[i], self.intervals[i + 1])] = 0
            self.roi_lib['dense_z_counts'][(self.intervals[i], self.intervals[i + 1])] = 0
            self.roi_lib['total_z_counts'][(self.intervals[i], self.intervals[i + 1])] = 0

        # determine the sparse ROI size
        psf_size = self.dict_psf_params['psf_size']
        factor = 4
        if (psf_size % 4 != 0):
            psf_size = (psf_size // factor + 1) * factor
        sparse_roi_size = psf_size + 2 * self.over_cut
        min_dist = np.hypot(self.dict_psf_params['psf_size'] * self.dict_psf_params['pixel_size_xy'][0],
                            self.dict_psf_params['psf_size'] * self.dict_psf_params['pixel_size_xy'][1])

        # determine the dense ROI size, not larger than the data size
        dense_roi_size = min(min(h // 4 * 4, w // 4 * 4), 2*psf_size+2*self.over_cut)
        dense_frame_num = 1
        assert n-2*extra_length-dense_frame_num >= 0 and dense_roi_size-2*self.over_cut > 0, \
            f'The experimental data size {real_data.shape} is too small.'

        sparse_roi_list = []
        sparse_z_list = []
        # first find the sparse ROIs
        if self.roilib_sparse_first:
            progress = 0
            # randomly traverse frames
            for frame in random.sample(range(extra_length, n-extra_length), n-2*extra_length):
                progress += 1
                if progress % 1000 == 0:
                    print(f'\rSparse ROI progress: {progress/(n-2*extra_length)*100:.0f}%', end='')
                mol_this_frame = ailoc.common.find_molecules(molecule_array, frame+1)
                if len(mol_this_frame) == 0:
                    continue
                # check the sparsity
                dist_matrix = scipy.spatial.distance_matrix(mol_this_frame[:,1:3], mol_this_frame[:,1:3])
                keep_matrix_idxs = np.where((0 == dist_matrix) | (dist_matrix > min_dist))
                unique, counts = np.unique(keep_matrix_idxs[0], return_counts=True)
                sparse_idxs = unique[counts == mol_this_frame.shape[0]]
                if len(sparse_idxs) == 0:
                    continue
                # randomly traverse the sparse molecules and crop the ROI
                for mol in mol_this_frame[random.sample(sparse_idxs.tolist(), len(sparse_idxs))]:
                    z = mol[3]
                    photon = mol[4]
                    p = mol[5]
                    interval_idx = np.digitize(z, self.intervals) - 1
                    if (interval_idx < 0 or interval_idx >= self.z_bins
                            or z < z_low_thre or z > z_up_thre or photon < photon_low_thre or photon > photon_up_thre
                            or p < p_low_thre):
                        continue
                    interval_key = (self.intervals[interval_idx], self.intervals[interval_idx + 1])
                    # check the target z counts and the current z counts
                    if self.roi_lib['sparse_z_counts'][interval_key] < self.target_z_counts[interval_key]:
                        # crop the ROI
                        w_range = (int(mol[1]/self.dict_psf_params['pixel_size_xy'][0] - sparse_roi_size // 2),
                                   int(mol[1]/self.dict_psf_params['pixel_size_xy'][0] + sparse_roi_size // 2))
                        h_range = (int(mol[2]/self.dict_psf_params['pixel_size_xy'][1] - sparse_roi_size // 2),
                                   int(mol[2]/self.dict_psf_params['pixel_size_xy'][1] + sparse_roi_size // 2))
                        sparse_roi = real_data[frame-extra_length:frame+extra_length+1,
                                               h_range[0]: h_range[1],
                                               w_range[0]: w_range[1]][None]
                        if sparse_roi.shape[-1] != sparse_roi_size or sparse_roi.shape[-2] != sparse_roi_size:
                            continue
                        # update the roi library
                        sparse_roi_list.append(sparse_roi)
                        sparse_z_list.append(z)
                        self.roi_lib['sparse_z_counts'][interval_key] += 1

                if sum(self.roi_lib['sparse_z_counts'].values()) >= sum(self.target_z_counts.values()):
                    break

            print(f'\rSparse ROI progress: 100%')

            self.roi_lib['sparse'] = np.concatenate(sparse_roi_list, axis=0) if sparse_roi_list else np.array([])
            self.roi_lib['sparse_z'] = np.array(sparse_z_list)
            self.roi_lib['total_z_counts'].update(self.roi_lib['sparse_z_counts'])

        # then find the dense ROIs
        if sum(self.roi_lib['total_z_counts'].values()) < sum(self.target_z_counts.values()):
            # first define the split setting of the data by dense ROI size and chunk frame number
            split_chunk_list = []
            row_sub_fov = int(np.ceil(h / dense_roi_size))
            col_sub_fov = int(np.ceil(w / dense_roi_size))
            progress = 0
            for frame in range(extra_length, n - extra_length - dense_frame_num, dense_frame_num):
                progress += 1
                if progress % 1000 == 0:
                    print(f'\rDense ROI progress: '
                          f'{progress/((n-2*extra_length-dense_frame_num)/dense_frame_num)*100:.0f}%', end='')
                frame_range = range(frame+1, frame+1 + dense_frame_num)
                mol_these_frames = ailoc.common.find_molecules(molecule_array, list(frame_range))
                if len(mol_these_frames) == 0:
                    continue
                # split the molecules into sub-FOVs
                for raw in range(row_sub_fov):
                    for col in range(col_sub_fov):
                        x_start = col * dense_roi_size if col * dense_roi_size + dense_roi_size <= w else w - dense_roi_size
                        y_start = raw * dense_roi_size if raw * dense_roi_size + dense_roi_size <= h else h - dense_roi_size
                        x_end = x_start + dense_roi_size
                        y_end = y_start + dense_roi_size
                        # find the molecules in this sub-FOV
                        x_range = ((x_start+self.over_cut)*self.dict_psf_params['pixel_size_xy'][0],
                                   (x_end-self.over_cut)*self.dict_psf_params['pixel_size_xy'][0])
                        y_range = ((y_start+self.over_cut)*self.dict_psf_params['pixel_size_xy'][1],
                                   (y_end-self.over_cut)*self.dict_psf_params['pixel_size_xy'][1])
                        mol_this_chunk = mol_these_frames[
                            (mol_these_frames[:, 1] >= x_range[0]) & (mol_these_frames[:, 1] <= x_range[1]) &
                            (mol_these_frames[:, 2] >= y_range[0]) & (mol_these_frames[:, 2] <= y_range[1])]
                        if len(mol_this_chunk) == 0:
                            continue
                        p_avg = mol_this_chunk[:, 5].mean()
                        photon_max = mol_this_chunk[:, 4].max()
                        photon_min = mol_this_chunk[:, 4].min()
                        z_max = mol_this_chunk[:, 3].max()
                        z_min = mol_this_chunk[:, 3].min()
                        if (p_avg < p_low_thre or
                            photon_min < photon_low_thre or
                            photon_max > photon_up_thre or
                            z_min < z_low_thre or
                            z_max > z_up_thre):
                            continue
                        # calculate the z counts of this chunk
                        split_chunk_list.append((mol_this_chunk,
                                                 (frame, frame+dense_frame_num),
                                                 (x_start, x_end),
                                                 (y_start, y_end)))

            print('\rDense ROI progress: 100%')

            # select the split chunks to form a new list with more balanced z distribution
            selected_chunks, selected_chunks_z_counts, total_z_counts = self.z_partition(split_chunk_list,
                                                                                         self.roi_lib['total_z_counts'],
                                                                                         self.target_z_counts,
                                                                                         self.intervals,
                                                                                         replacement=False)
            if len(selected_chunks):
                # build the dense ROI library
                dense_roi_list = []
                for chunk in selected_chunks:
                    frame_start, frame_end = chunk[1][0]-extra_length, chunk[1][1]+extra_length
                    x_start, x_end = chunk[2]
                    y_start, y_end = chunk[3]
                    dense_roi_list.append(real_data[frame_start:frame_end, y_start:y_end, x_start:x_end][None])

                self.roi_lib['dense'] = np.concatenate(dense_roi_list, axis=0)
                self.roi_lib['dense_z_counts'] = selected_chunks_z_counts
                self.roi_lib['total_z_counts'].update(self.roi_lib['dense_z_counts'])

        # if far away from the target z counts, loosen the filter requirements to rebuild the dense ROI library
        if sum(self.roi_lib['total_z_counts'].values()) < 0.5 * sum(self.target_z_counts.values()):
            print('The current ROI library is far away from the target z counts, '
                  'loosen the filter requirements to rebuild the dense ROI library...')

            # calculate the threshold to filter ROIs
            photon_low_thre = self.dict_sampler_params['photon_range'][0]
            photon_up_thre = self.dict_sampler_params['photon_range'][1]
            z_up_thre = self.dict_sampler_params['z_range'][1]
            z_low_thre = self.dict_sampler_params['z_range'][0]
            p_low_thre = 0.5

            # reset the dense ROI library
            self.roi_lib['dense'] = np.array([])
            self.roi_lib['dense_z_counts'] = collections.Counter()
            self.roi_lib['total_z_counts'] = collections.Counter()
            for i in range(self.z_bins):
                self.roi_lib['dense_z_counts'][(self.intervals[i], self.intervals[i + 1])] = 0
                self.roi_lib['total_z_counts'][(self.intervals[i], self.intervals[i + 1])] = 0
            self.roi_lib['total_z_counts'].update(self.roi_lib['sparse_z_counts'])

            # first define the split setting of the data by dense ROI size and chunk frame number
            split_chunk_list = []
            row_sub_fov = int(np.ceil(h / dense_roi_size))
            col_sub_fov = int(np.ceil(w / dense_roi_size))
            progress = 0
            for frame in range(extra_length, n - extra_length - dense_frame_num, dense_frame_num):
                progress += 1
                if progress % 1000 == 0:
                    print(f'\rDense ROI progress: '
                          f'{progress/((n-2*extra_length-dense_frame_num)/dense_frame_num)*100:.0f}%', end='')
                frame_range = range(frame+1, frame+1 + dense_frame_num)
                mol_these_frames = ailoc.common.find_molecules(molecule_array, list(frame_range))
                if len(mol_these_frames) == 0:
                    continue
                # split the molecules into sub-FOVs
                for raw in range(row_sub_fov):
                    for col in range(col_sub_fov):
                        x_start = col * dense_roi_size if col * dense_roi_size + dense_roi_size <= w else w - dense_roi_size
                        y_start = raw * dense_roi_size if raw * dense_roi_size + dense_roi_size <= h else h - dense_roi_size
                        x_end = x_start + dense_roi_size
                        y_end = y_start + dense_roi_size
                        # find the molecules in this sub-FOV
                        x_range = ((x_start+self.over_cut)*self.dict_psf_params['pixel_size_xy'][0],
                                   (x_end-self.over_cut)*self.dict_psf_params['pixel_size_xy'][0])
                        y_range = ((y_start+self.over_cut)*self.dict_psf_params['pixel_size_xy'][1],
                                   (y_end-self.over_cut)*self.dict_psf_params['pixel_size_xy'][1])
                        mol_this_chunk = mol_these_frames[
                            (mol_these_frames[:, 1] >= x_range[0]) & (mol_these_frames[:, 1] <= x_range[1]) &
                            (mol_these_frames[:, 2] >= y_range[0]) & (mol_these_frames[:, 2] <= y_range[1])]
                        if len(mol_this_chunk) == 0:
                            continue
                        p_avg = mol_this_chunk[:, 5].mean()
                        photon_max = mol_this_chunk[:, 4].max()
                        photon_min = mol_this_chunk[:, 4].min()
                        z_max = mol_this_chunk[:, 3].max()
                        z_min = mol_this_chunk[:, 3].min()
                        if (p_avg < p_low_thre or
                            photon_min < photon_low_thre or
                            photon_max > photon_up_thre or
                            z_min < z_low_thre or
                            z_max > z_up_thre):
                            continue
                        # calculate the z counts of this chunk
                        split_chunk_list.append((mol_this_chunk,
                                                 (frame, frame+dense_frame_num),
                                                 (x_start, x_end),
                                                 (y_start, y_end)))

            print('\rDense ROI progress: 100%')

            # select the split chunks to form a new list with more balanced z distribution
            selected_chunks, selected_chunks_z_counts, total_z_counts = self.z_partition(split_chunk_list,
                                                                                         self.roi_lib['total_z_counts'],
                                                                                         self.target_z_counts,
                                                                                         self.intervals,
                                                                                         replacement=False)
            if len(selected_chunks):
                # build the dense ROI library
                dense_roi_list = []
                for chunk in selected_chunks:
                    frame_start, frame_end = chunk[1][0]-extra_length, chunk[1][1]+extra_length
                    x_start, x_end = chunk[2]
                    y_start, y_end = chunk[3]
                    dense_roi_list.append(real_data[frame_start:frame_end, y_start:y_end, x_start:x_end][None])

                self.roi_lib['dense'] = np.concatenate(dense_roi_list, axis=0)
                self.roi_lib['dense_z_counts'] = selected_chunks_z_counts
                self.roi_lib['total_z_counts'].update(self.roi_lib['dense_z_counts'])

        print(f'ROI library built, cost {time.time()-t0}s, '
              f'sparse ROIs: {self.roi_lib["sparse"].shape if len(self.roi_lib["sparse"])>0 else len(self.roi_lib["sparse"])}, '
              f'dense ROIs: {self.roi_lib["dense"].shape if len(self.roi_lib["dense"])>0 else len(self.roi_lib["dense"])}')
        print(f'total z counts: ')
        for key in self.roi_lib['total_z_counts'].keys():
            print(f'{key}: {self.roi_lib["total_z_counts"][key]}')

        self.roi_lib_test = copy.deepcopy(self.roi_lib) if self.roi_lib_test is None else self.roi_lib_test

    def physics_learning(self, batch_size, num_sample=50, max_recon_psfs=5000):
        """
        for each physics learning iteration, using the current network to localization enough signals, and then
        use this signals and posterior samples to update the PSF model
        """

        # randomly select the sparse or dense roi library for learning
        sparse_prob = sum(self.roi_lib['sparse_z_counts'].values()) / sum(self.roi_lib['total_z_counts'].values())
        sparse_flag = random.random() < sparse_prob
        if sparse_flag:
            lib_length = len(self.roi_lib['sparse'])
            sample_batch_size = max(1,
                                    (batch_size * self.dict_sampler_params['train_size'] ** 2) //
                                    (self.attn_length *
                                     self.roi_lib['sparse'].shape[-1]
                                     * self.roi_lib['sparse'].shape[-2])
                                    )
        else:
            lib_length = len(self.roi_lib['dense'])
            sample_batch_size = max(1,
                                    (batch_size * self.dict_sampler_params['train_size'] ** 2) //
                                    (self.roi_lib['dense'].shape[-3] *
                                     self.roi_lib['dense'].shape[-1]
                                     * self.roi_lib['dense'].shape[-2]) // 2  # in case the density is too high
                                    )

        real_data_sample_list = []
        delta_map_sample_list = []
        xyzph_map_sample_list = []
        bg_sample_list = []
        xyzph_pred_list = []
        xyzph_sig_pred_list = []
        infer_round = 0
        patience = 20
        num_psfs = 0

        self.network.eval()
        with torch.no_grad():
            while num_psfs < max_recon_psfs and infer_round < patience:
                # random select data in the library
                start_idx = np.random.randint(0, max(0, lib_length-sample_batch_size)+1)
                end_idx = start_idx + sample_batch_size
                if sparse_flag:
                    real_data_sampled = ailoc.common.gpu(ailoc.common.cpu(self.roi_lib['sparse'][start_idx: end_idx]))
                else:
                    real_data_sampled = ailoc.common.gpu(ailoc.common.cpu(self.roi_lib['dense'][start_idx: end_idx]))

                p_pred, xyzph_pred, xyzph_sig_pred, bg_pred = self.inference(real_data_sampled,
                                                                             self.data_simulator.camera)
                delta_map_sample, xyzph_map_sample, bg_sample = self.sample_posterior(p_pred,
                                                                                      xyzph_pred,
                                                                                      xyzph_sig_pred,
                                                                                      bg_pred,
                                                                                      num_sample)

                if len(delta_map_sample.nonzero()) > 0.05 * p_pred.shape[0] * p_pred.shape[1] * p_pred.shape[
                    2] * num_sample:
                    print('too many non-zero elements in delta_map_sample, the network probably diverges, '
                          'consider decreasing the PSF learning rate to make the network more stable')
                    return np.nan

                real_data_sampled = self.data_simulator.camera.backward(real_data_sampled)
                if self.temporal_attn and self.attn_length // 2 > 0:
                    real_data_sampled = real_data_sampled[:, (self.attn_length // 2): -(self.attn_length // 2)]
                real_data_sampled = real_data_sampled.reshape(-1, real_data_sampled.shape[-2], real_data_sampled.shape[-1])

                # # consider the overcut
                # real_data_sampled = real_data_sampled[:, self.over_cut: -self.over_cut, self.over_cut: -self.over_cut]
                # delta_map_sample = delta_map_sample[:, :, self.over_cut: -self.over_cut, self.over_cut: -self.over_cut]
                # xyzph_map_sample = xyzph_map_sample[:, :, :, self.over_cut: -self.over_cut, self.over_cut: -self.over_cut]
                # bg_sample = bg_sample[:, self.over_cut: -self.over_cut, self.over_cut: -self.over_cut]
                # xyzph_pred = xyzph_pred[:, :, self.over_cut: -self.over_cut, self.over_cut: -self.over_cut]
                # xyzph_sig_pred = xyzph_sig_pred[:, :, self.over_cut: -self.over_cut, self.over_cut: -self.over_cut]

                real_data_sample_list.append(real_data_sampled)
                delta_map_sample_list.append(delta_map_sample)
                xyzph_map_sample_list.append(xyzph_map_sample)
                bg_sample_list.append(bg_sample)
                xyzph_pred_list.append(xyzph_pred)
                xyzph_sig_pred_list.append(xyzph_sig_pred)

                infer_round += 1
                num_psfs += len(delta_map_sample.nonzero())

        self.network.train()

        # in case no emitters are detected using current network
        if num_psfs == 0:
            return np.nan

        real_data_sample_list = torch.cat(real_data_sample_list, dim=0)
        delta_map_sample_list = torch.cat(delta_map_sample_list, dim=0)
        xyzph_map_sample_list = torch.cat(xyzph_map_sample_list, dim=1)
        bg_sample_list = torch.cat(bg_sample_list, dim=0)
        # smooth the bg to reduce the noise hampering the PSF learning
        bg_sample_list = torch.nn.functional.avg_pool2d(bg_sample_list[:, None], 9,
                                                        stride=1, padding=9//2, count_include_pad=False)[:, 0]
        xyzph_pred_list = torch.cat(xyzph_pred_list, dim=0)
        xyzph_sig_pred_list = torch.cat(xyzph_sig_pred_list, dim=0)

        # calculate the p_theta(x|h), q_phi(h|x) to compute the loss and optimize the psf using Adam
        reconstruction = self.data_simulator.reconstruct_posterior(self.learned_psf,
                                                                   delta_map_sample_list,
                                                                   xyzph_map_sample_list,
                                                                   bg_sample_list, )

        # consider the overcut
        real_data_sample_list = real_data_sample_list[:, self.over_cut: -self.over_cut, self.over_cut: -self.over_cut]
        reconstruction = reconstruction[:, :, self.over_cut: -self.over_cut, self.over_cut: -self.over_cut]
        delta_map_sample_list = delta_map_sample_list[:, :, self.over_cut: -self.over_cut, self.over_cut: -self.over_cut]
        xyzph_map_sample_list = xyzph_map_sample_list[:, :, :, self.over_cut: -self.over_cut, self.over_cut: -self.over_cut]
        xyzph_pred_list = xyzph_pred_list[:, :, self.over_cut: -self.over_cut, self.over_cut: -self.over_cut]
        xyzph_sig_pred_list = xyzph_sig_pred_list[:, :, self.over_cut: -self.over_cut, self.over_cut: -self.over_cut]

        loss, elbo_record = self.wake_loss(real_data_sample_list,
                                           reconstruction,
                                           delta_map_sample_list,
                                           xyzph_map_sample_list,
                                           xyzph_pred_list,
                                           xyzph_sig_pred_list,
                                           None)
        self.optimizer_psf.zero_grad()
        loss.backward()
        self.optimizer_psf.step()
        self.scheduler_psf.step()

        # update the psf simulator
        if isinstance(self.data_simulator.psf_model, ailoc.simulation.VectorPSFTorch):
            self.data_simulator.psf_model.zernike_coef = self.learned_psf.zernike_coef.detach()
        elif isinstance(self.data_simulator.psf_model, ailoc.simulation.VectorPSFCUDA):
            self.data_simulator.psf_model.zernike_coef = self.learned_psf.zernike_coef.detach().cpu().numpy()

        return elbo_record.detach().cpu().numpy()

    def check_zernike_convergence(self):
        """
        calculate the flatness of recent zernike coefficients to
        determine whether zernike coefficients are converged
        """

        # calculate the flatness of zernike coeffs
        flatness_thre = 0.08
        window_size = 20
        if self._iter_train < window_size*100 + self.warmup:
            return False

        zernike_history = []
        for (iter, zernike) in self.evaluation_recorder['learned_psf_zernike'].items():
            zernike_history.append(ailoc.common.cpu(zernike))

        window = zernike_history[len(zernike_history) - window_size:]
        std_window = np.std(window, axis=0)
        flatness = np.mean(std_window)
        if flatness < flatness_thre:
            print('\033[0;31m'+"Zernike coefficients converged, stop physics learning early"+'\033[0m')
            return True
        else:
            return False

    def compute_roilibtest_loss(self, batch_size):
        """
        Compute the negative log likelihood of the ROI library, using the current network and learned PSF
        to reconstruct the ROIs
        """

        if self.roi_lib_test is None:
            return np.nan, np.nan

        loss_list = []
        uncertainty_list = []
        self.network.eval()
        with torch.no_grad():
            for lib in ['sparse', 'dense']:
                if len(self.roi_lib_test[lib]) == 0:
                    continue
                lib_length = len(self.roi_lib_test[lib])
                lib_batch_size = max(1,
                            (batch_size * self.dict_sampler_params['train_size'] ** 2) //
                            (self.roi_lib_test[lib].shape[-3] *
                             self.roi_lib_test[lib].shape[-1]
                             * self.roi_lib_test[lib].shape[-2])
                            )

                for i in range(int(np.ceil(lib_length/ lib_batch_size))):
                    real_data_tmp = ailoc.common.gpu(ailoc.common.cpu(
                        self.roi_lib_test[lib][i * lib_batch_size: (i + 1) * lib_batch_size]))
                    p_pred, xyzph_pred, xyzph_sig_pred, bg_pred = self.inference(real_data_tmp,
                                                                                 self.data_simulator.camera)
                    n, h, w = p_pred.shape[0], p_pred.shape[-2], p_pred.shape[-1]
                    delta_map_sample = ailoc.common.gpu(ailoc.common.sample_prob(p_pred, n))[:, None].expand(-1, 1, -1, -1)
                    xyzph_map_sample = (xyzph_pred.permute([1, 0, 2, 3])[:, :, None]).expand(-1, -1, 1, -1, -1)
                    bg_sample = bg_pred.detach()

                    real_data_tmp = self.data_simulator.camera.backward(real_data_tmp)
                    if self.temporal_attn and self.attn_length // 2 > 0:
                        real_data_tmp = real_data_tmp[:, (self.attn_length // 2): -(self.attn_length // 2)]
                    real_data_tmp = real_data_tmp.reshape(-1, real_data_tmp.shape[-2], real_data_tmp.shape[-1])

                    # # consider the overcut
                    # real_data_tmp = real_data_tmp[:, self.over_cut: -self.over_cut, self.over_cut: -self.over_cut]
                    # delta_map_sample = delta_map_sample[:, :, self.over_cut: -self.over_cut, self.over_cut: -self.over_cut]
                    # xyzph_map_sample = xyzph_map_sample[:, :, :, self.over_cut: -self.over_cut, self.over_cut: -self.over_cut]
                    # bg_sample = bg_sample[:, self.over_cut: -self.over_cut, self.over_cut: -self.over_cut]

                    reconstruction = self.data_simulator.reconstruct_posterior(self.learned_psf,
                                                                               delta_map_sample,
                                                                               xyzph_map_sample,
                                                                               bg_sample, )

                    # consider the overcut
                    real_data_tmp = real_data_tmp[:, self.over_cut: -self.over_cut, self.over_cut: -self.over_cut]
                    reconstruction = reconstruction[:, :, self.over_cut: -self.over_cut, self.over_cut: -self.over_cut]

                    # nll_list.append(-ailoc.lunar.compute_log_p_x_given_h(data=real_data_tmp[:, None].expand(-1, 1, -1, -1),
                    #                                            model=reconstruction).mean().detach().cpu().numpy())
                    # use MSE
                    loss_list.append(torch.nn.MSELoss(reduction='none')(reconstruction, real_data_tmp[:, None]
                                                                        .expand(-1, 1, -1, -1)).mean().detach().cpu().numpy())

                    # also record the average posterior uncertainty on the roilib
                    pix_inds = tuple(delta_map_sample[:,0].nonzero().transpose(1, 0))
                    if len(pix_inds):
                        x_sig = xyzph_sig_pred[:,0][pix_inds] * self.data_simulator.psf_model.pixel_size_xy[0]
                        y_sig = xyzph_sig_pred[:,1][pix_inds] * self.data_simulator.psf_model.pixel_size_xy[1]
                        z_sig = xyzph_sig_pred[:,2][pix_inds] * self.data_simulator.mol_sampler.z_scale
                        uncertainty_list.append(torch.mean(torch.sqrt(x_sig**2+y_sig**2+z_sig**2)).detach().cpu().numpy())

        return np.mean(loss_list) if loss_list else np.nan, np.mean(uncertainty_list) if uncertainty_list else np.nan

    def plot_roilib_recon(self):
        """
        plot the reconstruction of the ROI library, choose one ROI for each z interval if sparse ROI,
        else randomly select 10 ROI from the dense ROI library.
        """

        print('-' * 200)
        print(f'Plot example ROIs and LUNAR reconstruction...')

        if self.temporal_attn:
            extra_length = self.attn_length//2
        else:
            extra_length = 0

        roi_list = []
        if len(self.roi_lib['sparse']) > 0:
            z_counts = collections.Counter()
            for i in range(self.z_bins):
                z_counts[(self.intervals[i], self.intervals[i + 1])] = 0
            for i in range(len(self.roi_lib['sparse'])):
                tmp_z = self.roi_lib['sparse_z'][i]
                interval_idx = np.digitize(tmp_z, self.intervals) - 1
                if interval_idx < 0 or interval_idx >= self.z_bins:
                    continue
                elif z_counts[(self.intervals[interval_idx], self.intervals[interval_idx + 1])] == 0:
                    z_counts[(self.intervals[interval_idx], self.intervals[interval_idx + 1])] = 1
                    roi_list.append([self.roi_lib['sparse'][i], tmp_z])
                else:
                    continue
        if len(roi_list) < self.z_bins:
            dense_idx = random.sample(range(len(self.roi_lib['dense'])), self.z_bins-len(roi_list))
            frame_idx = random.sample(range(extra_length, self.roi_lib['dense'].shape[-3]-extra_length), 1)[0]
            for i in dense_idx:
                roi_list.append([self.roi_lib['dense'][i, frame_idx-extra_length: frame_idx+extra_length+1], np.nan])

        recon_list = []
        roi_plot_list = []
        self.network.eval()
        with torch.no_grad():
            for roi, tmp_z in roi_list:
                real_data_tmp = ailoc.common.gpu(ailoc.common.cpu(roi))[None]
                p_pred, xyzph_pred, xyzph_sig_pred, bg_pred = self.inference(real_data_tmp,
                                                                             self.data_simulator.camera)
                n, h, w = p_pred.shape[0], p_pred.shape[-2], p_pred.shape[-1]
                delta_map_sample = ailoc.common.gpu(ailoc.common.sample_prob(p_pred, n))[:, None].expand(-1, 1, -1, -1)
                xyzph_map_sample = (xyzph_pred.permute([1, 0, 2, 3])[:, :, None]).expand(-1, -1, 1, -1, -1)
                bg_sample = bg_pred.detach()

                real_data_tmp = self.data_simulator.camera.backward(real_data_tmp)
                if extra_length > 0:
                    real_data_tmp = real_data_tmp[:, extra_length: -extra_length]
                real_data_tmp = real_data_tmp.reshape(-1, real_data_tmp.shape[-2], real_data_tmp.shape[-1])

                # # consider the overcut
                # real_data_tmp = real_data_tmp[:, self.over_cut: -self.over_cut, self.over_cut: -self.over_cut]
                # delta_map_sample = delta_map_sample[:, :, self.over_cut: -self.over_cut, self.over_cut: -self.over_cut]
                # xyzph_map_sample = xyzph_map_sample[:, :, :, self.over_cut: -self.over_cut, self.over_cut: -self.over_cut]
                # bg_sample = bg_sample[:, self.over_cut: -self.over_cut, self.over_cut: -self.over_cut]

                reconstruction = self.data_simulator.reconstruct_posterior(self.learned_psf,
                                                                           delta_map_sample,
                                                                           xyzph_map_sample,
                                                                           bg_sample, )

                # consider the overcut
                real_data_tmp = real_data_tmp[:, self.over_cut: -self.over_cut, self.over_cut: -self.over_cut]
                reconstruction = reconstruction[:, :, self.over_cut: -self.over_cut, self.over_cut: -self.over_cut]

                recon_list.append(ailoc.common.cpu(reconstruction))
                roi_plot_list.append([ailoc.common.cpu(real_data_tmp),tmp_z])

        # plot the data, model, error
        n_img = self.z_bins
        n_row = self.z_bins

        figure, ax_arr = plt.subplots(n_row, 1, constrained_layout=True, figsize=(6, 2 * n_row))
        figure.suptitle('Example ROI | Reconstruction | Error', fontsize=16)

        ax = []
        plts = []
        for i in ax_arr:
            try:
                for j in i:
                    ax.append(j)
            except:
                ax.append(i)

        for i in range(n_row):
            a_tmp = np.squeeze(roi_plot_list[i][0])
            b_tmp = np.squeeze(recon_list[i])
            mismatch_tmp = a_tmp - b_tmp
            image_tmp = np.concatenate([a_tmp, b_tmp, mismatch_tmp], axis=1)
            plts.append(ax[i].imshow(image_tmp, cmap='turbo'))
            plt.colorbar(mappable=plts[-1], ax=ax[i], fraction=0.046, pad=0.04)
            ax[i].set_title(f'Z(nm): {roi_plot_list[i][1]:.0f}', fontsize=16)
        plt.show()

    def print_recorder(self, max_iterations):
        try:
            print(f"Iterations: {self._iter_train}/{max_iterations} || "
                  f"Loss_sleep: {self.evaluation_recorder['loss_sleep'][self._iter_train]:.2f} || "
                  # f"Loss_wake: {self.evaluation_recorder['loss_wake'][self._iter_train]:.2f} || "
                  f"Loss_recon: {self.evaluation_recorder['loss_recon'][self._iter_train]:.2f} || "
                  # f"Sig_recon: {self.evaluation_recorder['avg_3Dsig_recon'][self._iter_train]:.2f} || "
                  f"IterTime: {self.evaluation_recorder['iter_time'][self._iter_train]:.2f} ms || "
                  f"ETA: {self.evaluation_recorder['iter_time'][self._iter_train] * (max_iterations - self._iter_train) / 3600000:.2f} h || ",
                  end='')

            print(f"SumProb: {self.evaluation_recorder['n_per_img'][self._iter_train]:.2f} || "
                  f"Eff_3D: {self.evaluation_recorder['eff_3d'][self._iter_train]:.2f} || "
                  f"Jaccard: {self.evaluation_recorder['jaccard'][self._iter_train]:.2f} || "
                  f"Recall: {self.evaluation_recorder['recall'][self._iter_train]:.2f} || "
                  f"Precision: {self.evaluation_recorder['precision'][self._iter_train]:.2f} || "
                  f"RMSE_lat: {self.evaluation_recorder['rmse_lat'][self._iter_train]:.2f} || "
                  f"RMSE_ax: {self.evaluation_recorder['rmse_ax'][self._iter_train]:.2f} || ")

            print(f"learned_psf_zernike: ", end="")
            for i in range(int(np.ceil(len(self.learned_psf.zernike_mode)/7))):
                for j in range(7):
                    print(f"{ailoc.common.cpu(self.learned_psf.zernike_mode)[i*7+j][0]:.0f},"
                          f"{ailoc.common.cpu(self.learned_psf.zernike_mode)[i*7+j][1]:.0f}:"
                          f"{self.evaluation_recorder['learned_psf_zernike'][self._iter_train][i*7+j]:.1f}", end='| ')

            print('')

        except KeyError:
            print('No record found')

    def save(self, file_name):
        """
        Save the whole LUNAR instance, including the network, optimizer, recorder, etc.
        """

        model_to_save = copy.deepcopy(self)
        # the roi_lib is not needed to be saved as it is too large
        model_to_save.roi_lib = {'sparse': np.array([]),
                                 'dense': np.array([]),
                                 'sparse_z_counts': collections.Counter(),
                                 'dense_z_counts': collections.Counter(),
                                 'total_z_counts': collections.Counter()}
        model_to_save.roi_lib_test = None
        with open(file_name, 'wb') as f:
            torch.save(model_to_save, f)
        print(f"LUNAR instance saved to {file_name}")

    def remove_gpu_attribute(self):
        """
        Remove the gpu attribute of the loc model, so that can be shared between processes.
        """

        self._network.to('cpu')
        self._network.tam.embedding_layer.attn_mask = None
        self.evaluation_dataset = None
        self.optimizer = None
        self.scheduler = None
        self.optimizer_psf = None
        self.scheduler_psf = None
        self._data_simulator = None
        self.learned_psf = None
