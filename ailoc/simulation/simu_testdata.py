from abc import ABC, abstractmethod
import torch
import torch.utils.data
from torch.cuda.amp import autocast
import numpy as np
import copy
import os
import time
import tifffile
import natsort
import pathlib
import csv
import matplotlib.pyplot as plt

import ailoc.common
import ailoc.simulation


class StructurePrior(ABC):
    """
    Abstract structure which can be sampled from. All implementation / childs must define a 'pop' method and an area
    property that describes the area the structure occupies.

    """

    @property
    @abstractmethod
    def area(self) -> float:
        """
        Calculate the area which is occupied by the structure. This is useful to later calculate the density,
        and the effective number of emitters). This is the 2D projection. Not the volume.

        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, n: int) -> torch.Tensor:
        """
        Sample n samples from structure.

        Args:
            n: number of samples

        """
        raise NotImplementedError


class RandomStructure(StructurePrior):
    """
    Random uniform 3D / 2D structure. As the name suggests, sampling from this structure gives samples from a 3D / 2D
    volume that origin from a uniform distribution.

    """

    def __init__(self, xextent, yextent, zextent):
        """
        Args:
            xextent: extent in x, unit is nm
            yextent: extent in y, unit is nm
            zextent: extent in z, set (0., 0.) for a 2D structure

        Example:
            The following initialises this class in a range of 6400 x 6400 nm in x and y and +/- 700nm in z.
            >>> prior_struct = RandomStructure(xextent=(0, 6400), yextent=(0, 6400), zextent=(-700., 700.))
        """

        super().__init__()
        self.xextent = xextent
        self.yextent = yextent
        self.zextent = zextent

        self.scale = torch.tensor([(self.xextent[1] - self.xextent[0]),
                                   (self.yextent[1] - self.yextent[0]),
                                   (self.zextent[1] - self.zextent[0])])

        self.shift = torch.tensor([self.xextent[0],
                                   self.yextent[0],
                                   self.zextent[0]])

    @property
    def area(self) -> float:
        return (self.xextent[1] - self.xextent[0]) * (self.yextent[1] - self.yextent[0])

    @property
    def area_um2(self) -> float:
        return self.area / 1e6

    def sample(self, n: int) -> torch.Tensor:
        xyz = torch.rand((n, 3)) * self.scale + self.shift
        return xyz


class SampleBlinkEvents:
    """
    Sample blinking events from a structure prior, the photophysics model only considers the liftime of an emitter.
    """
    def __init__(self, structure, photon_range, lifetime,
                 frame_range, pixel_size_xy, density,):
        """
        Args:
            structure: structure to sample from
            photon_range: range of photon value to sample from (uniformly)
            lifetime: average lifetime of the blink event
            frame_range: number of frames to simulate
            pixel_size_xy: pixel size in xy
            density: target emitter density, unit is um^-2
        """

        self.structure = structure
        self.photon_range = photon_range
        self.lifetime = lifetime
        self.frame_range = frame_range
        self.pixel_size_xy = pixel_size_xy
        self.density = density

        self.n_sampler = np.random.poisson
        self.photon_dist = torch.distributions.Uniform(*self.photon_range)
        self.lifetime_dist = torch.distributions.Exponential(1 / self.lifetime)  # parse the rate not the scale ...

        self.t0_dist = torch.distributions.uniform.Uniform(*self._frame_range_plus)

        self._emitter_per_frame = self.density * self.structure.area_um2

        """
        Determine the total number of emitters. Depends on lifetime, frames and emitters.
        (lifetime + 1) because of binning effect.
        """
        self._emitter_total = self._emitter_per_frame * self._num_frames_plus / (self.lifetime + 1)

    @property
    def _frame_range_plus(self):
        """
        Frame range including buffer in front and end to account for build up effects.
        """

        return self.frame_range[0] - 3 * self.lifetime, self.frame_range[1] + 3 * self.lifetime

    @property
    def num_frames(self):
        return self.frame_range[1] - self.frame_range[0] + 1

    @property
    def _num_frames_plus(self):
        return self._frame_range_plus[1] - self._frame_range_plus[0] + 1

    def sample(self):
        """
        Sample blink events (xyz photons and frame idxs).
        """

        """sample the number of fluorophores and their positions"""
        n = self.n_sampler(self._emitter_total)
        xyz = self.structure.sample(n)

        """Distribute emitters in time. Increase the range a bit."""
        t0 = self.t0_dist.sample(torch.Size((n,)))
        ontime = self.lifetime_dist.sample(torch.Size((n,)))
        te = t0 + ontime
        emitter_id = torch.arange(n).long()

        """Distribute emitters on frames"""
        photons = self.photon_dist.sample(torch.Size((n,)))
        xyz_, phot_, frame_ix_, id_ = self._distribute_framewise(xyz, t0, te, photons, emitter_id)

        """resample the photon from the distribution without considering the blinks, add more randomness"""
        phot_ = self.photon_dist.sample(phot_.size())

        """sort the blink events by frame number"""
        blink_events = torch.cat([id_[:, None], frame_ix_[:, None], xyz_, phot_[:,None]], dim=1)
        frame_tmp = blink_events[:, 1]
        sorted_indices = torch.argsort(frame_tmp)
        mol_list = blink_events[sorted_indices]

        """select the blink events in the target frame range"""
        mol_list = mol_list[mol_list[:, 1] >= self.frame_range[0]]
        mol_list = mol_list[mol_list[:, 1] < self.frame_range[1]]

        """return molecule list, shift the frame number from 1"""
        mol_list = ailoc.common.cpu(mol_list[:, 1:])
        mol_list[:, 0] += 1

        return mol_list

    def _distribute_framewise(self, xyz, t0, te, photons, emitter_id):
        """
        Distributes the emitters framewise.
        """

        frame_start = torch.floor(t0).long()
        frame_last = torch.floor(te).long()
        frame_count = (frame_last - frame_start).long()

        """delete the first and last on-state frame"""
        frame_count_full = frame_count - 2
        ontime_first = torch.min(te - t0, frame_start + 1 - t0)
        ontime_last = torch.min(te - t0, te - frame_last)

        """kick out everything that has no full frame_duration"""
        ix_full = frame_count_full >= 0
        xyz_ = xyz[ix_full, :]
        flux_ = photons[ix_full]
        id_ = emitter_id[ix_full]
        frame_start_full = frame_start[ix_full]
        frame_dur_full_clean = frame_count_full[ix_full]

        xyz_ = xyz_.repeat_interleave(frame_dur_full_clean + 1, dim=0)
        phot_ = flux_.repeat_interleave(frame_dur_full_clean + 1, dim=0)  # because intensity * 1 = phot
        id_ = id_.repeat_interleave(frame_dur_full_clean + 1, dim=0)
        # because 0 is first occurence
        frame_ix_ = frame_start_full.repeat_interleave(frame_dur_full_clean + 1, dim=0) \
                    + self.cum_count_per_group(id_) + 1

        """process first frame, partial duration"""
        xyz_ = torch.cat((xyz_, xyz), 0)
        phot_ = torch.cat((phot_, photons * ontime_first), 0)
        id_ = torch.cat((id_, emitter_id), 0)
        frame_ix_ = torch.cat((frame_ix_, frame_start), 0)

        """process last frame, partial duration, last (only if frame_last != frame_first)"""
        ix_with_last = frame_last >= frame_start + 1
        xyz_ = torch.cat((xyz_, xyz[ix_with_last]))
        phot_ = torch.cat((phot_, photons[ix_with_last] * ontime_last[ix_with_last]), 0)
        id_ = torch.cat((id_, emitter_id[ix_with_last]), 0)
        frame_ix_ = torch.cat((frame_ix_, frame_last[ix_with_last]))

        return xyz_, phot_, frame_ix_, id_

    def cum_count_per_group(self, arr: torch.Tensor):
        """
        Helper function that returns the cumulative sum per group.

        Example:
            [0, 0, 0, 1, 2, 2, 0] --> [0, 1, 2, 0, 0, 1, 3]
        """

        def grp_range(counts: torch.Tensor):
            """ToDo: Add docs"""
            assert counts.dim() == 1

            idx = counts.cumsum(0)
            id_arr = torch.ones(idx[-1], dtype=int)
            id_arr[0] = 0
            id_arr[idx[:-1]] = -counts[:-1] + 1
            return id_arr.cumsum(0)

        if arr.numel() == 0:
            return arr

        _, cnt = torch.unique(arr, return_counts=True)
        # ToDo: The following line in comment makes the test fail, replace once the torch implementation changes
        # return grp_range(cnt)[torch.argsort(arr).argsort()]
        return grp_range(cnt)[np.argsort(np.argsort(arr, kind='mergesort'), kind='mergesort')]


class TestDataSimulator:
    def __init__(self, psf_params_dict, camera_params_dict, emitter_params_dict, ):
        """
        Evaluator class for evaluating the localization model. Initialize this class using
        the PSF parameters, camera parameters and sampler parameters.
        """

        self.dict_psf_params, self.dict_camera_params, self.dict_emitter_params = \
            psf_params_dict, camera_params_dict, emitter_params_dict

        self.struct_prior = RandomStructure(
            xextent=self.dict_emitter_params['x_range'],
            yextent=self.dict_emitter_params['y_range'],
            zextent=self.dict_emitter_params['z_range']
        )

        self.blink_events_sampler = SampleBlinkEvents(
            structure=self.struct_prior,
            photon_range=self.dict_emitter_params['photon_range'],
            lifetime=self.dict_emitter_params['lifetime'],
            frame_range=(0, self.dict_emitter_params['frame_num']),
            density=self.dict_emitter_params['density'],
            pixel_size_xy=self.dict_psf_params['pixel_size_xy'],
        )

        self.psf_model = ailoc.simulation.VectorPSFTorch(self.dict_psf_params)

        self.camera = ailoc.simulation.instantiate_camera(self.dict_camera_params)

        self.bg_range = self.dict_emitter_params['bg_range']
        self.bg_perlin = self.dict_emitter_params['bg_perlin']

        self.dataset = {}

    def simulate_dataset(self,):
        """
        Simulate the dataset
        """
        t0 = time.time()

        mol_list = self.blink_events_sampler.sample()

        frame_num = self.dict_emitter_params['frame_num']
        psf_size = self.dict_psf_params['psf_size']
        image_size = self.dict_emitter_params['image_size']
        data_buffer = torch.zeros((frame_num, image_size, image_size), dtype=torch.float32)

        """simulate each frame"""
        for frame in range(1, frame_num+1):
            mol_this_frame = ailoc.common.find_molecules(mol_list, frame)
            if len(mol_this_frame) == 0:
                continue
            mol_x = mol_this_frame[:, 1]
            mol_y = mol_this_frame[:, 2]
            mol_z = mol_this_frame[:, 3]
            mol_photons = mol_this_frame[:, 4]

            mol_x_pix = np.floor(mol_x / self.dict_psf_params['pixel_size_xy'][0])
            mol_y_pix = np.floor(mol_y / self.dict_psf_params['pixel_size_xy'][1])

            mol_x_offset = mol_x - (mol_x_pix+0.5) * self.dict_psf_params['pixel_size_xy'][0]
            mol_y_offset = mol_y - (mol_y_pix+0.5) * self.dict_psf_params['pixel_size_xy'][1]

            psf_patches = self.psf_model.simulate(x=ailoc.common.gpu(mol_x_offset),
                                                  y=ailoc.common.gpu(mol_y_offset),
                                                  z=ailoc.common.gpu(mol_z),
                                                  photons=ailoc.common.gpu(mol_photons)).cpu()

            """process the edge situation"""
            canvas_x_start = np.clip(mol_x_pix - psf_size // 2, a_min=0, a_max=None).astype(int)
            canvas_y_start = np.clip(mol_y_pix - psf_size // 2, a_min=0, a_max=None).astype(int)

            psf_x_start = np.clip(psf_size // 2 - mol_x_pix, a_min=0, a_max=None).astype(int)
            psf_x_end = psf_size - np.clip(mol_x_pix + psf_size//2 + 1 - image_size, a_min=0, a_max=None).astype(int)

            psf_y_start = np.clip(psf_size // 2 - mol_y_pix, a_min=0, a_max=None).astype(int)
            psf_y_end = psf_size - np.clip(mol_y_pix + psf_size//2 + 1 - image_size, a_min=0, a_max=None).astype(int)

            """put the psf patches on the frame"""
            for i in range(len(mol_this_frame)):
                psf_patch = psf_patches[i, psf_y_start[i]:psf_y_end[i], psf_x_start[i]:psf_x_end[i]]
                data_buffer[frame-1,
                            canvas_y_start[i]:canvas_y_start[i]+psf_patch.shape[0],
                            canvas_x_start[i]:canvas_x_start[i]+psf_patch.shape[1]] += psf_patch

            if frame % 1000 == 0:
                print('Simulating frame %d / %d' % (frame, frame_num))

        """add background"""
        if self.bg_perlin:
            res = np.clip(image_size//64, a_min=1, a_max=None)
            perlin_noise = ailoc.simulation.generate_perlin_noise_2d((image_size, image_size), (res, res))
            perlin_noise = (perlin_noise - np.min(perlin_noise)) / (np.max(perlin_noise) - np.min(perlin_noise))
            bg = perlin_noise * (self.bg_range[1] - self.bg_range[0]) + self.bg_range[0]
        else:
            bg = np.random.uniform(self.bg_range[0], self.bg_range[1]) * np.ones_like(data_buffer[0])
        data_buffer += bg

        """add camera noise"""
        data_buffer = self.camera.forward(data_buffer)
        data_buffer = ailoc.common.cpu(data_buffer).astype(np.int16)

        print(f'Simulation time: {time.time() - t0:.2f} s, {len(mol_list)} molecules on {len(data_buffer)} frames')

        """save the dataset temporarily"""
        self.dataset['data'] = data_buffer
        self.dataset['molecule_list_gt'] = mol_list

    def save_datasets(self, data_path, gt_path):
        '''
        save the datasets to the disk
        '''

        dir = os.path.dirname(data_path)
        # create the directory if it does not exist
        pathlib.Path(dir).mkdir(parents=True, exist_ok=True)

        data = self.dataset['data']
        mol_list = self.dataset['molecule_list_gt']

        print(f'Saving data to {data_path},\nGround truth to {gt_path}')

        # save the data to a tiff file
        tifffile.imwrite(data_path, data.astype(np.uint16), imagej=True)

        # save the ground truth
        ailoc.common.write_csv_array(input_array=mol_list,
                                     filename=gt_path,
                                     write_mode='write simulated ground truth')

    def check_dataset(self, frame_num=None):
        """
        Check the dataset

        Args:
            frame_num: frame number to check, if None, a random frame will be selected, start from 1
        """
        if frame_num is None:
            frame_num = np.random.randint(1, self.dict_emitter_params['frame_num'])
        data = self.dataset['data'][frame_num-1]
        mol_list = self.dataset['molecule_list_gt']
        mol_this_frame = ailoc.common.find_molecules(mol_list, frame_num)
        if len(mol_this_frame) == 0:
            print(f'No molecules found in example frame {frame_num}.')

        else:
            mol_x = mol_this_frame[:, 1]
            mol_y = mol_this_frame[:, 2]
            mol_x_pix = np.floor(mol_x / self.dict_psf_params['pixel_size_xy'][0])
            mol_y_pix = np.floor(mol_y / self.dict_psf_params['pixel_size_xy'][1])

        """plot the image and gt"""
        cmap = 'gray'

        fig_base_size = data.shape[0] / 64 * 4

        im_num = 1
        n_col = 1
        n_row = int(np.ceil(im_num / n_col))

        fig, ax_arr = plt.subplots(n_row,
                                   n_col,
                                   figsize=(n_col * fig_base_size,
                                            n_row * fig_base_size),
                                   constrained_layout=True,
                                   dpi=300)
        ax = []
        plts = []
        try:
            for i in ax_arr:
                try:
                    for j in i:
                        ax.append(j)
                except:
                    ax.append(i)
        except:
            ax.append(ax_arr)

        for i in range(im_num):
            ax_num = i
            plts.append(ax[ax_num].imshow(data, cmap=cmap))
            ax[ax_num].set_title(f"Example frame {frame_num}")
            if len(mol_this_frame) > 0:
                ax[ax_num].scatter(mol_x_pix, mol_y_pix,
                                   s=16,
                                   marker='o',
                                   edgecolors='m',
                                   facecolors='none',
                                   linewidth=2,
                                   label='Ground truth')
            plt.colorbar(mappable=plts[-1], ax=ax[ax_num], fraction=0.046, pad=0.04)
            ax[ax_num].set_xlabel('X (pixel)')
            ax[ax_num].set_ylabel('Y (pixel)')
            # ax[ax_num].legend(loc='upper right', fontsize=10)
        plt.show()

    def check_psf(self, num_z_step=11):
        """
        Check the PSF.
        """
        print('checking pupil')
        pupil_phase = ailoc.common.cpu(2 * np.pi *
                                       torch.sum(self.psf_model.zernike_coef[:, None, None] *
                                                 self.psf_model.allzernikes, dim=0) /
                                       self.psf_model.wavelength)
        plt.figure(constrained_layout=True, dpi=300)
        plt.imshow(
            pupil_phase, cmap='turbo',
            # vmin=-1 * np.pi,
            # vmax=1 * np.pi
                   )
        plt.colorbar()
        plt.show()

        print(f"checking PSF...")
        x = ailoc.common.gpu(torch.zeros(num_z_step))
        y = ailoc.common.gpu(torch.zeros(num_z_step))
        z = ailoc.common.gpu(torch.linspace(*self.dict_emitter_params['z_range'], num_z_step))
        photons = ailoc.common.gpu(torch.ones(num_z_step))

        psf = ailoc.common.cpu(self.psf_model.simulate(x, y, z, photons))

        plt.figure(constrained_layout=True, dpi=300)
        for j in range(num_z_step):
            plt.subplot(int(np.ceil(num_z_step/11)), 11, j + 1)
            plt.imshow(psf[j], cmap='turbo')
            plt.title(f"{ailoc.common.cpu(z[j]):.0f} nm")
            plt.axis('off')
        plt.show()

    def eval_loc_model(self,):
        """
        Evaluate the localization model
        """
        raise NotImplementedError("eval_loc_model is not implemented yet")



