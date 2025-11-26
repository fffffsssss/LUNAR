import ctypes
import sys
from abc import ABC, abstractmethod  # abstract class
import torch
import torch.nn as nn
import numpy as np
import os
from deprecated import deprecated

import ailoc.common


class PSF(ABC):
    """
    Abstract base class for psf simulators
    """

    @abstractmethod
    def simulate(self, *args, **kwargs):
        """
        run the simulation
        """

        raise NotImplementedError

    def _setup_device(self, device: str):
        """Sets up the computation device, falling back to CPU if CUDA is unavailable."""
        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available. Switching to CPU.")
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)


class VectorPSF(PSF):
    """
    Abstract base class for vector psf simulators
    """

    @abstractmethod
    def simulate(self, *args, **kwargs):
        """
        run the simulation
        """

        raise NotImplementedError

    @staticmethod
    def _gauss2D_kernel(shape=(3, 3), sigmax=0.5, sigmay=0.5, data_type=torch.float32):
        """
        2D gaussian mask for VectorPSF otf rescale
        """

        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = torch.meshgrid(ailoc.common.gpu(torch.arange(-m, m + 1, 1), data_type=data_type),
                              ailoc.common.gpu(torch.arange(-n, n + 1, 1), data_type=data_type), indexing='ij')
        h = torch.exp(-(x * x) / (2. * sigmax * sigmax + 1e-6) - (y * y) / (2. * sigmay * sigmay + 1e-6))
        torch.clamp(h, min=0)
        # h[h < torch.finfo(h.dtype).eps * h.max()] = 0
        with torch.no_grad():
            sumh = h.sum()
        output = h / sumh if sumh != 0 else h
        return output

    @staticmethod
    def otf_rescale(psfdata, sigma_xy):
        """
        Convolve the psf with a gaussian kernel, namely otf rescale

        Args:
            psfdata (torch.Tensor): psf data to be blured
            sigma_xy (torch.Tensor): sigma x y of the gaussian kernel, unit pixel

        Returns:
            torch.Tensor: blured psf data
        """

        kernel = VectorPSF._gauss2D_kernel(shape=(5, 5),
                                           sigmax=sigma_xy[0],
                                           sigmay=sigma_xy[1],
                                           data_type=psfdata.dtype).reshape((1, 1, 5, 5))
        psf_size = psfdata.shape[1]
        psfdata = psfdata.view(-1, 1, psf_size, psf_size)
        tmp = nn.functional.conv2d(psfdata, kernel, padding=2, stride=1)
        outdata = tmp.view(-1, psf_size, psf_size)

        return outdata


class VectorPSFCUDA(VectorPSF):
    """
    Vector psf simulator using cuda, used for training due to its speed
    """

    def __init__(self, psf_params):
        """

        Args:
            psf_params (dict): PSF parameters for simulation except emitter positions.
                na: numerical aperture;
                wavelength: wavelength, unit nm;
                refmed: refractive index of the sample medium;
                refcov: refractive index of the coverslip;
                refimm: refractive index of the immersion oil;
                zernike_mode: zernike orders, 2D array, first column is radial order,
                    second column is azimuthal order;
                zernike_coef: amplitude for each zernike mode, unit nm;
                zernike_coef_map: spatially variant amplitude map for each zernike mode
                    with shape (num zernike, row, column), unit nm;
                objstage0: initial objStage position,relative to focus at coverslip, unit nm;
                zemit0: initial emitter z position, distance relative to coverslip, unit nm;
                pixel_size_xy: pixel sizes for x and y direction, unit nm;
                otf_rescale_xy: sigma x y of the gaussian kernel for otf rescale, unit pixel;
                npupil: number of pupil sampling points for simulation, unit pixel;
                psf_size: number of image pixels;

        Returns:
            VectorPSFCUDA: an instance of VectorPSFCUDA
        """

        self.na = psf_params['na']
        self.wavelength = psf_params['wavelength']

        self.refmed = psf_params['refmed']
        self.refcov = psf_params['refcov']
        self.refimm = psf_params['refimm']

        self.zernike_mode = psf_params['zernike_mode']
        try:
            self.zernike_coef = psf_params['zernike_coef']
        except KeyError:
            self.zernike_coef = None
        try:
            self.zernike_coef_map = psf_params['zernike_coef_map']
        except KeyError:
            self.zernike_coef_map = None
        if (self.zernike_coef is None) != (self.zernike_coef_map is None):
            pass
        else:
            raise ValueError('you must define either zernike_coef xor zernike_coef_map')

        self.pixel_size_xy = psf_params['pixel_size_xy']
        self.otf_rescale_xy = psf_params['otf_rescale_xy']

        self.npupil = psf_params['npupil']
        self.psf_size = psf_params['psf_size']

        self.objstage0 = psf_params['objstage0']
        try:
            self.zemit0 = psf_params['zemit0']
        except KeyError:
            self.zemit0 = -self.objstage0 / self.refimm * self.refmed

        thispath = os.path.dirname(os.path.abspath(__file__))
        self.dll_path = thispath + '/../extensions/psf_simu_gpu.dll'

    def simulate(self, x, y, z, photons, zernike_coefs=None):
        """
        Run the simulation to generate the vector psfs with the given positions

        Args:
            x (torch.Tensor): x positions of the psfs, unit nm
            y (torch.Tensor): y positions of the psfs, unit nm
            z (torch.Tensor): z positions of the psfs, unit nm
            photons (torch.Tensor): photon counts of the psfs, unit photons
            zernike_coefs (torch.Tensor or None): if not None, each psf can be assigned a different zernike
                coefficients from this array with shape (npsf, 21), otherwise use the common class
                property self.zernike_coef, unit nm

        Returns:
            torch.Tensor: psfs, unit photons
        """

        try:
            psf_dll = ctypes.CDLL(self.dll_path, winmode=0)
        except:
            thispath = os.path.dirname(os.path.abspath(__file__))
            self.dll_path = thispath + '/../extensions/psf_simu_gpu_v2.dll'
            psf_dll = ctypes.CDLL(self.dll_path, winmode=0)

        class _PSFParams(ctypes.Structure):
            _fields_ = [
                ('aberrations_', ctypes.POINTER(ctypes.c_float)),
                ('NA_', ctypes.c_float),
                ('refmed_', ctypes.c_float),
                ('refcov_', ctypes.c_float),
                ('refimm_', ctypes.c_float),
                ('lambdaX_', ctypes.c_float),
                ('objStage0_', ctypes.c_float),
                ('zemit0_', ctypes.c_float),
                ('pixelSizeX_', ctypes.c_float),
                ('pixelSizeY_', ctypes.c_float),
                ('sizeX_', ctypes.c_float),
                ('sizeY_', ctypes.c_float),
                ('PupilSize_', ctypes.c_float),
                ('Npupil_', ctypes.c_float),
                ('zernikeModesN_', ctypes.c_int),
                ('xemit_', ctypes.POINTER(ctypes.c_float)),
                ('yemit_', ctypes.POINTER(ctypes.c_float)),
                ('zemit_', ctypes.POINTER(ctypes.c_float)),
                ('objstage_', ctypes.POINTER(ctypes.c_float)),
                ('aberrationsParas_', ctypes.POINTER(ctypes.c_float)),
                ('psfOut_', ctypes.POINTER(ctypes.c_float)),
                ('aberrationOut_', ctypes.POINTER(ctypes.c_float)),
                ('Nmol_', ctypes.c_int),
                ('showAberrationNumber_', ctypes.c_int)
            ]

        param_struct = _PSFParams()

        # the x y positions and pixelsize should be inversed to ensure the 
        # input X_os corresponds to the column
        param_struct.Npupil_ = self.npupil
        param_struct.Nmol_ = x.shape[0]
        param_struct.NA_ = self.na
        param_struct.refmed_ = self.refmed
        param_struct.refcov_ = self.refcov
        param_struct.refimm_ = self.refimm
        param_struct.lambdaX_ = self.wavelength
        param_struct.objStage0_ = self.objstage0
        param_struct.zemit0_ = self.zemit0
        param_struct.pixelSizeX_ = self.pixel_size_xy[1]
        param_struct.pixelSizeY_ = self.pixel_size_xy[0]
        param_struct.zernikeModesN_ = self.zernike_mode.shape[0]
        param_struct.sizeX_ = self.psf_size
        param_struct.sizeY_ = self.psf_size
        param_struct.PupilSize_ = 1.0
        param_struct.showAberrationNumber_ = 1

        np_xemit = ailoc.common.cpu(y).astype('float32')  # nm
        np_yemit = ailoc.common.cpu(x).astype('float32')  # nm
        np_zemit = ailoc.common.cpu(z).astype('float32')  # nm
        np_objstage = 0 * np_zemit  # nm
        np_aberrations = np.array(np.pad(self.zernike_mode, pad_width=((0, 0), (0, 1)), mode='constant',
                                         constant_values=((0, 0), (0, 0))).flatten('F'), dtype=np.float32)

        param_struct.xemit_ = np_xemit.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        param_struct.yemit_ = np_yemit.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        param_struct.zemit_ = np_zemit.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        param_struct.objstage_ = np_objstage.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        param_struct.aberrations_ = np_aberrations.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        n_mol = param_struct.Nmol_
        outdata_buffer = np.empty((n_mol, self.psf_size, self.psf_size), dtype=np.float32, order='C')
        outpupil_buffer = np.empty((int(param_struct.Npupil_), int(param_struct.Npupil_)), dtype=np.float32, order='C')
        input_zernikecoefs_buffer = np.empty((n_mol, param_struct.zernikeModesN_), dtype=np.float32, order='C')

        if zernike_coefs is None:
            assert self.zernike_coef is not None, \
                'both self.zernike_coef and input zernike_coefs are None, please define either of them'
            zernike_coefs = np.tile(self.zernike_coef, reps=(n_mol, 1)).astype('float32')
        else:
            zernike_coefs = ailoc.common.cpu(zernike_coefs).astype('float32')

        for n in range(0, n_mol):
            input_zernikecoefs_buffer[n] = zernike_coefs[n]
        # input_zernikecoefs_buffer = input_zernikecoefs_buffer * self.wavelength  # multiply with lambda

        param_struct.aberrationsParas_ = input_zernikecoefs_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        param_struct.psfOut_ = outdata_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        param_struct.aberrationOut_ = outpupil_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # run simulation
        psf_dll.vectorPSFF1(param_struct)

        # check nan in the output PSFs
        assert not np.isnan(outdata_buffer).any(), "nan in the gpu psf, something wrong about the .dll call"

        psfs_out = ailoc.common.gpu(outdata_buffer)

        # otf rescale
        if self.otf_rescale_xy[0] or self.otf_rescale_xy[1] != 0:
            psfs_out = self.otf_rescale(psfdata=psfs_out, sigma_xy=ailoc.common.gpu(self.otf_rescale_xy))

        # normalize the psf to 1, then multiply with the photon number
        # psfs_out /= psfs_out.sum(-1).sum(-1)[:, None, None]
        psfs_out *= photons[:, None, None]

        return psfs_out


class PartialOptimizeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coef, indices_to_optimize):
        # save needed tensors and indices
        ctx.save_for_backward(coef, indices_to_optimize)
        return coef  # directly return coef in the forward pass

    @staticmethod
    def backward(ctx, grad_output):
        # get the saved tensors and indices
        coef, indices_to_optimize = ctx.saved_tensors
        # create a zero gradient tensor
        grad_input = torch.zeros_like(coef)
        # only assign gradients to specified indices
        grad_input[indices_to_optimize] = grad_output[indices_to_optimize]
        return grad_input, None


# 包装函数
def partial_optimize(coef, indices_to_optimize):
    return PartialOptimizeFunction.apply(coef, indices_to_optimize)


class VectorPSFTorch(VectorPSF):
    """
    Vector psf simulator using pytorch, thus psf parameters can be optimized by pytorch
    """

    def __init__(self, psf_params, req_grad=False, data_type=torch.float64, zernike_idx_learn=None, device='cuda'):
        """

        Args:
            psf_params (dict): PSF parameters for simulation except emitter positions
            req_grad (bool): whether the PSF parameters are required gradient
            data_type (torch.dtype): data type for the PSF parameters
            zernike_idx_learn (list of bool): specified zernike coefficients to learn for LUNAR SL.

        Returns:
            VectorPSFTorch: an instance of VectorPSFTorch
        """

        self.data_type = data_type
        if data_type == torch.float64:
            self.complex_type = torch.complex128
        elif data_type == torch.float32:
            self.complex_type = torch.complex64
        else:
            raise ValueError(f'unsupported data type {data_type}')
        self._setup_device(device)

        self.na = torch.tensor(psf_params['na'], dtype=self.data_type, device=self.device)
        self.wavelength = torch.tensor(psf_params['wavelength'], device=self.device, dtype=self.data_type)
        self.refmed = torch.tensor(psf_params['refmed'], device=self.device, dtype=self.data_type,)
        self.refcov = torch.tensor(psf_params['refcov'], device=self.device, dtype=self.data_type,)
        self.refimm = torch.tensor(psf_params['refimm'], device=self.device, dtype=self.data_type,)
        self.zernike_mode = torch.tensor(psf_params['zernike_mode'], device=self.device, dtype=self.data_type)
        self.zernike_coef = torch.tensor(psf_params['zernike_coef'], device=self.device, dtype=self.data_type,
                                         requires_grad=req_grad)
        if zernike_idx_learn is None:
            self.zernike_idx_learn = torch.arange(self.zernike_coef.shape[0])
        else:
            self.zernike_idx_learn = torch.tensor(zernike_idx_learn)
        self.zernike_coef_map = None
        self.objstage0 = torch.tensor(psf_params['objstage0'], device=self.device, dtype=self.data_type,)
        try:
            self.zemit0 = torch.tensor(psf_params['zemit0'], device=self.device, dtype=self.data_type, )
        except KeyError:
            self.zemit0 = torch.tensor(-psf_params['objstage0']/psf_params['refimm']*psf_params['refmed'],
                                       device=self.device,
                                       dtype=self.data_type,)

        self.pixel_size_xy = torch.tensor(psf_params['pixel_size_xy'], device=self.device, dtype=self.data_type)
        self.otf_rescale_xy = torch.tensor(psf_params['otf_rescale_xy'], device=self.device, dtype=self.data_type,)
        self.npupil = psf_params['npupil']
        self.psf_size = psf_params['psf_size']

        self.focus_norm = psf_params.get('focus_norm', False)

        self._pre_compute()

    def get_zernike(self, orders, xpupil, ypupil):
        """
        Calculate zernike polynomials on pupil plane

        Args:
            orders (torch.Tensor): zernike orders, 2D array, first column is radial order,
                second column is azimuthal order
            xpupil (torch.Tensor): x coordinate of pupil plane
            ypupil (torch.Tensor): y coordinate of pupil plane

        Returns:
            torch.Tensor: zernike polynomials
        """

        xpupil = torch.real(xpupil)
        ypupil = torch.real(ypupil)
        zersize = orders.shape
        Nzer = zersize[0]
        radormax = int(max(orders[:, 0]))
        azormax = int(max(abs(orders[:, 1])))
        [Nx, Ny] = xpupil.shape

        # zerpol = np.zeros( [radormax+1,azormax+1,Nx,Ny] )
        zerpol = torch.zeros([21, 6, Nx, Ny], device=self.device)
        rhosq = xpupil ** 2 + ypupil ** 2
        rho = torch.sqrt(rhosq)
        zerpol[0, 0, :, :] = torch.ones_like(xpupil)

        for jm in range(1, azormax + 2 + 1):
            m = jm - 1
            if m > 0:
                zerpol[jm - 1, jm - 1, :, :] = rho * torch.squeeze(zerpol[jm - 1 - 1, jm - 1 - 1, :, :])

            zerpol[jm + 2 - 1, jm - 1, :, :] = ((m + 2) * rhosq - m - 1) * torch.squeeze(zerpol[jm - 1, jm - 1, :, :])
            for p in range(2, radormax - m + 2 + 1):
                n = m + 2 * p
                jn = n + 1
                zerpol[jn - 1, jm - 1, :, :] = (2 * (n - 1) * (n * (n - 2) * (2 * rhosq - 1) - m ** 2) * torch.squeeze(
                    zerpol[jn - 2 - 1, jm - 1, :, :]) -
                                                n * (n + m - 2) * (n - m - 2) * torch.squeeze(
                            zerpol[jn - 4 - 1, jm - 1, :, :])) / ((n - 2) * (n + m) * (n - m))

        phi = torch.atan2(ypupil, xpupil)
        allzernikes = torch.zeros([Nzer, Nx, Ny], device=self.device)
        for j in range(1, Nzer + 1):
            n = int(orders[j - 1, 0])
            m = int(orders[j - 1, 1])
            if m >= 0:
                allzernikes[j - 1, :, :] = torch.squeeze(zerpol[n + 1 - 1, m + 1 - 1, :, :]) * torch.cos(m * phi)
            else:
                allzernikes[j - 1, :, :] = torch.squeeze(zerpol[n + 1 - 1, -m + 1 - 1, :, :]) * torch.sin(-m * phi)

        # plt.figure(constrained_layout=True)
        # for i in range(21):
        #     plt.subplot(3, 7, i + 1)
        #     plt.imshow(cpu(allzernikes[i]))
        # plt.show()

        return allzernikes

    def czt(self, datain, A, B, D):
        """
        Execute the chirp-z transform

        Args:
            datain (torch.Tensor): input data, 2D array
            A (torch.Tensor): chirp parameter, 1D array
            B (torch.Tensor): chirp parameter, 1D array
            D (torch.Tensor): chirp parameter, 1D array

        Returns:
            torch.Tensor: output data, 2D array
        """

        N = A.shape[1]
        M = B.shape[1]
        L = D.shape[1]
        K = datain.shape[0]

        # torch.repeat_interleave is too slow
        # t0 = time.time()
        # Amt = torch.repeat_interleave(A, K, 0)
        # Bmt = torch.repeat_interleave(B, K, 0)
        # Dmt = torch.repeat_interleave(D, K, 0)
        # print('torch czt: ', time.time() - t0)
        Amt = A.expand(K, N)
        Bmt = B.expand(K, M)
        Dmt = D.expand(K, L)

        cztin = torch.zeros([K, L], dtype=self.complex_type, device=self.device)
        cztin[:, 0:N] = Amt * datain
        tmp = Dmt * torch.fft.fft(cztin)
        cztout = torch.fft.ifft(tmp)

        dataout = Bmt * cztout[:, 0:M]

        return dataout

    def czt_parallel(self, datain, A, B, D):
        """
        Execute the chirp-z transform

        Args:
            datain (torch.Tensor): input data, 2D array
            A (torch.Tensor): chirp parameter, 1D array
            B (torch.Tensor): chirp parameter, 1D array
            D (torch.Tensor): chirp parameter, 1D array

        Returns:
            torch.Tensor: output data, 2D array
        """

        N = A.shape[1]
        M = B.shape[1]
        L = D.shape[1]
        K = datain.shape[-2]
        n_mol = datain.shape[-3]

        # torch.repeat_interleave is too slow
        # t0 = time.time()
        # Amt = torch.repeat_interleave(A, K, 0)
        # Bmt = torch.repeat_interleave(B, K, 0)
        # Dmt = torch.repeat_interleave(D, K, 0)
        # print('torch czt: ', time.time() - t0)
        Amt = A.expand(K, N)
        Bmt = B.expand(K, M)
        Dmt = D.expand(K, L)

        cztin = torch.zeros([2, 3, n_mol, K, L], dtype=self.complex_type, device=self.device)
        cztin[:, :, :, :, 0:N] = Amt[None, None, None] * datain
        try:
            tmp = Dmt * torch.fft.fft(cztin, dim=-1)
        except:
            print('fft error')
        cztout = torch.fft.ifft(tmp, dim=-1)

        dataout = Bmt[None, None, None] * cztout[:, :, :, :, 0:M]

        return dataout

    def prechirpz(self, xsize, qsize, N, M):
        """
        Calculate the auxiliary vectors for chirp-z.

        Args:
            xsize (float): normalized pupil radius 1.0
            qsize (float): the original pixel number that could cover the region of interest
                in the image plane if using FFT
            N (int): the sampling number on the pupil
            M (int): the sampling number on the region of interest on the image plane

        Returns:
            (torch.Tensor,torch.Tensor,torch.Tensor): auxiliary vectors
        """

        L = N + M - 1
        # STABILITY FIX: Use high-precision constants for intermediate calculations
        sigma = (2 * np.pi * xsize * qsize / N / M).to(self.complex_type)

        # fixed phase factor and amplitude factor
        Gfac = (2 * xsize / N) * torch.exp(1j * sigma * (1 - N) * (1 - M))

        # STABILITY FIX: Replace unstable cumulative product loops with
        # direct, vectorized exponentiation.

        # integration about n
        n_vec_N = torch.arange(N, device=self.device, dtype=self.data_type).unsqueeze(0)
        A = torch.exp(2j * sigma * (n_vec_N ** 2 + n_vec_N * (1 - M)))

        #  the factor before the summation
        n_vec_B = torch.arange(M, device=self.device, dtype=self.data_type).unsqueeze(0)
        B = Gfac * torch.exp(2j * sigma * (n_vec_B ** 2 + n_vec_B * (1 - N)))

        # for circular convolution
        n_vec_V = torch.arange(max(N, M) + 1, device=self.device, dtype=self.data_type).unsqueeze(0)
        Vtmp = torch.exp(2j * sigma * n_vec_V ** 2)

        D = torch.zeros([1, L], dtype=self.complex_type, device=self.device)
        # Fill D using Vtmp, ensuring boundary conditions are zero
        D[0, 0:M] = torch.conj(Vtmp[0, 0:M])

        # Original: D[0, L - 1 - i] = torch.conj(Vtmp[0, i + 1]) for i in 0..N-1
        # i=0 -> D[0, L-1] = conj(Vtmp[0, 1])
        # i=N-1 -> D[0, L-N] = conj(Vtmp[0, N])
        # This corresponds to a reversed slice from Vtmp[1] to Vtmp[N]
        D[0, L - N:L] = torch.conj(torch.flip(Vtmp[0, 1:N + 1], dims=[0]))

        D = torch.fft.fft(D, axis=1)

        return A, B, D

    def _pre_compute(self):
        """
        Compute the common intermediate variables in advance, this can save time for PSFs simulation
        """
        # pupil radius (in diffraction units) and pupil coordinate sampling
        pupil_size = 1.0
        dxypupil = 2 * pupil_size / self.npupil
        xypupil = torch.arange(-pupil_size + dxypupil / 2, pupil_size, dxypupil, device=self.device, dtype=self.data_type)
        [xpupil_f, ypupil_f] = torch.meshgrid(xypupil, xypupil, indexing='ij')

        # STABILITY FIX: Use complex128 for intermediate pupil coordinates
        # to prevent precision loss in sqrt(1-x^2)
        ypupil = torch.complex(ypupil_f, torch.zeros_like(ypupil_f)).to(torch.complex128)
        xpupil = torch.complex(xpupil_f, torch.zeros_like(xpupil_f)).to(torch.complex128)

        # STABILITY FIX: Use float64/complex128 for all intermediate
        # calculations involving costheta and Fresnel coefficients.
        rhosq_f64 = xpupil ** 2 + ypupil ** 2
        na_f64 = self.na.to(torch.float64)
        refmed_f64 = self.refmed.to(torch.float64)
        refcov_f64 = self.refcov.to(torch.float64)
        refimm_f64 = self.refimm.to(torch.float64)

        # calculation of relevant Fresnel-coefficients for the interfaces
        costhetamed_f64 = torch.sqrt(1.0 - rhosq_f64 * (na_f64 ** 2) / (refmed_f64 ** 2))
        costhetacov_f64 = torch.sqrt(1.0 - rhosq_f64 * (na_f64 ** 2) / (refcov_f64 ** 2))
        costhetaimm_f64 = torch.sqrt(1.0 - rhosq_f64 * (na_f64 ** 2) / (refimm_f64 ** 2))

        fresnelpmedcov = 2 * refmed_f64 * costhetamed_f64 / (
                    refmed_f64 * costhetacov_f64 + refcov_f64 * costhetamed_f64)
        fresnelsmedcov = 2 * refmed_f64 * costhetamed_f64 / (
                    refmed_f64 * costhetamed_f64 + refcov_f64 * costhetacov_f64)
        fresnelpcovimm = 2 * refcov_f64 * costhetacov_f64 / (
                    refcov_f64 * costhetaimm_f64 + refimm_f64 * costhetacov_f64)
        fresnelscovimm = 2 * refcov_f64 * costhetacov_f64 / (
                    refcov_f64 * costhetacov_f64 + refimm_f64 * costhetaimm_f64)
        fresnelp_f64 = fresnelpmedcov * fresnelpcovimm
        fresnels_f64 = fresnelsmedcov * fresnelscovimm

        # apodization
        apod_f64 = torch.sqrt(costhetaimm_f64) / costhetamed_f64  # Sjoerd Stallinga version
        # apod_f64 = 1 / torch.sqrt(costhetaimm_f64)  # previous version, for the simulated test dataset, should be deprecated

        # define aperture
        aperturemask_f = torch.where(rhosq_f64.real < 1.0, 1.0, 0.0).to(self.data_type)

        # STABILITY FIX: Cast back to the required precision *after* sensitive calculations
        self.amplitude = (aperturemask_f * apod_f64).to(self.complex_type)
        fresnelp = fresnelp_f64.to(self.complex_type)
        fresnels = fresnels_f64.to(self.complex_type)
        costhetamed = costhetamed_f64.to(self.complex_type)
        costhetaimm = costhetaimm_f64.to(self.complex_type)
        xpupil = xpupil.to(self.complex_type)
        ypupil = ypupil.to(self.complex_type)

        # setting of vectorial functions
        phi = torch.atan2(torch.real(ypupil), torch.real(xpupil))
        cosphi = torch.cos(phi)
        sinphi = torch.sin(phi)
        costheta = costhetamed
        sintheta = torch.sqrt(1 - costheta ** 2)

        pvec = torch.empty([3, self.npupil, self.npupil], dtype=self.complex_type, device=self.device)
        pvec[0] = fresnelp * costheta * cosphi
        pvec[1] = fresnelp * costheta * sinphi
        pvec[2] = -fresnelp * sintheta
        svec = torch.empty([3, self.npupil, self.npupil], dtype=self.complex_type, device=self.device)
        svec[0] = -fresnels * sinphi
        svec[1] = fresnels * cosphi
        svec[2] = 0 * cosphi

        self.polarizationvector = torch.empty([2, 3, self.npupil, self.npupil], dtype=self.complex_type, device=self.device)
        self.polarizationvector[0,] = cosphi * pvec - sinphi * svec
        self.polarizationvector[1,] = sinphi * pvec + cosphi * svec

        self.wavevector = torch.empty([2, self.npupil, self.npupil], dtype=self.complex_type, device=self.device)
        self.wavevector[0] = 2 * np.pi * self.na / self.wavelength * xpupil
        self.wavevector[1] = 2 * np.pi * self.na / self.wavelength * ypupil
        self.wavevectorzimm = 2 * np.pi * self.refimm / self.wavelength * costhetaimm
        self.wavevectorzmed = 2 * np.pi * self.refmed / self.wavelength * costhetamed

        # calculate aberration function
        normfac = torch.sqrt(
            2 * (self.zernike_mode[:, 0] + 1) / (1 + torch.where(self.zernike_mode[:, 1] == 0, 1.0, 0.0)))
        # STABILITY FIX: Use the float version of aperturemask
        self.allzernikes = self.get_zernike(self.zernike_mode, xpupil, ypupil) * normfac[:, None, None] * \
                           aperturemask_f[None]
        # Cast allzernikes to complex_type for phase calculations
        self.allzernikes = self.allzernikes.to(self.complex_type)

        # czt transform(fft the pupil)
        xrange = self.pixel_size_xy[0] * self.psf_size / 2
        yrange = self.pixel_size_xy[1] * self.psf_size / 2
        imagesizex = xrange * self.na / self.wavelength
        imagesizey = yrange * self.na / self.wavelength

        # calculate the auxiliary vectors for chirp-z, pixelsize_xy should be inverse to match row and column
        self.ax, self.bx, self.dx = self.prechirpz(pupil_size, imagesizey, self.npupil, self.psf_size)
        self.ay, self.by, self.dy = self.prechirpz(pupil_size, imagesizex, self.npupil, self.psf_size)

        # calculate intensity normalization function using the PSF at focus
        pupilfunction_norm = self.amplitude[None, None, None] * self.polarizationvector[:, :, None]
        inter_image_norm = torch.transpose(self.czt_parallel(pupilfunction_norm, self.ax, self.bx, self.dx), -1, -2)
        fieldmatrix_norm = torch.transpose(self.czt_parallel(inter_image_norm, self.ay, self.by, self.dy), -1, -2)

        int_focus = torch.zeros([self.psf_size, self.psf_size], dtype=self.data_type, device=self.device)
        int_focus += 1 / 3 * torch.sum(torch.abs(fieldmatrix_norm) ** 2, dim=(0, 1, 2))
        self.norm_intensity = torch.sum(int_focus)

    def simulate(self, x, y, z, photons, objstage=None, zernike_coefs=None):
        """
        Run the simulation to generate the vector PSFs with the given positions

        Args:
            x (torch.Tensor): x positions of the PSFs, unit nm
            y (torch.Tensor): y positions of the PSFs, unit nm
            z (torch.Tensor): z positions of the PSFs, unit nm
            photons (torch.Tensor): photon counts of the PSFs, unit photons
            objstage (torch.Tensor): objective stage positions relative to the cover-slip (0),
                the closer to the sample, the smaller this value is (-), unit nm
            zernike_coefs (torch.Tensor or None): if not None, each psf can be assigned a different zernike
                coefficients from this array with shape (npsf, 21), otherwise use the common class
                property self.zernike_coef, unit nm

        Returns:
            torch.Tensor: PSFs, unit photons
        """

        n_mol = x.shape[0]
        if n_mol == 0:
            return torch.zeros([0, self.psf_size, self.psf_size], device=self.device, dtype=torch.float32)

        objstage = torch.zeros(x.shape[0], dtype=self.data_type, device=self.device) if objstage is None else objstage

        # # temporally test, change the z to obj
        # objstage = z
        # z = torch.zeros_like(z, dtype=self.data_type, device='cuda')


        if zernike_coefs is None:
            try:
                # for partially optimized zernike coefficients in LUNAR SL physics learning
                zernike_coef_part_optm = partial_optimize(self.zernike_coef, self.zernike_idx_learn)
                zernike_phase = torch.exp(1j * 2 * np.pi *
                                          torch.sum(zernike_coef_part_optm[:, None, None] * self.allzernikes, dim=0)
                                          / self.wavelength)
            except:
                zernike_phase = torch.exp(1j * 2 * np.pi *
                                          torch.sum(self.zernike_coef[:, None, None] * self.allzernikes, dim=0)
                                          / self.wavelength)
            pupilmatrix = (self.amplitude[None, None] *
                           zernike_phase[None, None] *
                           self.polarizationvector)[:, :, None]

        # batch_size in parallel to save GPU memory
        slice_list = []
        batch_size = 100
        for i in np.arange(0, n_mol, batch_size):
            slice_list.append(slice(i, min(i + batch_size, n_mol)))

        psfs_out = torch.zeros([n_mol, self.psf_size, self.psf_size], device=self.device, dtype=torch.float32)
        for slice_tmp in slice_list:
            if zernike_coefs is not None:
                zernike_phase = torch.exp(1j * 2 * np.pi *
                                          torch.sum(zernike_coefs[slice_tmp, :, None, None] * self.allzernikes[None], dim=1)
                                          / self.wavelength)
                pupilmatrix = (zernike_phase[None, None] *
                               self.polarizationvector[:, :, None] *
                               self.amplitude[None, None, None])

            # length_tmp = slice_tmp.stop - slice_tmp.start
            # position_phase = torch.empty([length_tmp, self.npupil, self.npupil], dtype=self.complex_type, device=self.device)
            #
            # idx = torch.where(z[slice_tmp] + self.zemit0 >= 0)[0]
            # phase_xyz_tmp = -y[slice_tmp][idx][:, None, None] * self.wavevector[0][None] - x[slice_tmp][idx][:, None, None] * \
            #                 self.wavevector[1][None] + \
            #                 (z[slice_tmp][idx] + self.zemit0)[:, None, None] * self.wavevectorzmed[None]
            # position_phase[idx, :, :] = torch.exp(
            #     1j * (phase_xyz_tmp + (objstage[slice_tmp][idx][:, None, None] + self.objstage0) *
            #           self.wavevectorzimm[None]))
            #
            # idx = torch.where(z[slice_tmp] + self.zemit0 < 0)[0]
            # phase_xyz_tmp = -y[slice_tmp][idx][:, None, None] * self.wavevector[0][None] - x[slice_tmp][idx][:, None, None] * \
            #                 self.wavevector[1][None]
            # position_phase[idx, :, :] = torch.exp(
            #     1j * (phase_xyz_tmp + (objstage[slice_tmp][idx][:, None, None] + self.objstage0 + z[slice_tmp][idx][:, None, None]
            #                            + self.zemit0) * self.wavevectorzimm[None]))

            phase_xyz_tmp = -y[slice_tmp][:, None, None] * self.wavevector[0][None] - x[slice_tmp][:, None, None] * \
                            self.wavevector[1][None] + (z[slice_tmp] + self.zemit0)[:, None, None] * self.wavevectorzmed[None]
            position_phase = torch.exp(1j * (phase_xyz_tmp + (objstage[slice_tmp][:, None, None] + self.objstage0) *
                             self.wavevectorzimm[None]))

            pupil_tmp = position_phase[None, None] * pupilmatrix
            inter_image = torch.transpose(self.czt_parallel(pupil_tmp, self.ay, self.by, self.dy), -1, -2)
            field_matrix = torch.transpose(self.czt_parallel(inter_image, self.ax, self.bx, self.dx), -1, -2)

            psfs_out[slice_tmp] += 1 / 3 * torch.sum((torch.abs(field_matrix[:, :])) ** 2, dim=(0, 1))

        if self.focus_norm:
            # intensity normalization by focus
            psfs_out /= self.norm_intensity.clamp_min(1e-12)
        else:
            # normalize by themselves
            norm_factor = psfs_out.sum(dim=(-1, -2)).clamp_min(1e-12)
            psfs_out /= norm_factor[:, None, None]

        # otf rescale
        if torch.any(self.otf_rescale_xy != 0):
            psfs_out = self.otf_rescale(psfdata=psfs_out, sigma_xy=self.otf_rescale_xy)

        # multiply with the photon number
        psfs_out *= photons[:, None, None]
        if torch.isnan(psfs_out).any():
            raise ValueError('nan in the psf output, something wrong in the simulation!')

        return psfs_out

    def compute_crlb(self, x, y, z, photons, bgs):
        """
        Calculate the CRLB of this PSF model at give positions, photons and backgrounds.

        Args:
            x (torch.Tensor): x positions of the PSFs, unit nm
            y (torch.Tensor): y positions of the PSFs, unit nm
            z (torch.Tensor): z positions of the PSFs, unit nm
            photons (torch.Tensor): photon counts of the PSFs, unit photons
            bgs (torch.Tensor): background counts of the PSFs, unit photons

        Returns:
            (torch.Tensor,torch.Tensor): CRLB xyz (nPSFs, 3) and model PSFs, unit nm and photons
        """

        n_mol = x.shape[0]

        # calculate the derivatives
        [dudt, model] = self._compute_derivative(x, y, z, photons, bgs)
        # dudt = self._torch_jacobian_derivative(x, y, z, photons, bgs)

        # calculate hessian matrix, here only consider not shared parameters: x, y, z, photons, background
        num_pars = n_mol * 5
        t2 = 1 / model
        hessian = torch.zeros([num_pars, num_pars], device=self.device, dtype=self.data_type)
        for p1 in range(num_pars):
            temp1_zind = int(np.floor(p1 / 5))  # the index of data
            temp1_pind = int(p1 % 5)  # the index of parameter type
            temp1 = dudt[:, :, :, temp1_pind]  # the derivative of data concerning this parameter type
            for p2 in range(p1, num_pars):
                temp2_zind = int(np.floor(p2 / 5))
                temp2_pind = int(p2 % 5)
                temp2 = dudt[:, :, :, temp2_pind]

                # since all parameters are not shared, only the same molecule data makes sense
                # when multiply gradients of two parameters
                if temp1_zind == temp2_zind:
                    temp = t2[temp1_zind, :, :] * temp1[temp1_zind, :, :] * temp2[temp2_zind, :, :]
                    hessian[p1, p2] = torch.sum(temp)
                    hessian[p2, p1] = hessian[p1, p2]

        # calculate local fisher matrix and crlb
        xyz_crlb = torch.zeros([n_mol, 3], device=self.device)
        for j in range(n_mol):
            fisher_tmp = hessian[j * 5:j * 5 + 5, j * 5:j * 5 + 5]
            sqrt_crlb_tmp = torch.sqrt(torch.diag(torch.inverse(fisher_tmp)))
            xyz_crlb[j] = sqrt_crlb_tmp[0:3]

        return xyz_crlb, model

    def _compute_derivative(self, x, y, z, photons, bgs):
        """
        Calculate the analytical derivatives of the PSFs at given parameters with respect to x,y,z,photons,bg
        """

        n_mol = x.shape[0]
        objstage = torch.zeros(x.shape[0], dtype=self.data_type, device=self.device)

        zernike_phase = torch.exp(1j * 2 * np.pi *
                                  torch.sum(self.zernike_coef[:, None, None]*self.allzernikes, dim=0)
                                  / self.wavelength)
        pupilmatrix = (self.amplitude[None, None] *
                       zernike_phase[None, None] *
                       self.polarizationvector)[:, :, None]

        # batch_size in parallel to save GPU memory
        slice_list = []
        batch_size = 100
        for i in np.arange(0, n_mol, batch_size):
            slice_list.append(slice(i, min(i + batch_size, n_mol)))

        psfs = torch.zeros([n_mol, self.psf_size, self.psf_size], device=self.device, dtype=self.data_type)
        psfs_ders = torch.zeros([n_mol, self.psf_size, self.psf_size, 3], device=self.device, dtype=self.data_type)
        for slice_tmp in slice_list:
            # length_tmp = slice_tmp.stop - slice_tmp.start
            # position_phase = torch.empty([length_tmp, self.npupil, self.npupil], dtype=self.complex_type, device='cuda')

            # idx_0 = torch.where(z[slice_tmp] + self.zemit0 >= 0)[0]
            # phase_xyz_tmp = -y[slice_tmp][idx_0][:, None, None] * self.wavevector[0][None] - x[slice_tmp][idx_0][:, None, None] * \
            #                 self.wavevector[1][None] + \
            #                 (z[slice_tmp][idx_0] + self.zemit0)[:, None, None] * self.wavevectorzmed[None]
            # position_phase[idx_0, :, :] = torch.exp(
            #     1j * (phase_xyz_tmp + (objstage[slice_tmp][idx_0][:, None, None] + self.objstage0) *
            #           self.wavevectorzimm[None]))
            #
            # idx_1 = torch.where(z[slice_tmp] + self.zemit0 < 0)[0]
            # phase_xyz_tmp = -y[slice_tmp][idx_1][:, None, None] * self.wavevector[0][None] - x[slice_tmp][idx_1][:, None, None] * \
            #                 self.wavevector[1][None]
            # position_phase[idx_1, :, :] = torch.exp(
            #     1j * (phase_xyz_tmp + (objstage[slice_tmp][idx_1][:, None, None] + self.objstage0 + z[slice_tmp][idx_1][:, None, None]
            #                            + self.zemit0) * self.wavevectorzimm[None]))

            phase_xyz_tmp = -y[slice_tmp][:, None, None] * self.wavevector[0][None] - x[slice_tmp][:, None, None] * \
                            self.wavevector[1][None] + (z[slice_tmp] + self.zemit0)[:, None, None] * self.wavevectorzmed[None]
            position_phase = torch.exp(1j * (phase_xyz_tmp + (objstage[slice_tmp][:, None, None] + self.objstage0) *
                             self.wavevectorzimm[None]))

            pupil_tmp = position_phase[None, None] * pupilmatrix
            inter_image = torch.transpose(self.czt_parallel(pupil_tmp, self.ay, self.by, self.dy), -1, -2)
            field_matrix = torch.transpose(self.czt_parallel(inter_image, self.ax, self.bx, self.dx), -1, -2)
            psfs[slice_tmp] += 1 / 3 * torch.sum((torch.abs(field_matrix[:, :])) ** 2, dim=(0, 1))

            # derivatives with respect to x,y,z
            pupil_tmp_x = -1j * self.wavevector[1] * position_phase[None, None] * pupilmatrix
            inter_image_x = torch.transpose(self.czt_parallel(pupil_tmp_x, self.ay, self.by, self.dy), -1, -2)
            field_matrix_x = torch.transpose(self.czt_parallel(inter_image_x, self.ax, self.bx, self.dx), -1, -2)
            psfs_ders[slice_tmp, :, :, 0] += 2 / 3 * torch.sum(torch.real(torch.conj(field_matrix) * field_matrix_x),
                                                               dim=(0, 1))

            pupil_tmp_y = -1j * self.wavevector[0] * position_phase[None, None] * pupilmatrix
            inter_image_y = torch.transpose(self.czt_parallel(pupil_tmp_y, self.ay, self.by, self.dy), -1, -2)
            field_matrix_y = torch.transpose(self.czt_parallel(inter_image_y, self.ax, self.bx, self.dx), -1, -2)
            psfs_ders[slice_tmp, :, :, 1] += 2 / 3 * torch.sum(torch.real(torch.conj(field_matrix) * field_matrix_y),
                                                               dim=(0, 1))

            # pupil_tmp_z = torch.empty([2, 3, length_tmp, self.npupil, self.npupil], dtype=self.complex_type, device='cuda')
            # pupil_tmp_z[:, :, idx_0] = 1j * self.wavevectorzmed * position_phase[idx_0] * pupilmatrix
            # pupil_tmp_z[:, :, idx_1] = 1j * self.wavevectorzimm * position_phase[idx_1] * pupilmatrix
            pupil_tmp_z = 1j * self.wavevectorzmed * position_phase * pupilmatrix
            inter_image_z = torch.transpose(self.czt_parallel(pupil_tmp_z, self.ay, self.by, self.dy), -1, -2)
            field_matrix_z = torch.transpose(self.czt_parallel(inter_image_z, self.ax, self.bx, self.dx), -1, -2)
            psfs_ders[slice_tmp, :, :, 2] += 2 / 3 * torch.sum(torch.real(torch.conj(field_matrix) * field_matrix_z),
                                                               dim=(0, 1))
        if self.focus_norm:
            # normalize by focus intensity
            psfs /= self.norm_intensity
            psfs_ders /= self.norm_intensity
        else:
            # normalize by themselves
            norm_factor = psfs.sum(dim=(-1, -2))
            psfs /= norm_factor[:, None, None]
            psfs_ders /= norm_factor[:, None, None, None]

        # otf rescale
        if self.otf_rescale_xy[0] or self.otf_rescale_xy[1]:
            psfs = self.otf_rescale(psfdata=psfs, sigma_xy=self.otf_rescale_xy)
            psfs_ders[:, :, :, 0] = self.otf_rescale(psfdata=psfs_ders[:, :, :, 0], sigma_xy=self.otf_rescale_xy)
            psfs_ders[:, :, :, 1] = self.otf_rescale(psfdata=psfs_ders[:, :, :, 1], sigma_xy=self.otf_rescale_xy)
            psfs_ders[:, :, :, 2] = self.otf_rescale(psfdata=psfs_ders[:, :, :, 2], sigma_xy=self.otf_rescale_xy)

        psfs_out = ailoc.common.gpu(psfs * photons[:, None, None] + bgs[:, None, None])
        ders_out = torch.zeros([n_mol, self.psf_size, self.psf_size, 5], device=self.device, dtype=self.data_type)
        ders_out[:, :, :, 0:3] = psfs_ders * photons[:, None, None, None]
        ders_out[:, :, :, 3] = psfs
        ders_out[:, :, :, 4] = torch.ones_like(psfs)

        return ders_out, psfs_out

    def _torch_jacobian_derivative(self, x, y, z, photons, bgs):
        """Calculate the derivatives of the PSFs at given parameters with respect to x,y,z,photons,bg
        using torch.autograd.functional.jacobian"""
        x.requires_grad = True
        y.requires_grad = True
        z.requires_grad = True
        photons.requires_grad = True
        bgs.requires_grad = True
        jacobian = torch.autograd.functional.jacobian(self.simulate, (x, y, z, photons))

        ders_out = torch.zeros([x.shape[0], self.psf_size, self.psf_size, 5], device=self.device, dtype=self.data_type)
        for i in range(len(jacobian)):
            for j in range(x.shape[0]):
                ders_out[j, :, :, i] = jacobian[i][j, :, :, j]
        ders_out[:, :, :, 4] = torch.ones([x.shape[0], self.psf_size, self.psf_size])

        return ders_out

    def optimize_crlb(self, x, y, z, photons, bgs, tolerance):
        """
        Optimize the CRLB of this PSF model with respect to zernike coefficients
        at give positions, photons and backgrounds. The instance should be initialized with req_grad=True

        Args:
            x (torch.Tensor): x positions of the PSFs, unit nm
            y (torch.Tensor): y positions of the PSFs, unit nm
            z (torch.Tensor): z positions of the PSFs, unit nm
            photons (torch.Tensor): photon counts of the PSFs, unit photon
            bgs (torch.Tensor): background counts of the PSFs, unit photon
            tolerance (float): stop criteria, the difference of CRLB between two iterations

        Returns:

        """

        # # ADAM optimizer
        # n_mol = x.shape[0]
        # crlb_optimizer = torch.optim.Adam([self.zernike_coef], lr=0.1)
        # loss_1 = 1e6
        # iter_num = 1
        # # for iter_num in range(iterations):
        # while True:
        #     xyz_crlb, model = self.compute_crlb(x, y, z, photons, bgs)
        #     crlb_3d_avg = torch.sum(xyz_crlb[:, 0]**2 + xyz_crlb[:, 1]**2 + xyz_crlb[:, 2]**2)/n_mol
        #     print(f"iter: {iter_num}, crlb_3d_avg: {crlb_3d_avg}")
        #     crlb_optimizer.zero_grad()
        #     crlb_3d_avg.backward()
        #     crlb_optimizer.step()
        #     self._pre_compute()
        #     if torch.abs((loss_1 - crlb_3d_avg) / loss_1) < tolerance:
        #         break
        #     loss_1 = crlb_3d_avg.detach()
        #     iter_num += 1
        # print('CRLB optimization done')

        # LBFGS optimizer
        n_mol = x.shape[0]
        crlb_optimizer = torch.optim.LBFGS([self.zernike_coef], lr=0.1, max_iter=20, tolerance_grad=1e-7,
                                           tolerance_change=1e-9)
        loss_1 = 1e6
        iter_num = 1

        def closure():
            crlb_optimizer.zero_grad()
            xyz_crlb, model = self.compute_crlb(x, y, z, photons, bgs)
            crlb_3d_avg = torch.sum(xyz_crlb[:, 0] ** 2 + xyz_crlb[:, 1] ** 2 + xyz_crlb[:, 2] ** 2) / n_mol
            crlb_3d_avg.backward()
            return crlb_3d_avg

        while True:
            crlb_3d_avg = crlb_optimizer.step(closure)
            print(f"iter: {iter_num}, crlb_3d_avg: {crlb_3d_avg}")
            self._pre_compute()
            if torch.abs((loss_1 - crlb_3d_avg) / loss_1) < tolerance:
                break
            loss_1 = crlb_3d_avg.detach()
            iter_num += 1

        print('CRLB optimization done')