#! /usr/bin/env python

# Copyright (C) 2015 ETH Zurich, Institute for Astronomy

# System imports
from __future__ import print_function, division, absolute_import, unicode_literals


# External modules
import numpy as np

# lensDES imports
from easylens.DeLens.de_lens import DeLens
from easylens.DeLens.de_lens_multi_band import DeLensMultiBand
from easylens.DeLens.lens_model import LensModel
import easylens.util as util


class EasyLens(object):
    """
    class to model multi-band SED lenses
    """
    def __init__(self, lens_system, frame_bool=None, subgrid_res=2, lens_type='SIS'):
        self.system = lens_system
        frame_all_list = ["g_band", "r_band", "i_band", "Y_band", "z_band"]
        if frame_bool is None:
            frame_bool = {"g_band": True, "r_band": True, "i_band": True,
            "Y_band": True, "z_band": True}
        self.frame_list = []
        for frame in frame_all_list:
            if frame_bool[frame] is True and frame in self.system.available_frames:
                self.frame_list.append(frame)
        self.modeled_sources = []
        self.name_list = []
        self.subgrid_res = subgrid_res

        self.deLens = DeLens()
        self.deLensMultiBand = DeLensMultiBand()
        self.mask_frames = self.system.get_mask(kwargs_mask=None)
        self._kwargs_lens = None
        self.lens_type = lens_type

    def add_source(self, source, over_write=False):
        """
        add a new source class to the source list
        """
        source.update_lens_type(self.lens_type)
        name = source.name
        if name in self.name_list:
            if over_write:
                self.del_source(name)
            else:
                print("Source with name %s is already in the source list."
                             "Change source name or delete the old source" % name)
                return
        self.modeled_sources.append(source)
        self.name_list.append(name)
        print("source with name %s added" % name)

    def del_source(self, name):
        """
        delete a source class from the source list
        """
        for i in range(len(self.name_list)):
            if name == self.name_list[i]:
                del self.name_list[i]
                del self.modeled_sources[i]
                print("source with name %s deleted" % name)
                return
        print("source with name %s not found" % name)

    def del_sources_all(self):
        try:
            n = len(self.name_list)
        except:
            return
        for i in range(n):
            print("source with name %s deleted" % self.name_list[i])
        self.modeled_sources = []
        self.name_list = []

    def add_lens(self, kwargs_lens, print_statement=True):
        self._kwargs_lens = kwargs_lens
        if print_statement is True:
            print("lens model parameters added")

    def del_lens(self):
        self._kwargs_lens = None

    def get_lens(self):
        return self._kwargs_lens

    def update_mask(self, kwargs_mask):
        """

        :param kwargs_mask: keyword arguments for mask
        :return:
        """
        if type(kwargs_mask)==dict:
            self.mask_frames, self._num_pix_effective = self.system.get_mask(kwargs_mask)
        else:
            self.mask_frames, self._num_pix_effective = self.system.get_mask(kwargs_mask[0])
            for i in range(len(kwargs_mask)-1):
                self.mask_frames, self._num_pix_effective = self.system.add_mask(kwargs_mask[i+1],self.mask_frames,
                                                                                 self._num_pix_effective)

    def get_pixels_unmasked(self):
        """

        :return: number of pixels which are not masked out of all bands
        """
        return self._num_pix_effective

    def get_data_vector(self):
        """
        returns the data vector of multiple images
        """
        d_list = []
        for frame in self.frame_list:
            d = util.image2array(self.system.get_image(frame)) * self.mask_frames[frame]
            d_list.append(d)
        self._data_list = d_list
        d_final = np.concatenate(d_list, axis=0)
        return d_final

    def get_C_D_inv_vector(self):
        """
        returns the inverse covariance matrix (only diagonals) of multiple images
        """
        C_D_inv_list = []
        for frame in self.frame_list:
            C_D_inv = self.get_C_D_inv_frame(frame)
            C_D_inv_list.append(C_D_inv)
        self._C_D_inv_list = C_D_inv_list
        C_D_inv_final = np.concatenate(C_D_inv_list, axis=0)
        return C_D_inv_final

    def get_C_D_inv_frame(self, frame):
        """
        returns the inverse covariance matrix (only diagonals) of one frame
        """
        d = self.system.get_image(frame)
        sigma_bkg = self.system.get_sigma_bkg(frame)
        weight_map = self.system.get_weight_map(frame)
        mask = self.mask_frames[frame]
        M_cov_z = self.deLens.get_covariance_matrix(d, sigma_bkg, weight_map)
        C_D_inv = 1./util.image2array(M_cov_z) * mask
        return C_D_inv

    def get_response(self):
        """
        computes the list of response matrices for each band
        """
        A_list = []
        for frame in self.frame_list:
            A = self.get_response_frame(frame)
            A_list.append(A)
        A_final = np.concatenate(A_list, axis=1)
        return A_final

    def get_response_frame(self, frame):
        """
        get the response matrix for one frame
        """
        A_f_list = []
        ra_grid, dec_grid = self.system.get_coords(frame)
        numPix = self.system.get_numPix(frame)
        deltaPix = self.system.get_deltaPix(frame)
        psf_kwargs = self.system.get_psf_kwargs(frame)
        mask = self.mask_frames[frame]
        for source in self.modeled_sources:
            w_sed = source.w_SED[frame]
            A_f_i = source.get_response_matrix(ra_grid, dec_grid, psf_kwargs, numPix, deltaPix, self.subgrid_res, mask=mask, kwargs_lens=self._kwargs_lens)
            A_f_list.append(A_f_i*w_sed)
        A_f = np.concatenate(A_f_list, axis=0)
        return A_f

    def get_response_sed(self, param_array):
        """
        get the response matrix for all frames
        """
        A_list = []
        i = 0
        for frame in self.frame_list:
            A = self.get_response_sed_frame(param_array, frame, i)
            A_list.append(A)
            i += 1
        A_final = np.concatenate(A_list, axis=1)
        return A_final

    def get_response_sed_frame(self, param_array, frame, i):
        """
        get the response matrix for one frame for the SED fitting
        """

        num_frames = len(self.frame_list)
        num_sources = len(self.modeled_sources)
        ra_grid, dec_grid = self.system.get_coords(frame)
        numPix = self.system.get_numPix(frame)
        deltaPix = self.system.get_deltaPix(frame)
        psf_kwargs = self.system.get_psf_kwargs(frame)
        mask = self.mask_frames[frame]
        num = 0
        j = 0
        A_f = np.zeros((num_frames*num_sources, len(ra_grid)))
        for source in self.modeled_sources:
            numParam = source.get_num_param()
            param = param_array[num:num+numParam]
            num += numParam
            A_s_f = source.get_response_single(ra_grid, dec_grid, psf_kwargs, numPix, deltaPix, self.subgrid_res, param=param, mask=mask, kwargs_lens=self._kwargs_lens)
            A_f[i+j] = A_s_f
            j += num_frames
        return A_f

    def get_inverted(self, A, C_D_inv, d):
        """
        get the response matrix for one frame
        B:1d array of all parameters
        image: 1d array of all data (multiple bands)
        """
        param_array, _, image_array = self.deLens.get_param_WLS(A, C_D_inv, d, inv_bool=False)
        return param_array, image_array

    def get_model_images(self, model_array):
        """
        returns a list of 2d images (of all modeled bands) given a 1d list of the combined bands
        """
        if not hasattr(self, "_data_list"):
            raise ValueError("There is no instance of _data_list. You must run this routine with the same instance"
                             "of the class as the inversion.")
        image_list = []
        num = 0
        for i in range(len(self._data_list)):
            numPix = len(self._data_list[i])
            image_list.append(model_array[num:num+numPix])
            num += numPix
        return image_list

    def get_residuals(self, model_array):
        """
        returns a list of 2d residuals (of all modeled bands) given a 1d list of the combined bands
        """
        if not hasattr(self, "_data_list"):
            raise ValueError("There is no instance of _data_list. You must run this routine with the same instance"
                             "of the class as the inversion.")
        residual_list = []
        num = 0
        for i in range(len(self._data_list)):
            numPix = len(self._data_list[i])
            model = model_array[num:num+numPix]
            residual = (self._data_list[i] - model)*np.sqrt(self._C_D_inv_list[i])
            residual_list.append(residual)
            num += numPix
        return residual_list

    def get_deconvolved_frame(self, param_array, frame):
        """
        returns the deconvolved bands given the parameter list for one band=frame
        """
        ra_grid, dec_grid = self.system.get_coords(frame)
        mask = self.mask_frames[frame]
        image_un_convolve = np.zeros_like(ra_grid)
        num_param_list = self.get_num_param_list()
        num = 0
        for i in range(len(self.modeled_sources)):
            source = self.modeled_sources[i]
            numParam = num_param_list[i]
            image_un_convolve += source.get_lensed_unconvolved(param_array[num:num+numParam], ra_grid, dec_grid, frame, self._kwargs_lens, mask=mask)
            num += numParam
        return image_un_convolve

    def get_deconvolved(self, param_array):
        """
        returns list of de-convolved images
        """
        de_convolve_list = []
        for frame in self.frame_list:
            de_convolve_frame = self.get_deconvolved_frame(param_array, frame)
            de_convolve_list.append(de_convolve_frame)
        return de_convolve_list

    def get_sources_original(self, param_array, frame, scale=1):
        """
        returns the different SED sources separately for one given frame of choice
        """
        ra_grid, dec_grid = self.system.get_coords(frame)
        ra_grid *= scale
        dec_grid *= scale
        mask = self.mask_frames[frame]
        num_param_list = self.get_num_param_list()
        source_list = []
        num = 0
        for i in range(len(self.modeled_sources)):
            source = self.modeled_sources[i]
            numParam = num_param_list[i]
            source_image = source.get_unconvolved(param_array[num:num+numParam], ra_grid, dec_grid, frame, self._kwargs_lens, mask=mask)
            source_list.append(source_image)
            num += numParam
        return source_list

    def get_sources_lensed(self, param_array, frame):
        """
        returns the lensed (but unconvolved sources)
        """
        ra_grid, dec_grid = self.system.get_coords(frame)
        mask = self.mask_frames[frame]
        num_param_list = self.get_num_param_list()
        source_list = []
        num = 0
        for i in range(len(self.modeled_sources)):
            source = self.modeled_sources[i]
            numParam = num_param_list[i]
            source_image = source.get_lensed_unconvolved(param_array[num:num+numParam], ra_grid, dec_grid, frame, self._kwargs_lens, mask=mask)
            source_list.append(source_image)
            num += numParam
        return source_list

    def get_sources_image(self, param_array, frame):
        """
        returns the lensed and convolved images per source in one band
        """
        ra_grid, dec_grid = self.system.get_coords(frame)
        mask = self.mask_frames[frame]
        num_param_list = self.get_num_param_list()
        numPix = self.system.get_numPix(frame)
        deltaPix = self.system.get_deltaPix(frame)
        psf_kwargs = self.system.get_psf_kwargs(frame)
        source_list = []
        num = 0
        for i in range(len(self.modeled_sources)):
            source = self.modeled_sources[i]
            numParam = num_param_list[i]
            source_image = source.get_response_single(ra_grid, dec_grid, psf_kwargs, numPix, deltaPix, self.subgrid_res, param=param_array[num:num+numParam], mask=mask, kwargs_lens=self._kwargs_lens)
            source_list.append(source_image)
            num += numParam
        return source_list

    def get_num_param_list(self):
        """
        returns a list with the number of parameters per source
        """
        num_param_list = []
        for source in self.modeled_sources:
            numParam = source.get_num_param()
            num_param_list.append(numParam)
        return num_param_list

    def get_data_list(self):
        """
        returns a list with the original data to be modeled
        """
        if not hasattr(self, "_data_list"):
            raise ValueError("There is no instance of _data_list. You must run this routine with the same instance"
                             "of the class as the inversion.")
        return self._data_list


class Source(object):
    """
    class for the sources with shapelet configurations, positions and SED component
    """
    def __init__(self, name, ra_pos, dec_pos, beta, n_max, w_SED=None, lens_bool=False, lens_type="SIS"):
        self.name = name
        self.ra_pos = ra_pos
        self.dec_pos = dec_pos
        self.beta = beta
        self.n_max = n_max
        if w_SED is None:
            w_SED = {"g_band": 1, "r_band": 1, "i_band": 1,
            "Y_band": 1, "z_band": 1}
        self.w_SED = w_SED
        self.lens_bool = lens_bool
        self.deLens = DeLens()
        self.lensModel = LensModel(lens_type)

    def update_SED(self, w_SED):
        self.w_SED = w_SED

    def update_lens_type(self, lens_type):
        self.lensModel = LensModel(lens_type)

    def get_response_matrix(self, ra_grid, dec_grid, psf_kwargs, numPix, deltaPix, subgrid_res, kwargs_lens=None, mask=1):
        """
        response matrix of the source with given grid
        :param ra_grid:
        :param dec_grid:
        :param psf_kwargs:
        :param numPix:
        :param deltaPix:
        :param subgrid_res:
        :return:
        """
        if self.lens_bool is True:
            ra_coords, dec_coords = self.lensing(ra_grid, dec_grid, kwargs_lens)
            ra_c, dec_c = self.lensing(self.ra_pos, self.dec_pos, kwargs_lens)
        else:
            ra_coords, dec_coords = ra_grid, dec_grid
            ra_c, dec_c = self.ra_pos, self.dec_pos
        A = self.deLens.get_response_matrix(self.n_max, ra_coords, dec_coords, psf_kwargs, numPix, deltaPix, subgrid_res
                                            , ra_c, dec_c, self.beta, mask=mask)
        return A

    def get_response_single(self, ra_grid, dec_grid, psf_kwargs, numPix, deltaPix, subgrid_res, param, kwargs_lens=None, mask=1):
        """
        response matrix of the source with given grid
        :param ra_grid:
        :param dec_grid:
        :param psf_kwargs:
        :param numPix:
        :param deltaPix:
        :param subgrid_res:
        :return:
        """
        if self.lens_bool is True:
            ra_coords, dec_coords = self.lensing(ra_grid, dec_grid, kwargs_lens)
            ra_c, dec_c = self.lensing(self.ra_pos, self.dec_pos, kwargs_lens)
        else:
            ra_coords, dec_coords = ra_grid, dec_grid
            ra_c, dec_c = self.ra_pos, self.dec_pos
        A = self.deLens.get_response_single(self.n_max, ra_coords, dec_coords, psf_kwargs, numPix, deltaPix, subgrid_res
                                            , ra_c, dec_c, self.beta, param, mask=mask)
        return A

    def get_unconvolved(self, param, ra_grid, dec_grid, frame, kwargs_lens, mask=1):
        """
        return the unconvolved array (1d) for one band and one source type with the parameter values
        :param param:
        :param ra_grid:
        :param dec_grid:
        :return:
        """
        if self.lens_bool is True:
            ra_c, dec_c = self.lensing(self.ra_pos, self.dec_pos, kwargs_lens)
        else:
            ra_c, dec_c = self.ra_pos, self.dec_pos
        A_unconv = self.deLens.get_unconvloved(param, self.n_max, self.beta, ra_c, dec_c, ra_grid, dec_grid, mask)
        return A_unconv * self.w_SED[frame]

    def get_lensed_unconvolved(self, param, ra_grid, dec_grid, frame, kwargs_lens, mask=1):
        """
        return the unconvolved but lensed  array (1d) for one band and one source type with the parameter values
        :param param:
        :param ra_grid:
        :param dec_grid:
        :return:
        """
        if self.lens_bool is True:
            ra_coords, dec_coords = self.lensing(ra_grid, dec_grid, kwargs_lens)
            ra_c, dec_c = self.lensing(self.ra_pos, self.dec_pos, kwargs_lens)
        else:
            ra_coords, dec_coords = ra_grid, dec_grid
            ra_c, dec_c = self.ra_pos, self.dec_pos
        A_unconv = self.deLens.get_unconvloved(param, self.n_max, self.beta, ra_c, dec_c, ra_coords, dec_coords, mask)
        return A_unconv * self.w_SED[frame]

    def get_num_param(self):
        """

        :return: int, number of parameters involved in the source
        """
        num_param = (self.n_max+2)*(self.n_max+1)/2
        return num_param

    def lensing(self, ra_grid, dec_grid, kwargs_lens):
        """

        :param ra_grid: grid in RA in the image plane
        :param dec_grid: grid in DEC in the image plane
        :param kwargs_lens: keyword arguments of the lens mapping
        :return: coordinates in ra, dec in the source plane
        """
        alpha1, alpha2 = self.lensModel.alpha(ra_grid, dec_grid, **kwargs_lens)
        return ra_grid-alpha1, dec_grid-alpha2