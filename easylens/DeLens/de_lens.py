__author__ = 'sibirrer'

import numpy as np
import sys
import scipy.ndimage as ndimage
import scipy.signal as signal


from easylens.FunctionSet.shapelets import Shapelets
import easylens.util as util


class DeLens(object):
    """
    class for the de-lensing algorithm
    """
    def __init__(self):
        self.shapelets = Shapelets()

    def get_param_WLS(self, A, C_D_inv, d, inv_bool=True):
        """
        returns the parameter values given
        :param A: response matrix Nd x Ns (Nd = # data points, Ns = # parameters)
        :param C_D_inv: inverse covariance matrix of the data, Nd x Nd, only diagonal entries (Nd_ii)
        :param d: data array, 1-d Nd
        :param inv_bool: boolean, wheter returning also the inverse matrix or just solve the linear system
        :return: 1-d array of parameter values
        """
        M = A.dot(np.multiply(C_D_inv, A).T)

        if inv_bool:
            if np.linalg.cond(M) < 1/sys.float_info.epsilon:
                M_inv = np.linalg.inv(M)
            else:
                M_inv = np.zeros_like(M)
            R = A.dot(np.multiply(C_D_inv, d))
            B = M_inv.dot(R)
        else:
            if np.linalg.cond(M) < 1/sys.float_info.epsilon:
                R = A.dot(np.multiply(C_D_inv, d))
                B = np.linalg.solve(M, R).T
            else:
                B = np.zeros(len(A))
            M_inv = None
        image = A.T.dot(B)
        return B, M_inv, image

    def get_covariance_matrix(self, d, sigma_b, weight_map, mask=1):
        """
        returns a diagonal matrix for the covariance estimation
        :param d: data array
        :param sigma_b: background noise
        :param weight_map: weighting poissonian factor
        :return: len(d) x len(d) matrix
        """
        sigma = np.abs(d-sigma_b)*weight_map + sigma_b**2
        return sigma*mask

    def get_unconvloved(self, param, num_order, beta, center_x, center_y, x_grid, y_grid, mask=1):
        """

        :param param:
        :param num_order:
        :param beta:
        :param center_x:
        :param center_y:
        :return:
        """
        shapelets = Shapelets(interpolation=False, precalc=False)
        source = np.zeros(len(x_grid))

        n1 = 0
        n2 = 0
        for i in range(0, len(param)):
            source += shapelets.function(x_grid, y_grid, param[i], beta, n1, n2, center_x, center_y)
            if n1 + n2 < num_order:
                n1 += 1
            else:
                n1 = 0
                n2 += 1
        return source * mask

    def get_response_matrix(self, num_order, x_source, y_source, kwargs_psf, numPix, deltaPix, subgrid_res, center_x, center_y, beta, mask=1):
        """

        :param makeImage: instance of a class makeImage with shapelets
        :param x_grid:
        :param y_grid:
        :param num_param:
        :param kwargs_lens:
        :param center_x:
        :param center_y:
        :return:
        """
        num_param = (num_order+2)*(num_order+1)/2
        A = np.zeros((num_param, numPix**2))
        n1 = 0
        n2 = 0
        H_x, H_y = self.shapelets.pre_calc(x_source, y_source, beta, num_order, center_x, center_y)
        for i in range(num_param):
            kwargs_source_shapelet = {'center_x': center_x, 'center_y': center_y, 'n1': n1, 'n2': n2, 'beta': beta, 'amp': 1}
            image = self.shapelets.function(H_x, H_y, **kwargs_source_shapelet)
            image = util.array2image(image)
            image = self.re_size_convolve(image, numPix, deltaPix, subgrid_res, kwargs_psf)
            response = util.image2array(image) * mask
            A[i, :] = response
            if n1 + n2 < num_order:
                n1 += 1
            else:
                n1 = 0
                n2 += 1
        return A

    def get_response_single(self, num_order, x_source, y_source, kwargs_psf, numPix, deltaPix, subgrid_res, center_x, center_y, beta, param, mask=1):
        """

        :param makeImage: instance of a class makeImage with shapelets
        :param x_grid:
        :param y_grid:
        :param num_param:
        :param kwargs_lens:
        :param center_x:
        :param center_y:
        :return:
        """
        num_param = (num_order+2)*(num_order+1)/2
        A = np.zeros(numPix**2)
        n1 = 0
        n2 = 0
        H_x, H_y = self.shapelets.pre_calc(x_source, y_source, beta, num_order, center_x, center_y)
        for i in range(num_param):
            kwargs_source_shapelet = {'center_x': center_x, 'center_y': center_y, 'n1': n1, 'n2': n2, 'beta': beta, 'amp': param[i]}
            image = self.shapelets.function(H_x, H_y, **kwargs_source_shapelet)
            image = util.array2image(image)
            image = self.re_size_convolve(image, numPix, deltaPix, subgrid_res, kwargs_psf)
            response = util.image2array(image) * mask
            A += response
            if n1 + n2 < num_order:
                n1 += 1
            else:
                n1 = 0
                n2 += 1
        return A

    def re_size_convolve(self, image, numPix, deltaPix, subgrid_res, kwargs_psf):
        gridScale = deltaPix/subgrid_res
        if 'kernel' in kwargs_psf:
            grid_re_sized = self.re_size(image, numPix)
            grid_final = self.psf_convolution(grid_re_sized, gridScale, **kwargs_psf)
        else:
            grid_conv = self.psf_convolution(image, gridScale, **kwargs_psf)
            grid_final = self.re_size(grid_conv, numPix)
        return grid_final

    def re_size(self, grid, numPix):
        """
        smooths a given grid to larger pixels
        """
        numGrid = len(grid)

        if numGrid == numPix: #if the grid has the same size as the pixelized image
            return grid
        else:
            numAverage = numGrid/numPix
            if int(numAverage) == numAverage:
                return util.averaging(grid, numGrid, numPix)
            else:
                raise ValueError("grid size = %f is not a integer factor of pixel size = %f " % (numGrid,numPix))

    def psf_convolution(self, grid, grid_scale, **kwargs):
        """
        convolves a given pixel grid with a PSF
        """
        if kwargs['psf_type'] == 'gaussian':
            sigma = kwargs['sigma']/grid_scale
            if 'truncate' in kwargs:
                sigma_truncate = kwargs['truncate']
            else:
                sigma_truncate = 3.
            img_conv = ndimage.filters.gaussian_filter(grid, sigma, mode='nearest', truncate=sigma_truncate)
            return img_conv
        elif kwargs['psf_type'] == 'pixel':
            kernel = kwargs['kernel']
            img_conv = signal.fftconvolve(grid, kernel, mode='same')
            return img_conv
        return grid
