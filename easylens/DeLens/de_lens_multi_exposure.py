__author__ = 'sibirrer'

from lensDES.FunctionSet.shapelets import Shapelets
from lensDES.DeLens.de_lens import DeLens

import numpy as np
import sys

class DeLensMultiExp(DeLens):
    """
    class to deal with multiple exposures of the same band
    """

    def get_param_WLS_multi(self, A_list, C_D_inv_list, d_list, inv_bool=True):
        """
        returns the parameter values given
        :param A: response matrix Nd x Ns (Nd = # data points, Ns = # parameters)
        :param C_D_inv: inverse covariance matrix of the data, Nd x Nd, only diagonal entries (Nd_ii)
        :param d: data array, 1-d Nd
        :param inv_bool: boolean, wheter returning also the inverse matrix or just solve the linear system
        :return: 1-d array of parameter values
        """
        num_exp = len(d_list)
        num_data_list = []
        for i in range(num_exp):
            num_data_list.append(len(d_list[i]))
        A_joint = np.concatenate(A_list, axis=1)
        C_D_inv_joint = np.concatenate(C_D_inv_list, axis=0)
        d_joint = np.concatenate(d_list, axis=0)
        B, M_inv, image_joint = self.get_param_WLS(A_joint, C_D_inv_joint, d_joint, inv_bool)
        image_list = []
        n = 0
        for i in range(num_exp):
            numPix = num_data_list[i]
            image_list.append(image_joint[n:n+numPix])
            n += numPix
        return B, M_inv, image_list