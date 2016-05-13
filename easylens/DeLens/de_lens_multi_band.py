__author__ = 'sibirrer'

from lensDES.DeLens.de_lens import DeLens
from lensDES.FunctionSet.shapelets import Shapelets

import numpy as np

class DeLensMultiBand(DeLens):
    """
    multi-band reconstruction
    """

    def get_param_WLS_multi_band(self, A_list, C_D_inv_list, d_list, w_SED, inv_bool=True):
        """
        returns the parameter values given
        :param A: response matrix Nd x Ns (Nd = # data points, Ns = # parameters)
        :param C_D_inv: inverse covariance matrix of the data, Nd x Nd, only diagonal entries (Nd_ii)
        :param d: data array, 1-d Nd
        :param w_SED: Spectral Energy Distribution description of the object (1D-list)
        (relation between the fluxes of different bands)
        :param inv_bool: boolean, wheter returning also the inverse matrix or just solve the linear system
        :return: 1-d array of parameter values
        """
        num_exp = len(d_list)
        num_data_list = []
        for i in range(num_exp):
            num_data_list.append(len(d_list[i]))
            A_list[i] *= w_SED[i]
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

    def get_param_WLS_multi_band_multi_SED(self, A_list, C_D_inv_list, d_list, w_SED_list, inv_bool=True):
        """
        returns the parameter values given
        :param A: response matrix Nd x Ns (Nd = # data points, Ns = # parameters)
        :param C_D_inv: inverse covariance matrix of the data, Nd x Nd, only diagonal entries (Nd_ii)
        :param d: data array, 1-d Nd
        :param w_SED_list: Spectral Energy Distribution description of the object (1D-lists of lists)
        (relation between the fluxes of different bands)
        :param inv_bool: boolean, wheter returning also the inverse matrix or just solve the linear system
        :return: 1-d array of parameter values
        """
        num_exp = len(d_list)
        num_data_list = []
        A_joint_list = [[] for i in w_SED_list]
        num_param_list = []
        for i in range(num_exp):
            num_data_list.append(len(d_list[i]))
            num_param_list.append(len(A_list[i]))
        for k in range(len(w_SED_list)):
            for i in range(num_exp):
                A_list[i] *= w_SED_list[k][i]
            A_temp = np.concatenate(A_list, axis=1)
            A_joint_list[k] = A_temp
        A_joint = np.concatenate(A_joint_list, axis=0)
        C_D_inv_joint = np.concatenate(C_D_inv_list, axis=0)
        d_joint = np.concatenate(d_list, axis=0)
        B, M_inv, image_joint = self.get_param_WLS(A_joint, C_D_inv_joint, d_joint, inv_bool)

        image_list = []
        B_list = []
        n_data = 0
        n_param = 0
        for i in range(num_exp):
            numPix = num_data_list[i]
            image_list.append(image_joint[n_data:n_data+numPix])
            numParam = num_param_list[i]
            B_list.append(B[n_param:n_param+numParam])
            n_data += numPix
            n_param += numParam
        return B_list, M_inv, image_list

    def get_unconvloved_multi_SED(self, param_list, num_order, beta, center_x, center_y, x_grid, y_grid, w_SED_list):
        """

        :param param:
        :param num_order:
        :param beta:
        :param center_x:
        :param center_y:
        :return: list of source models (with different SEDs)
        """
        source_list = [[] for i in w_SED_list]
        shapelets = Shapelets(interpolation=False, precalc=False)
        for k in range(len(w_SED_list)):
            source = np.zeros(len(x_grid))
            param = param_list[k]
            n1 = 0
            n2 = 0
            for i in range(0, len(param)):
                source += shapelets.function(x_grid, y_grid, param[i], beta, n1, n2, center_x, center_y)
                if n1 + n2 < num_order:
                    n1 += 1
                else:
                    n1 = 0
                    n2 += 1
            source_list[k] = source
        return source_list