__author__ = 'sibirrer'

import numpy as np
from fastell4py import fastell4py
#TODO write fastell4py as a requirment

class SPEMD(object):
    """
    class for smooth power law ellipse mass density profile
    """
    def __init__(self):
        self.s2 = 0.01

    def function(self, x, y, phi_E, gamma, q, phi_G, center_x=0, center_y=0):
        if gamma < 1.4:
            gamma = 1.4
        if gamma > 2.9:
            gamma = 2.9
        if q < 0.3:
            q = 0.3
        if q > 1:
            q = 1.
        x_shift = x - center_x
        y_shift = y - center_y
        q_fastell, gam = self.convert_params(phi_E, gamma, q)

        cos_phi = np.cos(phi_G)
        sin_phi = np.sin(phi_G)

        x1 = cos_phi*x_shift+sin_phi*y_shift
        x2 = -sin_phi*x_shift+cos_phi*y_shift

        potential = fastell4py.ellipphi(x1, x2, q_fastell, gam, arat=q, s2=self.s2)
        return potential

    def derivatives(self, x, y, phi_E, gamma, q, phi_G, center_x=0, center_y=0):
        if gamma < 1.4:
            gamma = 1.4
        if gamma > 2.9:
            gamma = 2.9
        if q < 0.3:
            q = 0.3
        if q > 1:
            q = 1.
        x_shift = x - center_x
        y_shift = y - center_y
        q_fastell, gam = self.convert_params(phi_E, gamma, q)

        cos_phi = np.cos(phi_G)
        sin_phi = np.sin(phi_G)

        x1 = cos_phi*x_shift+sin_phi*y_shift
        x2 = -sin_phi*x_shift+cos_phi*y_shift

        f_x_prim, f_y_prim = fastell4py.fastelldefl(x1, x2, q_fastell, gam, arat=q, s2=self.s2)
        f_x = cos_phi*f_x_prim - sin_phi*f_y_prim
        f_y = sin_phi*f_x_prim + cos_phi*f_y_prim
        return f_x, f_y

    def hessian(self, x, y, phi_E, gamma, q, phi_G, center_x=0, center_y=0):
        if gamma < 1.4:
            gamma = 1.4
        if gamma > 2.9:
            gamma = 2.9
        if q < 0.3:
            q = 0.3
        if q > 1:
            q = 1.
        x_shift = x - center_x
        y_shift = y - center_y
        q_fastell, gam = self.convert_params(phi_E, gamma, q)

        cos_phi = np.cos(phi_G)
        sin_phi = np.sin(phi_G)

        x1 = cos_phi*x_shift+sin_phi*y_shift
        x2 = -sin_phi*x_shift+cos_phi*y_shift

        f_x_prim, f_y_prim, f_xx_prim, f_yy_prim, f_xy_prim = fastell4py.fastellmag(x1, x2, q_fastell, gam, arat=q, s2=self.s2)

        kappa = (f_xx_prim + f_yy_prim)/2
        gamma1_value = (f_xx_prim - f_yy_prim)/2
        gamma2_value = f_xy_prim

        gamma1 = np.cos(2*phi_G)*gamma1_value-np.sin(2*phi_G)*gamma2_value
        gamma2 = +np.sin(2*phi_G)*gamma1_value+np.cos(2*phi_G)*gamma2_value

        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        return f_xx, f_yy, f_xy

    def all(self, x, y, phi_E, gamma, q, phi_G, center_x=0, center_y=0):
        if gamma < 1.4:
            gamma = 1.4
        if gamma > 2.9:
            gamma = 2.9
        if q < 0.3:
            q = 0.3
        if q > 1:
            q = 1.
        x_shift = x - center_x
        y_shift = y - center_y
        q_fastell, gam = self.convert_params(phi_E, gamma, q)

        cos_phi = np.cos(phi_G)
        sin_phi = np.sin(phi_G)

        x1 = cos_phi*x_shift+sin_phi*y_shift
        x2 = -sin_phi*x_shift+cos_phi*y_shift
        f_ = fastell4py.ellipphi(x1, x2, q_fastell, gamma, arat=q, s2=0)
        f_x_prim, f_y_prim, f_xx_prim, f_yy_prim, f_xy_prim = fastell4py.fastellmag(x1, x2, q_fastell, gam, arat=q, s2=self.s2)
        f_x = cos_phi*f_x_prim - sin_phi*f_y_prim
        f_y = sin_phi*f_x_prim + cos_phi*f_y_prim

        kappa = (f_xx_prim + f_yy_prim)/2
        gamma1_value = (f_xx_prim - f_yy_prim)/2
        gamma2_value = f_xy_prim

        gamma1 = np.cos(2*phi_G)*gamma1_value-np.sin(2*phi_G)*gamma2_value
        gamma2 = +np.sin(2*phi_G)*gamma1_value+np.cos(2*phi_G)*gamma2_value

        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        return f_, f_x, f_y, f_xx, f_yy, f_xy

    def convert_params(self, phi_E, gamma, q):
        """

        :param phi_E: Einstein radius
        :param gamma: power law slope
        :param q: axis ratio
        :return:   prefactor to SPEMP profile for FASTELL
        """
        gam = (gamma-1)/2.
        # gam = gamma
        q_fastell = (3-gamma)/2. * (phi_E**2/q)**gam
        return q_fastell, gam