__author__ = 'sibirrer'

import lenstronomy.util as util
import numpy as np


class ExternalShear(object):
    """
    new class for external shear e1, e2 expression
    """

    def function(self, x, y, e1, e2):
        # change to polar coordinates
        psi_ext, gamma_ext = util.ellipticity2phi_gamma(e1, e2)
        theta, phi = util.cart2polar(x, y)
        f_ = 1./2 * gamma_ext * theta * np.cos(2*(phi - psi_ext))
        return f_

    def derivatives(self, x, y, e1, e2):
        # rotation angle
        f_x = e1*x - e2*y
        f_y = -e2*x - e1*y
        return f_x, f_y

    def hessian(self, x, y, e1, e2):
        gamma1 = e1
        gamma2 = e2
        kappa = 0
        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        return f_xx, f_yy, f_xy

    def all(self, x, y, e1, e2):
        psi_ext, gamma_ext = util.ellipticity2phi_gamma(e1, e2)
        theta, phi = util.cart2polar(x, y)
        f_ = 1./2 * gamma_ext * theta * np.cos(2*(phi - psi_ext))
        f_x = e1*x - e2*y
        f_y = -e2*x - e1*y
        gamma1 = e1
        gamma2 = e2
        kappa = 0
        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2
        return f_, f_x, f_y, f_xx, f_yy, f_xy