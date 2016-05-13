__author__ = 'sibirrer'

import numpy as np

class SIS(object):
    """
    this class contains the function and the derivatives of the Singular Isothermal Sphere
    """
    def function(self, x, y, phi_E_sis, center_x_sis=0, center_y_sis=0):
        x_shift = x - center_x_sis
        y_shift = y - center_y_sis
        f_ = phi_E_sis * np.sqrt(x_shift*x_shift + y_shift*y_shift)
        return f_

    def derivatives(self, x, y, phi_E_sis, center_x_sis=0, center_y_sis=0):
        """
        returns df/dx and df/dy of the function
        """
        x_shift = x - center_x_sis
        y_shift = y - center_y_sis
        R = np.sqrt(x_shift*x_shift + y_shift*y_shift)
        if isinstance(R, int) or isinstance(R, float):
            a = phi_E_sis/max(0.000001,R)
        else:
            a=np.empty_like(R)
            r = R[R > 0]  #in the SIS regime
            a[R == 0] = 0
            a[R > 0] = phi_E_sis/r
        f_x = a * x_shift
        f_y = a * y_shift
        return f_x, f_y

    def hessian(self, x, y, phi_E_sis, center_x_sis=0, center_y_sis=0):
        """
        returns Hessian matrix of function d^2f/dx^2, d^f/dy^2, d^2/dxdy
        """
        x_shift = x - center_x_sis
        y_shift = y - center_y_sis
        R = (x_shift*x_shift + y_shift*y_shift)**(3./2)
        if isinstance(R, int) or isinstance(R, float):
            prefac = phi_E_sis/max(0.000001,R)
        else:
            prefac = np.empty_like(R)
            r = R[R>0]  #in the SIS regime
            prefac[R==0] = 0.
            prefac[R>0] = phi_E_sis/r

        f_xx = y_shift*y_shift * prefac
        f_yy = x_shift*x_shift * prefac
        f_xy = -x_shift*y_shift * prefac
        return f_xx, f_yy, f_xy

    def all(self, x, y, phi_E_sis, center_x_sis=0, center_y_sis=0):
        """
        returns f,f_x,f_y,f_xx, f_yy, f_xy
        """
        x_shift = x - center_x_sis
        y_shift = y - center_y_sis
        R = np.sqrt(x_shift*x_shift + y_shift*y_shift)
        if isinstance(R, int) or isinstance(R, float):
            a = phi_E_sis/max(0.000001,R)
        else:
            a=np.empty_like(R)
            r = R[R>0]  #in the SIS regime
            a[R==0] = 0.
            a[R>0] = phi_E_sis/r


        f_ = phi_E_sis * np.sqrt(x_shift*x_shift + y_shift*y_shift)
        f_x = a * x_shift
        f_y = a * y_shift
        R = (x_shift*x_shift + y_shift*y_shift)**(3./2)
        if isinstance(R, int) or isinstance(R, float):
            prefac = phi_E_sis/max(0.000001,R)
        else:
            prefac = np.empty_like(R)
            r = R[R>0]  #in the SIS regime
            prefac[R==0] = 0.
            prefac[R>0] = phi_E_sis/r

        f_xx = y_shift*y_shift * prefac
        f_yy = x_shift*x_shift * prefac
        f_xy = -x_shift*y_shift * prefac
        return f_, f_x, f_y, f_xx, f_yy, f_xy