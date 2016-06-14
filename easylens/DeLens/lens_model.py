__author__ = 'sibirrer'

#file which contains class for lens model routines



class LensModel(object):
    """
    class for handling different lens models
    """
    def __init__(self, lens_type):
        if lens_type == 'SIS':
            from easylens.FunctionSet.sis import SIS
            self.func = SIS()
        elif lens_type == 'SPEMD':
            from easylens.FunctionSet.spemd import SPEMD
            self.func = SPEMD()
        else:
            raise ValueError('lens type %s is not a valid lens model!' % lens_type)

    def mass(self, x, y, sigma_crit, **kwargs):
        kappa = self.kappa(x, y, **kwargs)
        mass = sigma_crit*kappa
        return mass

    def potential(self, x, y, **kwargs):
        potential = self.func.function(x, y, **kwargs)
        return potential

    def alpha(self, x, y, **kwargs):
        """
        a = grad(phi)
        """
        f_x, f_y = self.func.derivatives(x, y, **kwargs)
        alpha1 = f_x  # attention on units
        alpha2 = f_y  # attention on units
        return alpha1, alpha2

    def kappa(self, x, y, **kwargs):
        """
        k = 1/2 laplacian(phi)
        """
        f_xx, f_yy, f_xy = self.func.hessian(x, y, **kwargs)
        kappa = 1./2 * (f_xx + f_yy)  # attention on units
        return kappa

    def gamma(self, x, y, **kwargs):
        """
        g1 = 1/2(d^2phi/dx^2 - d^2phi/dy^2)
        g2 = d^2phi/dxdy
        """
        f_xx, f_yy, f_xy = self.func.hessian(x, y, **kwargs)
        gamma1 = 1./2 * (f_xx - f_yy)  # attention on units
        gamma2 = f_xy  # attention on units
        return gamma1, gamma2

    def magnification(self, x, y, **kwargs):
        """
        mag = 1/det(A)
        A = 1 - d^2phi/d_ij
        """
        f_xx, f_yy, f_xy = self.func.hessian(x, y, **kwargs)
        det_A = (1 - f_xx) * (1 - f_yy) - f_xy*f_xy  # attention, only works in right units of critical density
        return 1./det_A  # attention, if dividing to zero

    def all(self, x, y, **kwargs):
        """
        specially build to reduce computational costs
        """
        f_, f_x, f_y, f_xx, f_yy, f_xy = self.func.all(x, y, **kwargs)
        potential = f_
        alpha1 = f_x  # attention on units
        alpha2 = f_y  # attention on units
        kappa = 1./2 * (f_xx + f_yy)  # attention on units
        gamma1 = 1./2 * (f_xx - f_yy)  # attention on units
        gamma2 = f_xy  # attention on units
        det_A = (1 - f_xx) * (1 - f_yy) - f_xy*f_xy  # attention, only works in right units of critical density
        mag = 1./det_A
        return potential, alpha1, alpha2, kappa, gamma1, gamma2, mag