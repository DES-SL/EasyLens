__author__ = 'sibirrer'

import numpy as np
import numpy.polynomial.hermite as hermite
import math
import scipy.interpolate as interpolate

class Shapelets(object):
    """

    """
    def __init__(self, interpolation=False, precalc=True):
        """
        load interpolation of the Hermite polynomials in a range [-30,30] in order n<= 50
        :return:
        """
        self.interpolation = interpolation
        self.precalc = precalc
        if interpolation:
            n_order = 50
            self.H_interp = [[] for i in range(0, n_order)]
            self.x_grid = np.linspace(-50, 50, 6000)
            for k in range(0, n_order):
                n_array = np.zeros(k+1)
                n_array[k] = 1
                values = hermite.hermval(self.x_grid, n_array)
                self.H_interp[k] = values
            print 'H interpolated'

    def function(self, x, y, amp, beta, n1, n2, center_x, center_y):
        """

        :param amp: amplitude of shapelet
        :param beta: scale factor of shapelet
        :param n1: x-order
        :param n2: y-order
        :param center_x: center in x
        :param center_y: center in y
        :return:
        """
        if self.precalc:
            return amp * x[n1] * y[n2]  # / beta
        x_ = x - center_x
        y_ = y - center_y
        return amp * self.phi_n(n1, x_/beta) * self.phi_n(n2, y_/beta)  # /beta

    def H_n(self, n, x):
        """
        constructs the Hermite polynomial of order n at position x (dimensionless)

        :param n: The n'the basis function.
        :type name: int.
        :param x: 1-dim position (dimensionless)
        :type state: float or numpy array.
        :returns:  array-- H_n(x).
        :raises: AttributeError, KeyError
        """
        if not self.interpolation:
            n_array = np.zeros(n+1)
            n_array[n] = 1
            return hermite.hermval(x, n_array, tensor=False) #attention, this routine calculates every single hermite polynomial and multiplies it with zero (exept the right one)
        else:
            return np.interp(x, self.x_grid, self.H_interp[n])
            #return self.H_interp[n](x)

    def phi_n(self, n, x):
        """
        constructs the 1-dim basis function (formula (1) in Refregier et al. 2001)

        :param n: The n'the basis function.
        :type name: int.
        :param x: 1-dim position (dimensionless)
        :type state: float or numpy array.
        :returns:  array-- phi_n(x).
        :raises: AttributeError, KeyError
        """
        prefactor = 1./np.sqrt(2**n*np.sqrt(np.pi)*math.factorial(n))
        return prefactor*self.H_n(n, x)*np.exp(-x**2/2.)

    def pre_calc(self, x, y, beta, n_order, center_x, center_y):
        """
        calculates the H_n(x) and H_n(y) for a given x-array and y-array
        :param x:
        :param y:
        :param amp:
        :param beta:
        :param n_order:
        :param center_x:
        :param center_y:
        :return: list of H_n(x) and H_n(y)
        """
        x_ = x - center_x
        y_ = y - center_y
        H_x = np.empty((n_order+1, len(x)))
        H_y = np.empty((n_order+1, len(x)))
        if n_order > 170:
            raise ValueError('polynomial order to large', n_order)
        for n in range(0, n_order+1):

            prefactor = 1./np.sqrt(2**n*np.sqrt(np.pi)*math.factorial(n))
            n_array = np.zeros(n+1)
            n_array[n] = 1
            H_x[n] = hermite.hermval(x_/beta, n_array) * prefactor * np.exp(-(x_/beta)**2/2.)
            H_y[n] = hermite.hermval(y_/beta, n_array) * prefactor * np.exp(-(y_/beta)**2/2.)
        return H_x, H_y