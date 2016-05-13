__author__ = 'sibirrer'

import numpy as np


from lensDES.lensDES import LensDES


class SED_estimate(LensDES):
    """
    class to estimate the best fit SED given a source model
    """

    def get_response_frame(self, frame):
        """
        get the response matrix for one frame
        """
        A_f_list = []
        ra_grid, dec_grid = self.system.get_coords(frame)
        numPix = self.system.get_numPix(frame)
        deltaPix = self.system.get_deltaPix(frame)
        psf_kwargs = self.system.get_psf_kwargs(frame)
        for source in self.modeled_sources:
            w_sed = source.w_SED[frame]
            A_f_i = source.get_response_matrix(ra_grid, dec_grid, psf_kwargs, numPix, deltaPix, self.subgrid_res)
            A_f_list.append(A_f_i*w_sed)
        A_f = np.concatenate(A_f_list, axis=0)
        return A_f