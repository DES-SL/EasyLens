__author__ = 'sibirrer'

import numpy as np
import scipy.ndimage.interpolation as interp
import astropy.io.fits as pyfits

import lenstronomy.DataAnalysis.Sextractor_wrapper.pysex as pysex
import lensDES.util as util


class ImageAnalysis(object):
    """
    class for analysis routines acting on a single image
    """
    def __init__(self):
        pass

    def estimate_bkg(self, image):
        """

        :param image: 2d numpy array
        :return: mean and sigma of background estimate
        """
        HDUFile, _ = self._get_cat(image)
        mean, rms = self._get_background(HDUFile)
        return mean, rms

    def estimate_psf(self, path2exposure, kernel_size=21, kwargs_cut={}, restrict_psf=None):
        """
        esitmates a psf kernel
        :param image:
        :return:
        """
        fits = pyfits.open(path2exposure)
        image = fits[0].data
        fits.close()
        HDUFile, image_no_borders = self._get_cat(image)
        mean, rms = self._get_background(HDUFile)
        cat = self._get_source_cat(HDUFile)
        if kwargs_cut == {}:
            kwargs_cut = self._estimate_star_thresholds(cat)
        mask = self._find_objects(cat, kwargs_cut)
        mag = np.array(cat.data['MAG_BEST'], dtype=float)
        size = np.array(cat.data['FLUX_RADIUS'], dtype=float)
        x_list, y_list, restrict_psf = self._get_coordinates(image_no_borders, cat, mask, numPix=41, restrict_psf=restrict_psf)
        if len(x_list) == 0:
            return np.zeros((kernel_size,kernel_size)), restrict_psf, x_list, y_list, mask, mag, size, kwargs_cut
        star_list = self._get_objects_image(image_no_borders, x_list, y_list, numPix=41)
        kernel = self._stacking(star_list, x_list, y_list)
        kernel =util.cut_edges(kernel, kernel_size)
        kernel = util.kernel_norm(kernel)
        return kernel, restrict_psf, x_list, y_list, mask, mag, size, kwargs_cut

    def _get_cat(self, image, conf_args={}):
        """
        returns the sextractor catalogue of a given image
        :param system:
        :param image_name:
        :return:
        """
        nx, ny = image.shape
        borders = int(nx/10)
        image_no_borders = image[borders:ny-borders,borders:nx-borders]

        params = ['NUMBER', 'FLAGS', 'X_IMAGE', 'Y_IMAGE', 'FLUX_BEST', 'FLUXERR_BEST', 'MAG_BEST', 'MAGERR_BEST',
                    'FLUX_RADIUS', 'CLASS_STAR', 'A_IMAGE', 'B_IMAGE', 'THETA_IMAGE', 'ELLIPTICITY']
        HDUFile = pysex.run(image_no_borders, params=params, conf_file=None, conf_args=conf_args, keepcat=False, rerun=False, catdir=None)
        return HDUFile, image_no_borders

    def _get_source_cat(self, HDUFile):
        """

        :param HDUFile:
        :return: catalogue
        """
        return HDUFile[2]

    def _get_background(self, HDUFile):
        """
        filters the mean and rms value of the background computed by sextractor
        :param cat:
        :return: mean, rms
        """
        mean_found = False
        rms_found = False
        list = HDUFile[1].data[0][0]
        for line in list:
            line = line.strip()
            line = line.split()
            if line[0] == 'SEXBKGND' or line[0] == 'SEXBKGND=':
                mean = float(line[1])
                mean_found = True
            if line[0] == 'SEXBKDEV' or line[0] == 'SEXBKDEV=':
                rms = float(line[1])
                rms_found = True
        if mean_found == False or rms_found == False:
            raise ValueError('no mean and rms value found in list.')
        return mean, rms

    def _estimate_star_thresholds(self, cat):
        """
        estimates the cuts in the different sextractor quantities
        :param cat:
        :return:
        """
        mag = np.array(cat.data['MAG_BEST'],dtype=float)
        size = np.array(cat.data['FLUX_RADIUS'],dtype=float)
        #ellipticity = cat.data['ELLIPTICITY']

        kwargs_cuts = {}
        mag_max = min(np.max(mag), 34)
        mag_min = np.min(mag)
        delta_mag = mag_max - mag_min
        kwargs_cuts['MagMaxThresh'] = mag_max - 0.7*delta_mag
        kwargs_cuts['MagMinThresh'] = mag_min #+ 0.01*delta_mag

        mask = (mag<mag_max-0.5*delta_mag)
        kwargs_cuts['SizeMinThresh'] = max(0, np.min(size[mask]))
        kwargs_cuts['SizeMaxThresh'] = max(0, np.min(size[mask])+4)
        kwargs_cuts['EllipticityThresh'] = 0.1
        kwargs_cuts['ClassStarMax'] = 1.
        kwargs_cuts['ClassStarMin'] = 0.5
        return kwargs_cuts

    def _find_objects(self, cat, kwargs_cut):
        """

        :param cat: hdu[2] catalogue objects comming from sextractor
        :return: selected objects in the catalogue data list
        """
        mag = np.array(cat.data['MAG_BEST'],dtype=float)
        size = np.array(cat.data['FLUX_RADIUS'],dtype=float)
        ellipticity = cat.data['ELLIPTICITY']
        classStar = cat.data['CLASS_STAR']
        SizeMaxThresh = kwargs_cut['SizeMaxThresh']
        SizeMinThresh = kwargs_cut['SizeMinThresh']
        EllipticityThresh = kwargs_cut['EllipticityThresh']
        MagMaxThresh = kwargs_cut['MagMaxThresh']
        MagMinThresh = kwargs_cut['MagMinThresh']
        ClassStarMax = kwargs_cut['ClassStarMax']
        ClassStarMin = kwargs_cut['ClassStarMin']

        mask = (size<SizeMaxThresh) & (ellipticity<EllipticityThresh) & (size>SizeMinThresh) & (mag<MagMaxThresh) & (mag>MagMinThresh) & (classStar<ClassStarMax) & (classStar>ClassStarMin)
        return mask

    def _get_coordinates(self, image, cat, mask, numPix=10, restrict_psf=None):
        """

        :param image:
        :param cat:
        :param mask:
        :param restrict_psf:
        :return:
        """
        nx, ny = image.shape
        x_center = np.array(cat.data['X_IMAGE'], dtype=float)
        y_center = np.array(cat.data['Y_IMAGE'], dtype=float)
        x_center_mask = x_center[mask]
        y_center_mask = y_center[mask]
        num_objects = len(x_center_mask)
        if restrict_psf == None:
            restrict_psf = [True]*num_objects
        x_list = []
        y_list = []
        for i in range(num_objects):
            xc, yc = x_center_mask[i], y_center_mask[i]
            if (int(xc)-numPix > 0) and (int(xc)+numPix < nx) and (int(yc)-numPix > 0) and (int(yc)+numPix < ny):
                if restrict_psf[i]:
                    x_list.append(xc)
                    y_list.append(yc)
        return x_list, y_list, restrict_psf

    def _get_objects_image(self, image, x_list, y_list, numPix=10):
        """
        returns all the cutouts of the locations of the selected objects
        :param image:
        :param cat:
        :param mask:
        :return:
        """
        num_objects = len(x_list)
        cutout_list = []
        print("number of objects: ", num_objects)
        for i in range(np.minimum(10, num_objects)):
            xc, yc = x_list[i], y_list[i]
            cutout = image[int(xc)-numPix-1:int(xc)+numPix, int(yc)-numPix-1:int(yc)+numPix]
            cutout_list.append(cutout)
        return cutout_list

    def _stacking(self, star_list, x_list, y_list):
        """

        :param star_list:
        :return:
        """
        n_stars = len(star_list)

        shifteds = []
        for i in range(n_stars):
            xc, yc = x_list[i], y_list[i]
            data = star_list[i]
            x_shift = int(xc) - xc
            y_shift = int(yc) - yc
            shifted = interp.shift(data, [-y_shift, -x_shift], order=1)
            shifteds.append(shifted)
            print '=== object ===', i
            import matplotlib.pylab as plt
            fig, ax1 = plt.subplots()
            im = ax1.matshow(np.log10(shifted), origin='lower')
            plt.axes(ax1)
            fig.colorbar(im)
            plt.show()

        combined = sum(shifteds)
        new=np.empty_like(combined)
        max_pix = np.max(combined)
        p = combined[combined>=max_pix/10**6]  #in the SIS regime
        new[combined < max_pix/10**6] = 0
        new[combined >= max_pix/10**6] = p
        kernel = util.kernel_norm(new)
        return kernel