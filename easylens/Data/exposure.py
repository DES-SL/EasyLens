__author__ = 'sibirrer'


import astropy.io.fits as pyfits
import astropy.wcs as pywcs
import numpy as np

from lensDES.Data.image_analysis import ImageAnalysis
import lensDES.util as util

class Exposure(object):
    """
    class to handle an exposure
    """
    def __init__(self, ra_center, dec_center):
        self.analysis = ImageAnalysis()
        self._ra_center = ra_center
        self._dec_center = dec_center
        pass

    def load_exposure(self, path2exposure_fits, cutout_pix=None):
        """

        :param path2fits: path to fits file
        :return: instance of the exposure in the class
        """
        fits = pyfits.open(path2exposure_fits)
        image_full = fits[0].data
        self._header_full = fits[0].header
        fits.close()
        if cutout_pix is None:
            self._image_data = image_full
            self._head = self._header_full
            numPix = len(image_full)
            wcs = pywcs.WCS(self._head)
            x_coords = np.linspace(0, numPix-1, numPix)
            y_coords = np.linspace(0, numPix-1, numPix)
            x_coords, y_coords = np.meshgrid(x_coords, y_coords)
            ra_coords, dec_coords = wcs.all_pix2world(x_coords, y_coords, 0)
            self._ra_coords = util.image2array((ra_coords-self._ra_center)*3600)
            self._dec_coords = util.image2array((dec_coords-self._dec_center)*3600)
        else:
            numPix = cutout_pix*2
            self._image_data, self._head, self._ra_coords, self._dec_coords = self.get_cutout(image_full, self._header_full, cutout_pix)
        self._deltaPix = self._head['CD2_2']*3600.
        self._mean_bkg, self._sigma_bkg = self.analysis.estimate_bkg(image_full)


        self._numPix = numPix

    def load_psf(self, path2psf):
        """

        :param path2psf: path to psf file
        :return:
        """
        self._psf = pyfits.open(path2psf)[0].data

    def load_weight_map(self, path2weight, cutout_pix=None):
        """

        :param path2weight: path to weight map
        :return:
        """
        fits = pyfits.open(path2weight)
        if cutout_pix is None:
            self._weight_map = fits[0].data
        else:
            weight_map_full = fits[1].data
            header_full = fits[1].header
            self._weight_map, _, _, _ = self.get_cutout(weight_map_full, header_full, cutout_pix)
        fits.close()

    def load_all(self, path2exposure_fits, path2psf, path2weight, cutout_pix=None):
        """
        load exposure, psf and weigh map
        :param path2exposure_fits:
        :param path2psf:
        :param path2weight:
        :return:
        """
        self.load_exposure(path2exposure_fits, cutout_pix=cutout_pix)
        if path2psf == path2exposure_fits:
            kernel, restrict_psf, x_list, y_list, mask, mag, size, kwargs_cut = self.analysis.estimate_psf(path2exposure_fits)
            self._psf = kernel
        else:
            self.load_psf(path2psf)
        self.load_weight_map(path2weight, cutout_pix=cutout_pix)

    def update_psf(self, kernel):
        self._psf = kernel

    def get_cutout(self, image_full, header_full, cutout_pix):
        """

        :param image_full:
        :param header_full:
        :param cutout_pix:
        :return:
        """
        xw = cutout_pix
        yw = cutout_pix
        xc = self._ra_center
        yc = self._dec_center
        xmin, xmax, ymin, ymax, ra_coords, dec_coords = self._cutout_coords(header_full, xc, yc, xw, yw)
        img = self._get_cutout_image(image_full, xmin, xmax, ymin, ymax)
        head = self._get_cutout_header(header_full, xmin, xmax, ymin, ymax)
        return img, head, ra_coords, dec_coords


    @property
    def image(self):
        return self._image_data

    @property
    def weight_map(self):
        return self._weight_map

    @property
    def numPix(self):
        return self._numPix

    @property
    def deltaPix(self):
        return self._deltaPix

    @property
    def header(self):
        return self._head

    @property
    def header_full(self):
        return self._header_full

    @property
    def coords(self):
        return self._ra_coords, self._dec_coords

    @property
    def center(self):
        return self._ra_center, self._dec_center

    @property
    def sigma_bkg(self):
        return self._sigma_bkg

    @property
    def psf_kwargs(self):
        if not hasattr(self, '_psf_kwargs'):
            self._psf_kwargs = {}
            self._psf_kwargs['kernel'] = self._psf
            self._psf_kwargs['kernel_large'] = self._psf
            self._psf_kwargs['psf_type'] = 'pixel'
        return self._psf_kwargs

    def psf_kwargs_update(self):
        self._psf_kwargs = {}
        self._psf_kwargs['kernel'] = self._psf
        self._psf_kwargs['kernel_large'] = self._psf
        self._psf_kwargs['psf_type'] = 'pixel'

    def _cutout_coords(self, head, xc, yc, xw, yw):
        """
        Inputs:
            file  - pyfits HDUList (must be 2D)
            xc,yc - x and y coordinates in the fits files' coordinate system (CTYPE)
            xw,yw - x and y width (pixels or wcs)
            outfile - optional output file
        """
        wcs = pywcs.WCS(head)
        naxis1 = head['NAXIS1']
        naxis2 = head['NAXIS2']
        xx, yy = wcs.all_world2pix(xc, yc, 0)
        print('the center of the image is at pixel coordinates %f, %f.' % (xx, yy))
        xmin, xmax = np.max([0, xx-xw]), np.min([naxis1, xx+xw])
        ymin, ymax = np.max([0, yy-yw]), np.min([naxis2, yy+yw])

        if xmax < 0 or ymax < 0:
            raise ValueError("Max Coordinate is outside of map: %f,%f." % (xmax, ymax))
        if ymin >= head.get('NAXIS2') or xmin >= head.get('NAXIS1'):
            raise ValueError("Min Coordinate is outside of map: %f,%f." % (xmin, ymin))
        x_coords = np.linspace(xmin, xmax, xmax-xmin)
        y_coords = np.linspace(ymin, ymax, ymax-ymin)
        x_coords, y_coords = np.meshgrid(x_coords, y_coords)
        ra_coords, dec_coords = wcs.all_pix2world(x_coords, y_coords, 0)
        ra_coords = util.image2array((ra_coords - self._ra_center)*3600)
        dec_coords = util.image2array((dec_coords - self._dec_center)*3600)
        return xmin, xmax, ymin, ymax, ra_coords, dec_coords

    def _get_cutout_image(self, image_full, xmin, xmax, ymin, ymax):
        img = image_full[ymin:ymax, xmin:xmax].copy()
        return img

    def _get_cutout_header(self, header_full, xmin, xmax, ymin, ymax):
        head = self.change_header(header_full, xmin, xmax, ymin, ymax)
        return head

    def change_header(self, head, xmin, xmax, ymin, ymax):
        """
        changes the header to adjust information to the cutout image
        """
        head['CRPIX1'] -= xmin
        head['CRPIX2'] -= ymin
        head['NAXIS1'] = int(xmax-xmin)
        head['NAXIS2'] = int(ymax-ymin)
        if head.get('NAXIS1') == 0 or head.get('NAXIS2') == 0:
            raise ValueError("Map has a 0 dimension: %i,%i." % (head.get('NAXIS1'), head.get('NAXIS2')))
        return head
