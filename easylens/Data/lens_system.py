__author__ = 'sibirrer'

import numpy as np
import astropy.wcs as pywcs

import easylens.util as util


class LensSystem(object):
    """
    data class for handling the different exposures/bands of an object
    """
    def __init__(self, name, ra, dec):
        self.name = name
        self.available_frames = []
        self.ra = ra
        self.dec = dec

    def add_info_attribute(self, attrname, info_data, replace=False):
        """
        creates an attribute to store particular information.

        """
        attset = ['name', 'ra', 'dec']
        assert attrname in attset, '%s cannot be used. Supported names are: %s' % (attrname, attset)

        if replace or (not hasattr(self, attrname)):
            setattr(self, attrname, info_data)
        elif not getattr(self, attrname) == info_data:
            raise TypeError("The number of images is already set to %f" % getattr(self, attrname))

    def add_image_data(self, imagedata, attrname, replace=False):
        """
        simplest possible version of _add_image_data. Here we simply create an
        attribute for imagedata instance. *Can be replaced by subclass*
        """
        attset = ['g_band', 'r_band', 'i_band', 'Y_band', 'z_band']
        assert attrname in attset, '%s cannot be used. Supported names are: %s' % (attrname, attset)
        if not hasattr(self, attrname):
            self.available_frames.append(attrname)
        if replace or (not hasattr(self, attrname)):
            setattr(self, attrname, imagedata)

    def get_image(self, attrname):
        image_data_obj = getattr(self, attrname)
        return image_data_obj.image

    def get_weight_map(self, attrname):
        image_data_obj = getattr(self, attrname)
        return image_data_obj.weight_map

    def get_numPix(self, attrname):
        image_data_obj = getattr(self, attrname)
        return image_data_obj.numPix

    def get_deltaPix(self, attrname):
        image_data_obj = getattr(self, attrname)
        return image_data_obj.deltaPix

    def get_coords(self, attrname):
        image_data_obj = getattr(self, attrname)
        cos_dec = np.cos(self.dec/360*2*np.pi)
        ra_coords, dec_coords = image_data_obj.coords
        return ra_coords*cos_dec, dec_coords

    def get_center(self, attrname):
        image_data_obj = getattr(self, attrname)
        return image_data_obj.center

    def get_psf_kwargs(self, attrname):
        image_data_obj = getattr(self, attrname)
        return image_data_obj.psf_kwargs

    def update_psf_kwargs(self, attrname, kernel):
        image_data_obj = getattr(self, attrname)
        image_data_obj.update_psf(kernel)
        image_data_obj.psf_kwargs_update()

    def get_sigma_bkg(self, attrname):
        image_data_obj = getattr(self, attrname)
        return image_data_obj.sigma_bkg

    def get_pixel_coord(self, attrname, ra_pos, dec_pos, relative=True):
        image_data_obj = getattr(self, attrname)
        head = image_data_obj.header
        wcs = pywcs.WCS(head)
        ra_c, dec_c = self.get_center(attrname)
        if relative is True:
            cos_dec = np.cos(dec_c/360.*2*np.pi)
            x_pos, y_pos = wcs.wcs_world2pix(ra_pos/cos_dec/3600. + ra_c, dec_pos/3600. + dec_c, 0)
        else:
            x_pos, y_pos = wcs.wcs_world2pix(ra_pos/3600. + ra_c, dec_pos/3600. + dec_c, 0)
        return x_pos, y_pos

    def get_sed_estimate(self, ra_pos, dec_pos):
        """
        returns an estimate of an SED at a given position
        :param ra_pos:
        :param dec_pos:
        :return:
        """
        w_SED = {}
        num_frames = self.num_frames
        for i in range(num_frames):
            frame = self.available_frames[i]
            x_pos, y_pos = self.get_pixel_coord(frame, ra_pos, dec_pos)
            image = self.get_image(frame)
            x = int(x_pos)
            y = int(y_pos)
            w_SED[frame] = np.average(image[y-8:y+9, x-8:x+9])
        return w_SED

    def get_mask(self, kwargs_mask):
        """

        :return:
        """
        mask = {}
        num_pix = 0
        num_frames = self.num_frames
        for i in range(num_frames):
            frame = self.available_frames[i]
            mask_f = self.get_mask_frame(kwargs_mask, frame)
            num_pix += np.sum(mask_f)
            mask[frame] = mask_f
        return mask, num_pix

    def get_mask_frame(self, kwargs_mask, frame):
        """
        returns a pixel grid with 1 or 0 in each pixel depending on the mask
        :param kwargs_mask:
        :param frame:
        :return:
        """
        if kwargs_mask is None:
            return 1
        else:
            ra_coords, dec_coords = self.get_coords(frame)
            if kwargs_mask["type"] == "circular":
                ra_c = kwargs_mask["ra_c"]
                dec_c = kwargs_mask["dec_c"]
                r = kwargs_mask["radius"]
                mask = util.get_mask_circular(ra_c, dec_c, r, ra_coords, dec_coords)
            elif kwargs_mask["type"] == "square":
                ra_c = kwargs_mask["ra_c"]
                dec_c = kwargs_mask["dec_c"]
                d = kwargs_mask["width"]
                mask = util.get_mask_square(ra_c, dec_c, d, ra_coords, dec_coords)
            elif kwargs_mask["type"] == "circular_hole":
                ra_c = kwargs_mask["ra_c"]
                dec_c = kwargs_mask["dec_c"]
                r = kwargs_mask["radius"]
                mask = util.get_mask_circular(ra_c, dec_c, r, ra_coords, dec_coords)
                ra_c = kwargs_mask["ra_c_hole"]
                dec_c = kwargs_mask["dec_c_hole"]
                r = kwargs_mask["r_hole"]
                mask_hole = (1 - util.get_mask_circular(ra_c, dec_c, r, ra_coords, dec_coords))
                mask *= mask_hole
            else:
                raise ValueError("Mask option %s is not valid!" % kwargs_mask["option"])
            return mask

    @property
    def num_frames(self):
        return len(self.available_frames)


