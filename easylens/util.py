__author__ = 'sibirrer'

import numpy as np
import astropy.wcs as pywcs
import astropy.io.fits as pyfits


def array2image(array):
    """
    returns the information contained in a 1d array into an n*n 2d array (only works when lenght of array is n**2)

    :param array: image values
    :type array: array of size n**2
    :returns:  2d array
    :raises: AttributeError, KeyError
    """
    n=int(np.sqrt(len(array)))
    if n**2 != len(array):
        raise ValueError("lenght of input array given as %s is not square of integer number!" % (len(array)))
    image = array.reshape(n, n)
    return image


def image2array(image):
    """
    returns the information contained in a 2d array into an n*n 1d array

    :param array: image values
    :type array: array of size (n,n)
    :returns:  1d array
    :raises: AttributeError, KeyError
    """
    nx, ny = image.shape  # find the size of the array
    imgh = np.reshape(image, nx*ny)  # change the shape to be 1d
    return imgh


def make_grid(numPix, deltapix, subgrid_res=1):
    """
    returns x, y position information in two 1d arrays
    """
    numPix_eff = numPix*subgrid_res
    deltapix_eff = deltapix/float(subgrid_res)
    a = np.arange(numPix_eff)
    matrix = np.dstack(np.meshgrid(a, a)).reshape(-1, 2)
    x_grid = (matrix[:, 0] - numPix_eff/2.)*deltapix_eff
    y_grid = (matrix[:, 1] - numPix_eff/2.)*deltapix_eff
    shift = (subgrid_res-1)/(2.*subgrid_res)*deltapix
    return x_grid - shift, y_grid - shift


def averaging(grid, numGrid, numPix):
    """
    resize pixel grid with numGrid to numPix and averages over the pixels
    """
    Nbig = numGrid
    Nsmall = numPix
    small = grid.reshape([Nsmall, Nbig/Nsmall, Nsmall, Nbig/Nsmall]).mean(3).mean(1)
    return small


def make_subgrid(ra_coord, dec_coord, subgrid_res=2):
    """
    return a grid with subgrid resolution
    :param ra_coord:
    :param dec_coord:
    :param subgrid_res:
    :return:
    """
    if subgrid_res == 1:
        return ra_coord, dec_coord
    elif subgrid_res == 2:
        ra_array = array2image(ra_coord)
        dec_array = array2image(dec_coord)
        n = len(ra_array)
        d_ra_x = ra_array[0][1] - ra_array[0][0]
        d_ra_y = ra_array[1][0] - ra_array[0][0]
        d_dec_x = dec_array[0][1] - dec_array[0][0]
        d_dec_y = dec_array[1][0] - dec_array[0][0]

        ra_array_new = np.zeros((n*subgrid_res, n*subgrid_res))
        dec_array_new = np.zeros((n*subgrid_res, n*subgrid_res))
        ra_array_new[0::2, 0::2] = ra_array - d_ra_x/2. - d_ra_y/2.
        ra_array_new[1::2, 1::2] = ra_array + d_ra_x/2. + d_ra_y/2.
        ra_array_new[0::2, 1::2] = ra_array + d_ra_x/2. - d_ra_y/2.
        ra_array_new[1::2, 0::2] = ra_array - d_ra_x/2. + d_ra_y/2.

        dec_array_new[0::2, 0::2] = dec_array - d_dec_x/2. - d_dec_y/2.
        dec_array_new[1::2, 1::2] = dec_array + d_dec_x/2. + d_dec_y/2.
        dec_array_new[0::2, 1::2] = dec_array + d_dec_x/2. - d_dec_y/2.
        dec_array_new[1::2, 0::2] = dec_array - d_dec_x/2. + d_dec_y/2.
        ra_coords_sub = image2array(ra_array_new)
        dec_coords_sub = image2array(dec_array_new)
        return ra_coords_sub, dec_coords_sub
    else:
        raise ValueError('Subgridresolution with higher than 2 not implemented yet!')
    # TODO implement subgrid resolution higher than 2


def get_ra_dec_center(path2image):
    """
    returns the RA DEC coords of the center of an image
    :param path2image:
    :return:
    """
    image_data = pyfits.open(path2image)[0].data
    head = pyfits.open(path2image)[0].header
    numPix = len(image_data)
    wcs = pywcs.WCS(head)
    ra_center, dec_center = wcs.wcs_pix2world(numPix/2.,numPix/2.,0)
    return ra_center, dec_center


def get_mask_circular(ra_c, dec_c, r, ra_coords, dec_coords):
    """

    :param center: 2D coordinate of center position of circular mask
    :param r: radius of mask in pixel values
    :param data: data image
    :return:
    """
    ra_shift = ra_coords - ra_c
    dec_shift = dec_coords - dec_c
    R = np.sqrt(ra_shift*ra_shift + dec_shift*dec_shift)
    mask = np.empty_like(R)
    mask[R > r] = 0
    mask[R <= r] = 1
    return mask


def get_mask_square(ra_c, dec_c, d, ra_coords, dec_coords):
    """

    :param center: 2D coordinate of center position of circular mask
    :param r: radius of mask in pixel values
    :param data: data image
    :return:
    """
    ra_shift = ra_coords - ra_c
    dec_shift = dec_coords - dec_c
    mask = np.ones_like(ra_coords)
    mask[ra_shift > d] = 0
    mask[ra_shift < -d] = 0
    mask[dec_shift > d] = 0
    mask[dec_shift < -d] = 0
    return mask


def kernel_norm(kernel):
    """

    :param kernel:
    :return: normalisation of the psf kernel
    """
    norm = np.sum(np.array(kernel))
    kernel /= norm
    return kernel


def cut_edges(image, numPix):
    """
    cuts out the edges of a 2d image and returns re-sized image to numPix
    :param image: 2d numpy array
    :param numPix:
    :return:
    """
    nx, ny = image.shape
    if nx < numPix or ny < numPix:
        print('WARNING: image can not be resized.')
        return image
    dx = int((nx-numPix)/2)
    dy = int((ny-numPix)/2)
    resized = image[dx:nx-dx, dy:ny-dy]
    return resized


def rotate(x, y, center_x, center_y, phi_G):
    """
    rotate the coordinates
    :param x:
    :param y:
    :param center_x:
    :param center_y:
    :param phi_G:
    :return:
    """
    x_shift = x - center_x
    y_shift = y - center_y
    cos_phi = np.cos(phi_G)
    sin_phi = np.sin(phi_G)
    x1 = cos_phi*x_shift+sin_phi*y_shift
    x2 = -sin_phi*x_shift+cos_phi*y_shift
    return x1, x2


def ellipticity2phi_gamma(e1, e2):
    """
    :param e1:
    :param e2:
    :return:
    """
    phi = np.arctan2(e2, e1)/2
    gamma = np.sqrt(e1**2+e2**2)
    return phi, gamma


def cart2polar(x, y, center=np.array([0, 0])):
    """
	transforms cartesian coords [x,y] into polar coords [r,phi] in the frame of the lense center

	:param coord: set of coordinates
	:type coord: array of size (n,2)
	:param center: rotation point
	:type center: array of size (2)
	:returns:  array of same size with coords [r,phi]
	:raises: AttributeError, KeyError
	"""
    coordShift_x = x - center[0]
    coordShift_y = y - center[1]
    r = np.sqrt(coordShift_x**2+coordShift_y**2)
    phi = np.arctan2(coordShift_y, coordShift_x)
    return r, phi


def ellipticity2phi_q(e1,e2):
    """
    :param e1:
    :param e2:
    :return:
    """
    phi = np.arctan2(e2, e1)/2
    c = np.sqrt(e1**2+e2**2)
    q = (1-c)/(1+c)
    return phi, q