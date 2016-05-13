__author__ = 'sibirrer'

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid, make_axes_locatable

import easylens.util as util


class ShowLens(object):
    """
    class to plot the lens system
    """
    def __init__(self, lensSystem):
        """

        :param lensDES: lensDES class
        :return:
        """
        self.lensSystem = lensSystem

    def show_images(self, ra_pos=None, dec_pos=None, kwargs_mask=None):
        """
        plots all the available frames
        :return:
        """
        if ra_pos is not None and dec_pos is not None:
            plot_point = True
        else:
            plot_point = False
        f, axes = plt.subplots(1, self.lensSystem.num_frames, figsize=(5*self.lensSystem.num_frames, 5), sharex=False, sharey=False)
        for i in range(self.lensSystem.num_frames):
            ax = axes[i]
            frame = self.lensSystem.available_frames[i]
            image = self.lensSystem.get_image(frame)
            deltaPix = self.lensSystem.get_deltaPix(frame)
            numPix = self.lensSystem.get_numPix(frame)
            mask = self.lensSystem.get_mask_frame(kwargs_mask, frame)
            try:
                mask = util.array2image(mask)
            except:
                mask = 1
            im = ax.matshow(np.log10(image*mask), origin='lower')#, extent=[0, deltaPix*numPix, 0, deltaPix*numPix])
            ax.autoscale(False)
            if plot_point:
                x_pos, y_pos = self.lensSystem.get_pixel_coord(frame, ra_pos, dec_pos)
                #ax.plot(x_pos*deltaPix, y_pos*deltaPix, 'go')
                ax.plot(x_pos, y_pos, 'go')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            delta = 5./deltaPix
            ax.plot([1, 1+delta], [1, 1], linewidth=3, color='k')
            ax.plot([1, 1], [1, 1+delta], linewidth=3, color='k')
            ax.text(2, 2, '5"', fontsize=25)
            ax.text(0.1, 0.85,  frame,
                verticalalignment='bottom', horizontalalignment='left',
                transform=ax.transAxes, fontsize=30)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)

        f.tight_layout()
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=+0.25, hspace=0.05)
        return f, axes

    def show_list(self, model_list, frame_list):
        """
        plots all the available frames
        :return:
        """
        num_frames = len(frame_list)
        f, axes = plt.subplots(1, num_frames, figsize=(5*num_frames, 5), sharex=False, sharey=False)
        for i in range(num_frames):
            if num_frames == 1:
                ax = axes
            else:
                ax = axes[i]
            frame = frame_list[i]
            image = util.array2image(model_list[i])
            deltaPix = self.lensSystem.get_deltaPix(frame)
            numPix = self.lensSystem.get_numPix(frame)
            im = ax.matshow(image, origin='lower', extent=[0, deltaPix*numPix, 0, deltaPix*numPix])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.autoscale(False)
            ax.plot([1, 6], [1, 1], linewidth=3, color='k')
            ax.plot([1, 1], [1, 6], linewidth=3, color='k')
            ax.text(2, 2, '5"', fontsize=25)
            ax.text(0.1, 0.85,  frame,
                verticalalignment='bottom', horizontalalignment='left',
                transform=ax.transAxes, fontsize=30)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)

        f.tight_layout()
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=+0.25, hspace=0.05)
        return f, axes

    def show_psf(self):
        """
        plots the psf models for all the frames
        :return:
        """
        f, axes = plt.subplots(1, self.lensSystem.num_frames, figsize=(5*self.lensSystem.num_frames, 5), sharex=False, sharey=False)
        for i in range(self.lensSystem.num_frames):
            if self.lensSystem.num_frames == 1:
                ax = axes
            else:
                ax = axes[i]
            frame = self.lensSystem.available_frames[i]
            psf = self.lensSystem.get_psf_kwargs(frame)['kernel']

            im = ax.matshow(psf, origin='lower')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            ax.text(0.1, 0.85,  frame,
                verticalalignment='bottom', horizontalalignment='left',
                transform=ax.transAxes, fontsize=30)
        f.tight_layout()
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=+0.25, hspace=0.05)
        return f, axes

    def show_pixel_hist(self, bins=100):
        """

        :return: pixel histograms of all bands
        """
        f, axes = plt.subplots(1, self.lensSystem.num_frames, figsize=(5*self.lensSystem.num_frames, 5), sharex=False, sharey=False)
        for i in range(self.lensSystem.num_frames):
            ax = axes[i]
            frame = self.lensSystem.available_frames[i]
            image = util.image2array(self.lensSystem.get_image(frame))
            deltaPix = self.lensSystem.get_deltaPix(frame)
            numPix = self.lensSystem.get_numPix(frame)
            ax.hist(image, bins=bins, normed=False)

            ax.text(0.1, 0.85,  frame,
                verticalalignment='bottom', horizontalalignment='left',
                transform=ax.transAxes, fontsize=30)
        f.tight_layout()
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=+0.25, hspace=0.05)
        return f, axes