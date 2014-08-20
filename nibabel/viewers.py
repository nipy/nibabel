""" Utilities for viewing images

Includes version of OrthoSlicer3D code by our own Paul Ivanov
"""
from __future__ import division, print_function

import numpy as np

from .optpkg import optional_package

plt, _, _ = optional_package('matplotlib.pyplot')
mpl_img, _, _ = optional_package('matplotlib.image')

# Assumes the following layout
#
# ^ +---------+   ^ +---------+
# | |         |   | |         |
#   |         |     |         |
# z |    2    |   z |    3    |
#   |         |     |         |
# | |         |   | |         |
# v +---------+   v +---------+
#   <--  x  -->     <--  y  -->
# ^ +---------+
# | |         |
#   |         |
# y |    1    |
#   |         |
# | |         |
# v +---------+
#   <--  x  -->

class OrthoSlicer3D(object):
    """Orthogonal-plane slicer.

    OrthoSlicer3d expects 3-dimensional data, and by default it creates a
    figure with 3 axes, one for each slice orientation.

    There are two modes, "following on" and "following off".  In "following on"
    mode, moving the mouse in any one axis will select out the corresponding
    slices in the other two.  The mode is "following off" when the figure is
    first created.  Clicking the left mouse button toggles mouse following and
    triggers a full redraw (to update the ticks, for example). Scrolling up and
    down moves the slice up and down in the current axis.

    Example
    -------
    import numpy as np
    a = np.sin(np.linspace(0,np.pi,20))
    b = np.sin(np.linspace(0,np.pi*5,20))
    data = np.outer(a,b)[..., np.newaxis]*a
    OrthoSlicer3D(data).show()
    """
    def __init__(self, data, axes=None, aspect_ratio=(1, 1, 1), cmap='gray',
                 pcnt_range=None):
        """
        Parameters
        ----------
        data : 3 dimensional ndarray
            The data that will be displayed by the slicer
        axes : None or length 3 sequence of mpl.Axes, optional
            3 axes instances for the X, Y, and Z slices, or None (default)
        aspect_ratio : float or length 3 sequence, optional
            stretch factors for X, Y, Z directions
        cmap : colormap identifier, optional
            String or cmap instance specifying colormap. Will be passed as
            ``cmap`` argument to ``plt.imshow``.
        pcnt_range : length 2 sequence, optional
            Percentile range over which to scale image for display. If None,
            scale between image mean and max.  If sequence, min and max
            percentile over which to scale image.
        """
        data_shape = np.array(data.shape[:3]) # allow trailing RGB dimension
        aspect_ratio = np.array(aspect_ratio)
        if axes is None: # make the axes
            # ^ +---------+   ^ +---------+
            # | |         |   | |         |
            #   |         |     |         |
            # z |    2    |   z |    3    |
            #   |         |     |         |
            # | |         |   | |         |
            # v +---------+   v +---------+
            #   <--  x  -->     <--  y  -->
            # ^ +---------+
            # | |         |
            #   |         |
            # y |    1    |
            #   |         |
            # | |         |
            # v +---------+
            #   <--  x  -->
            fig = plt.figure()
            x, y, z = data_shape * aspect_ratio
            maxw = float(x + y)
            maxh = float(y + z)
            yh = y / maxh
            xw = x / maxw
            yw = y / maxw
            zh = z / maxh
            # z slice (if usual transverse acquisition => axial slice)
            ax1 = fig.add_axes((0., 0., xw, yh))
            # y slice (usually coronal)
            ax2 = fig.add_axes((0,  yh, xw, zh))
            # x slice (usually sagittal)
            ax3 = fig.add_axes((xw, yh, yw, zh))
            axes = (ax1, ax2, ax3)
        else:
            if not np.all(aspect_ratio == 1):
                raise ValueError('Aspect ratio must be 1 for external axes')
            ax1, ax2, ax3 = axes

        self.data = data

        if pcnt_range is None:
            vmin, vmax = data.min(), data.max()
        else:
            vmin, vmax = np.percentile(data, pcnt_range)

        kw = dict(vmin=vmin,
                  vmax=vmax,
                  aspect='auto',
                  interpolation='nearest',
                  cmap=cmap,
                  origin='lower')
        # Start midway through each axis
        st_x, st_y, st_z = (data_shape - 1) / 2.
        n_x, n_y, n_z = data_shape
        z_get_slice = lambda i: self.data[:, :, min(i, n_z-1)].T
        y_get_slice = lambda i: self.data[:, min(i, n_y-1), :].T
        x_get_slice = lambda i: self.data[min(i, n_x-1), :, :].T
        im1 = ax1.imshow(z_get_slice(st_z), **kw)
        im2 = ax2.imshow(y_get_slice(st_y), **kw)
        im3 = ax3.imshow(x_get_slice(st_x), **kw)
        im1.get_slice, im2.get_slice, im3.get_slice = (
            z_get_slice, y_get_slice, x_get_slice)
        # idx is the current slice number for each panel
        im1.idx, im2.idx, im3.idx = st_z, st_y, st_x
        # set the maximum dimensions for indexing
        im1.size, im2.size, im3.size = n_z, n_y, n_x
        # setup pairwise connections between the slice dimensions
        im1.imx = im3 # x move in panel 1 (usually axial)
        im1.imy = im2 # y move in panel 1
        im2.imx = im3 # x move in panel 2 (usually coronal)
        im2.imy = im1
        im3.imx = im2 # x move in panel 3 (usually sagittal)
        im3.imy = im1

        self.follow = False
        self.figs = set([ax.figure for ax in axes])
        for fig in self.figs:
            fig.canvas.mpl_connect('button_press_event', self.on_click)
            fig.canvas.mpl_connect('scroll_event', self.on_scroll)
            fig.canvas.mpl_connect('motion_notify_event', self.on_mousemove)

    def show(self):
        """ Show the slicer; convenience for ``plt.show()``
        """
        plt.show()

    def _axis_artist(self, event):
        """ Return artist if within axes, and is an image, else None
        """
        if not getattr(event, 'inaxes'):
            return None
        artist = event.inaxes.images[0]
        return artist if isinstance(artist, mpl_img.AxesImage) else None

    def on_click(self, event):
        if event.button == 1:
            self.follow = not self.follow
            plt.draw()

    def on_scroll(self, event):
        assert event.button in ('up', 'down')
        im = self._axis_artist(event)
        if im is None:
            return
        im.idx += 1 if event.button == 'up' else -1
        im.idx %= im.size
        im.set_data(im.get_slice(im.idx))
        ax = im.axes
        ax.draw_artist(im)
        ax.figure.canvas.blit(ax.bbox)

    def on_mousemove(self, event):
        if not self.follow:
            return
        im = self._axis_artist(event)
        if im is None:
            return
        ax = im.axes
        imx, imy = im.imx, im.imy
        x, y = np.round((event.xdata, event.ydata)).astype(int)
        imx.set_data(imx.get_slice(x))
        imy.set_data(imy.get_slice(y))
        imx.idx = x
        imy.idx = y
        for i in imx, imy:
            ax = i.axes
            ax.draw_artist(i)
            ax.figure.canvas.blit(ax.bbox)


if __name__ == '__main__':
    a = np.sin(np.linspace(0,np.pi,20))
    b = np.sin(np.linspace(0,np.pi*5,20))
    data = np.outer(a,b)[..., np.newaxis]*a
    # all slices
    OrthoSlicer3D(data).show()

    # broken out into three separate figures
    f, ax1 = plt.subplots()
    f, ax2 = plt.subplots()
    f, ax3 = plt.subplots()
    OrthoSlicer3D(data, axes=(ax1, ax2, ax3)).show()
