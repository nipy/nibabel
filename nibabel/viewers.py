""" Utilities for viewing images

Includes version of OrthoSlicer3D code originally written by our own
Paul Ivanov.
"""
from __future__ import division, print_function

import numpy as np
from functools import partial

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


def _set_viewer_slice(idx, im):
    """Helper to set a viewer slice number"""
    im.idx = idx
    im.set_data(im.get_slice(im.idx))
    for fun in im.cross_setters:
        fun([idx] * 2)


class OrthoSlicer3D(object):
    """Orthogonal-plane slicer.

    OrthoSlicer3d expects 3-dimensional data, and by default it creates a
    figure with 3 axes, one for each slice orientation.

    Clicking and dragging the mouse in any one axis will select out the
    corresponding slices in the other two. Scrolling up and
    down moves the slice up and down in the current axis.

    Example
    -------
    >>> import numpy as np
    >>> a = np.sin(np.linspace(0,np.pi,20))
    >>> b = np.sin(np.linspace(0,np.pi*5,20))
    >>> data = np.outer(a,b)[..., np.newaxis]*a
    >>> OrthoSlicer3D(data).show()  # doctest: +SKIP
    """
    # Skip doctest above b/c not all systems have mpl installed
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
        data_shape = np.array(data.shape[:3])  # allow trailing RGB dimension
        aspect_ratio = np.array(aspect_ratio, float)
        if axes is None:  # make the axes
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
        sts = (st_x, st_y, st_z)
        n_x, n_y, n_z = data_shape
        z_get_slice = lambda i: self.data[:, :, min(i, n_z-1)].T
        y_get_slice = lambda i: self.data[:, min(i, n_y-1), :].T
        x_get_slice = lambda i: self.data[min(i, n_x-1), :, :].T
        im1 = ax1.imshow(z_get_slice(st_z), **kw)
        im2 = ax2.imshow(y_get_slice(st_y), **kw)
        im3 = ax3.imshow(x_get_slice(st_x), **kw)
        im1.get_slice, im2.get_slice, im3.get_slice = (
            z_get_slice, y_get_slice, x_get_slice)
        self._ims = (im1, im2, im3)

        # idx is the current slice number for each panel
        im1.idx, im2.idx, im3.idx = st_z, st_y, st_x

        # set the maximum dimensions for indexing
        im1.size, im2.size, im3.size = n_z, n_y, n_x

        # set up axis crosshairs
        colors = ['r', 'g', 'b']
        for ax, im, idx_1, idx_2 in zip(axes, self._ims, [0, 0, 1], [1, 2, 2]):
            im.x_line = ax.plot([sts[idx_1]] * 2,
                                [-0.5, data.shape[idx_2] - 0.5],
                                color=colors[idx_1], linestyle='-',
                                alpha=0.25)[0]
            im.y_line = ax.plot([-0.5, data.shape[idx_1] - 0.5],
                                [sts[idx_2]] * 2,
                                color=colors[idx_2], linestyle='-',
                                alpha=0.25)[0]
            ax.axis('tight')
            ax.patch.set_visible(False)
            ax.set_frame_on(False)
            ax.axes.get_yaxis().set_visible(False)
            ax.axes.get_xaxis().set_visible(False)

        # monkey-patch some functions
        im1.set_viewer_slice = partial(_set_viewer_slice, im=im1)
        im2.set_viewer_slice = partial(_set_viewer_slice, im=im2)
        im3.set_viewer_slice = partial(_set_viewer_slice, im=im3)

        # setup pairwise connections between the slice dimensions
        im1.x_im = im3  # x move in panel 1 (usually axial)
        im1.y_im = im2  # y move in panel 1
        im2.x_im = im3  # x move in panel 2 (usually coronal)
        im2.y_im = im1  # y move in panel 2
        im3.x_im = im2  # x move in panel 3 (usually sagittal)
        im3.y_im = im1  # y move in panel 3

        # when an index changes, which crosshairs need to be updated
        im1.cross_setters = [im2.y_line.set_ydata, im3.y_line.set_ydata]
        im2.cross_setters = [im1.y_line.set_ydata, im3.x_line.set_xdata]
        im3.cross_setters = [im1.x_line.set_xdata, im2.x_line.set_xdata]

        self.figs = set([ax.figure for ax in axes])
        for fig in self.figs:
            fig.canvas.mpl_connect('scroll_event', self.on_scroll)
            fig.canvas.mpl_connect('motion_notify_event', self.on_mousemove)

    def show(self):
        """ Show the slicer; convenience for ``plt.show()``
        """
        plt.show()

    def close(self):
        """Close the viewer figures
        """
        for f in self.figs:
            plt.close(f)

    def _axis_artist(self, event):
        """Return artist if within axes, and is an image, else None
        """
        if not getattr(event, 'inaxes'):
            return None
        artist = event.inaxes.images[0]
        return artist if isinstance(artist, mpl_img.AxesImage) else None

    def on_scroll(self, event):
        assert event.button in ('up', 'down')
        im = self._axis_artist(event)
        if im is None:
            return
        idx = (im.idx + (1 if event.button == 'up' else -1))
        idx = max(min(idx, im.size - 1), 0)
        im.set_viewer_slice(idx)
        self._draw_ims()

    def on_mousemove(self, event):
        if event.button != 1:  # only enabled while dragging
            return
        im = self._axis_artist(event)
        if im is None:
            return
        x_im, y_im = im.x_im, im.y_im
        x, y = np.round((event.xdata, event.ydata)).astype(int)
        for i, idx in zip((x_im, y_im), (x, y)):
            i.set_viewer_slice(idx)
        self._draw_ims()

    def _draw_ims(self):
        for im in self._ims:
            ax = im.axes
            ax.draw_artist(im)
            ax.draw_artist(im.x_line)
            ax.draw_artist(im.y_line)
            ax.figure.canvas.blit(ax.bbox)
