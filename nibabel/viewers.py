""" Utilities for viewing images

Includes version of OrthoSlicer3D code originally written by our own
Paul Ivanov.
"""
from __future__ import division, print_function

import numpy as np

from .optpkg import optional_package

plt, _, _ = optional_package('matplotlib.pyplot')
mpl_img, _, _ = optional_package('matplotlib.image')
mpl_patch, _, _ = optional_package('matplotlib.patches')


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
                 pcnt_range=(1., 99.), figsize=(8, 8)):
        """
        Parameters
        ----------
        data : ndarray
            The data that will be displayed by the slicer. Should have 3+
            dimensions.
        axes : tuple of mpl.Axes | None, optional
            3 or 4 axes instances for the X, Y, Z slices plus volumes,
            or None (default).
        aspect_ratio : array-like, optional
            Stretch factors for X, Y, Z directions.
        cmap : str | instance of cmap, optional
            String or cmap instance specifying colormap. Will be passed as
            ``cmap`` argument to ``plt.imshow``.
        pcnt_range : array-like, optional
            Percentile range over which to scale image for display. If None,
            scale between image mean and max.  If sequence, min and max
            percentile over which to scale image.
        figsize : tuple
            Figure size (in inches) to use if axes are None.
        """
        ar = np.array(aspect_ratio, float)
        if ar.shape != (3,) or np.any(ar <= 0):
            raise ValueError('aspect ratio must have exactly 3 elements >= 0')
        aspect_ratio = dict(x=ar[0], y=ar[1], z=ar[2])
        data = np.asanyarray(data)
        if data.ndim < 3:
            raise ValueError('data must have at least 3 dimensions')
        self._volume_dims = data.shape[3:]
        self._current_vol_data = data[:, :, :, 0] if data.ndim > 3 else data
        self._data = data
        pcnt_range = (0, 100) if pcnt_range is None else pcnt_range
        vmin, vmax = np.percentile(data, pcnt_range)
        del data

        if axes is None:  # make the axes
            # ^ +---------+   ^ +---------+
            # | |         |   | |         |
            #   |         |     |         |
            # z |    2    |   z |    3    |
            #   |         |     |         |
            # | |         |   | |         |
            # v +---------+   v +---------+
            #   <--  x  -->     <--  y  -->
            # ^ +---------+   ^ +---------+
            # | |         |   | |         |
            #   |         |     |         |
            # y |    1    |   A |    4    |
            #   |         |     |         |
            # | |         |   | |         |
            # v +---------+   v +---------+
            #   <--  x  -->     <--  t  -->

            fig, axes = plt.subplots(2, 2)
            fig.set_size_inches(figsize, forward=True)
            self._axes = dict(x=axes[0, 1], y=axes[0, 0], z=axes[1, 0],
                              v=axes[1, 1])
            plt.tight_layout(pad=0.1)
            if self.n_volumes <= 1:
                fig.delaxes(self._axes['v'])
                del self._axes['v']
        else:
            self._axes = dict(z=axes[0], y=axes[1], x=axes[2])
            if len(axes) > 3:
                self._axes['v'] = axes[3]

        kw = dict(vmin=vmin, vmax=vmax, aspect=1, interpolation='nearest',
                  cmap=cmap, origin='lower')

        # Start midway through each axis, idx is current slice number
        self._ims, self._sizes, self._idx = dict(), dict(), dict()
        colors = dict()
        for k, size in zip('xyz', self._data.shape[:3]):
            self._idx[k] = size // 2
            self._ims[k] = self._axes[k].imshow(self._get_slice_data(k), **kw)
            self._sizes[k] = size
            colors[k] = (0, 1, 0)
        self._idx['v'] = 0
        labels = dict(z='ILSR', y='ALPR', x='AIPS')

        # set up axis crosshairs
        for type_, i_1, i_2 in zip('zyx', 'xxy', 'yzz'):
            ax = self._axes[type_]
            im = self._ims[type_]
            label = labels[type_]
            # add slice lines
            im.vert_line = ax.plot([self._idx[i_1]] * 2,
                                   [-0.5, self._sizes[i_2] - 0.5],
                                   color=colors[i_1], linestyle='-')[0]
            im.horiz_line = ax.plot([-0.5, self._sizes[i_1] - 0.5],
                                    [self._idx[i_2]] * 2,
                                    color=colors[i_2], linestyle='-')[0]
            # add text labels (top, right, bottom, left)
            lims = [0, self._sizes[i_1], 0, self._sizes[i_2]]
            bump = 0.01
            poss = [[lims[1] / 2., lims[3]],
                    [(1 + bump) * lims[1], lims[3] / 2.],
                    [lims[1] / 2., 0],
                    [lims[0] - bump * lims[1], lims[3] / 2.]]
            anchors = [['center', 'bottom'], ['left', 'center'],
                       ['center', 'top'], ['right', 'center']]
            im.texts = [ax.text(pos[0], pos[1], lab,
                                horizontalalignment=anchor[0],
                                verticalalignment=anchor[1])
                        for pos, anchor, lab in zip(poss, anchors, label)]
            ax.axis(lims)
            ax.set_aspect(aspect_ratio[type_])
            ax.patch.set_visible(False)
            ax.set_frame_on(False)
            ax.axes.get_yaxis().set_visible(False)
            ax.axes.get_xaxis().set_visible(False)

        # Set up volumes axis
        if self.n_volumes > 1 and 'v' in self._axes:
            ax = self._axes['v']
            ax.set_axis_bgcolor('k')
            ax.set_title('Volumes')
            y = self._get_voxel_levels()
            x = np.arange(self.n_volumes + 1) - 0.5
            step = ax.step(x, y, where='post', color='y')[0]
            ax.set_xticks(np.unique(np.linspace(0, self.n_volumes - 1,
                                                5).astype(int)))
            ax.set_xlim(x[0], x[-1])
            yl = [self._data.min(), self._data.max()]
            yl = [l + s * np.diff(lims)[0] for l, s in zip(yl, [-1.01, 1.01])]
            patch = mpl_patch.Rectangle([-0.5, yl[0]], 1., np.diff(yl)[0],
                                        fill=True, facecolor=(0, 1, 0),
                                        edgecolor=(0, 1, 0), alpha=0.25)
            ax.add_patch(patch)
            ax.set_ylim(yl)
            self._volume_ax_objs = dict(step=step, patch=patch)

        # setup pairwise connections between the slice dimensions
        self._click_update_keys = dict(x='yz', y='xz', z='xy')

        # when an index changes, which crosshairs need to be updated
        self._cross_setters = dict(
            x=[self._ims['z'].vert_line.set_xdata,
               self._ims['y'].vert_line.set_xdata],
            y=[self._ims['z'].horiz_line.set_ydata,
               self._ims['x'].vert_line.set_xdata],
            z=[self._ims['y'].horiz_line.set_ydata,
               self._ims['x'].horiz_line.set_ydata])

        self._figs = set([a.figure for a in self._axes.values()])
        for fig in self._figs:
            fig.canvas.mpl_connect('scroll_event', self._on_scroll)
            fig.canvas.mpl_connect('motion_notify_event', self._on_mousemove)
            fig.canvas.mpl_connect('button_press_event', self._on_mousemove)
            fig.canvas.mpl_connect('key_press_event', self._on_keypress)

    def show(self):
        """ Show the slicer in blocking mode; convenience for ``plt.show()``
        """
        plt.show()

    def close(self):
        """Close the viewer figures
        """
        for f in self._figs:
            plt.close(f)

    @property
    def n_volumes(self):
        """Number of volumes in the data"""
        return int(np.prod(self._volume_dims))

    def set_indices(self, x=None, y=None, z=None, v=None):
        """Set current displayed slice indices

        Parameters
        ----------
        x : int | None
            Index to use. If None, do not change.
        y : int | None
            Index to use. If None, do not change.
        z : int | None
            Index to use. If None, do not change.
        v : int | None
            Volume index to use. If None, do not change.
        """
        x = int(x) if x is not None else None
        y = int(y) if y is not None else None
        z = int(z) if z is not None else None
        v = int(v) if v is not None else None
        draw = False
        if v is not None:
            if self.n_volumes <= 1:
                raise ValueError('cannot change volume index of single-volume '
                                 'image')
            self._set_vol_idx(v)
            draw = True
        for key, val in zip('zyx', (z, y, x)):
            if val is not None:
                self._set_viewer_slice(key, val)
                draw = True
        if draw:
            self._update_voxel_levels()
            self._draw()

    def _get_voxel_levels(self):
        """Get levels of the current voxel as a function of volume"""
        y = self._data[self._idx['x'],
                       self._idx['y'],
                       self._idx['z'], :].ravel()
        y = np.concatenate((y, [y[-1]]))
        return y

    def _update_voxel_levels(self):
        """Update voxel levels in time plot"""
        if self.n_volumes > 1:
            self._volume_ax_objs['step'].set_ydata(self._get_voxel_levels())

    def _set_vol_idx(self, idx):
        """Change which volume is shown"""
        max_ = np.prod(self._volume_dims)
        self._idx['v'] = max(min(int(round(idx)), max_ - 1), 0)
        # Must reset what is shown
        self._current_vol_data = self._data[:, :, :, self._idx['v']]
        for key in 'xyz':
            self._ims[key].set_data(self._get_slice_data(key))
        self._volume_ax_objs['patch'].set_x(self._idx['v'] - 0.5)

    def _get_slice_data(self, key):
        """Helper to get the current slice image"""
        ii = dict(x=0, y=1, z=2)[key]
        return np.take(self._current_vol_data, self._idx[key], axis=ii).T

    def _set_viewer_slice(self, key, idx):
        """Helper to set a viewer slice number"""
        self._idx[key] = max(min(int(round(idx)), self._sizes[key] - 1), 0)
        self._ims[key].set_data(self._get_slice_data(key))
        for fun in self._cross_setters[key]:
            fun([self._idx[key]] * 2)

    def _in_axis(self, event):
        """Return axis key if within one of our axes, else None"""
        if getattr(event, 'inaxes') is None:
            return None
        for key, ax in self._axes.items():
            if event.inaxes is ax:
                return key

    def _on_scroll(self, event):
        assert event.button in ('up', 'down')
        key = self._in_axis(event)
        if key is None:
            return
        delta = 10 if event.key is not None and 'control' in event.key else 1
        if event.key is not None and 'shift' in event.key:
            if self.n_volumes <= 1:
                return
            key = 'v'  # shift: change volume in any axis
        idx = self._idx[key] + (delta if event.button == 'up' else -delta)
        if key == 'v':
            self._set_vol_idx(idx)
        else:
            self._set_viewer_slice(key, idx)
        self._update_voxel_levels()
        self._draw()

    def _on_mousemove(self, event):
        if event.button != 1:  # only enabled while dragging
            return
        key = self._in_axis(event)
        if key is None:
            return
        if key == 'v':
            self._set_vol_idx(event.xdata)
        else:
            for sub_key, idx in zip(self._click_update_keys[key],
                                    (event.xdata, event.ydata)):
                self._set_viewer_slice(sub_key, idx)
        self._update_voxel_levels()
        self._draw()

    def _on_keypress(self, event):
        if event.key is not None and 'escape' in event.key:
            self.close()

    def _draw(self):
        for im in self._ims.values():
            ax = im.axes
            ax.draw_artist(im)
            ax.draw_artist(im.vert_line)
            ax.draw_artist(im.horiz_line)
            ax.figure.canvas.blit(ax.bbox)
            for t in im.texts:
                ax.draw_artist(t)
        if self.n_volumes > 1 and 'v' in self._axes:  # user might only pass 3
            ax = self._axes['v']
            ax.draw_artist(ax.patch)  # axis bgcolor to erase old lines
            for key in ('step', 'patch'):
                ax.draw_artist(self._volume_ax_objs[key])
            ax.figure.canvas.blit(ax.bbox)
