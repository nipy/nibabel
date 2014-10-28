""" Utilities for viewing images

Includes version of OrthoSlicer3D code originally written by our own
Paul Ivanov.
"""
from __future__ import division, print_function

import numpy as np

from .optpkg import optional_package
from .orientations import aff2axcodes, axcodes2ornt

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
    >>> b = np.sin(np.linspace(0,np.pi*5,20))asa
    >>> data = np.outer(a,b)[..., np.newaxis]*a
    >>> OrthoSlicer3D(data).show()  # doctest: +SKIP
    """
    # Skip doctest above b/c not all systems have mpl installed
    def __init__(self, data, affine=None, axes=None, cmap='gray',
                 pcnt_range=(1., 99.), figsize=(8, 8)):
        """
        Parameters
        ----------
        data : ndarray
            The data that will be displayed by the slicer. Should have 3+
            dimensions.
        affine : array-like | None
            Affine transform for the data. This is used to determine
            how the data should be sliced for plotting into the saggital,
            coronal, and axial view axes. If None, identity is assumed.
            The aspect ratio of the data are inferred from the affine
            transform.
        axes : tuple of mpl.Axes | None, optional
            3 or 4 axes instances for the 3 slices plus volumes,
            or None (default).
        cmap : str | instance of cmap, optional
            String or cmap instance specifying colormap.
        pcnt_range : array-like, optional
            Percentile range over which to scale image for display.
        figsize : tuple
            Figure size (in inches) to use if axes are None.
        """
        data = np.asanyarray(data)
        if data.ndim < 3:
            raise ValueError('data must have at least 3 dimensions')
        affine = np.array(affine, float) if affine is not None else np.eye(4)
        if affine.ndim != 2 or affine.shape != (4, 4):
            raise ValueError('affine must be a 4x4 matrix')
        # determine our orientation
        self._affine = affine.copy()
        codes = axcodes2ornt(aff2axcodes(self._affine))
        order = np.argsort([c[0] for c in codes])
        flips = np.array([c[1] for c in codes])[order]
        self._order = dict(x=int(order[0]), y=int(order[1]), z=int(order[2]))
        self._flips = dict(x=flips[0], y=flips[1], z=flips[2])
        self._scalers = np.abs(self._affine).max(axis=0)[:3]
        self._inv_affine = np.linalg.inv(affine)
        # current volume info
        self._volume_dims = data.shape[3:]
        self._current_vol_data = data[:, :, :, 0] if data.ndim > 3 else data
        self._data = data
        vmin, vmax = np.percentile(data, pcnt_range)
        del data

        if axes is None:  # make the axes
            # ^ +---------+   ^ +---------+
            # | |         |   | |         |
            #   |   Sag   |     |   Cor   |
            # S |    1    |   S |    2    |
            #   |         |     |         |
            #   |         |     |         |
            #   +---------+     +---------+
            #        A  -->     <--  R
            # ^ +---------+     +---------+
            # | |         |     |         |
            #   |  Axial  |     |         |
            # A |    3    |     |    4    |
            #   |         |     |         |
            #   |         |     |         |
            #   +---------+     +---------+
            #   <--  R          <--  t  -->

            fig, axes = plt.subplots(2, 2)
            fig.set_size_inches(figsize, forward=True)
            self._axes = dict(x=axes[0, 0], y=axes[0, 1], z=axes[1, 0],
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
        self._vol = 0
        colors = dict()
        for k in 'xyz':
            size = self._data.shape[self._order[k]]
            self._idx[k] = size // 2
            self._ims[k] = self._axes[k].imshow(self._get_slice_data(k), **kw)
            self._sizes[k] = size
            colors[k] = (0, 1, 0)
        labels = dict(x='SAIP', y='SLIR', z='ALPR')

        # set up axis crosshairs
        self._crosshairs = dict()
        for type_, i_1, i_2 in zip('xyz', 'yxx', 'zzy'):
            ax, label = self._axes[type_], labels[type_]
            vert = ax.plot([self._idx[i_1]] * 2,
                           [-0.5, self._sizes[i_2] - 0.5],
                           color=colors[i_1], linestyle='-')[0]
            horiz = ax.plot([-0.5, self._sizes[i_1] - 0.5],
                            [self._idx[i_2]] * 2,
                            color=colors[i_2], linestyle='-')[0]
            self._crosshairs[type_] = dict(vert=vert, horiz=horiz)
            # add text labels (top, right, bottom, left)
            lims = [0, self._sizes[i_1], 0, self._sizes[i_2]]
            bump = 0.01
            poss = [[lims[1] / 2., lims[3]],
                    [(1 + bump) * lims[1], lims[3] / 2.],
                    [lims[1] / 2., 0],
                    [lims[0] - bump * lims[1], lims[3] / 2.]]
            anchors = [['center', 'bottom'], ['left', 'center'],
                       ['center', 'top'], ['right', 'center']]
            for pos, anchor, lab in zip(poss, anchors, label):
                ax.text(pos[0], pos[1], lab,
                        horizontalalignment=anchor[0],
                        verticalalignment=anchor[1])
            ax.axis(lims)
            # ax.set_aspect(aspect_ratio[type_])  # XXX FIX
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
            x=[self._crosshairs['z']['vert'].set_xdata,
               self._crosshairs['y']['vert'].set_xdata],
            y=[self._crosshairs['z']['horiz'].set_ydata,
               self._crosshairs['x']['vert'].set_xdata],
            z=[self._crosshairs['y']['horiz'].set_ydata,
               self._crosshairs['x']['horiz'].set_ydata])

        self._figs = set([a.figure for a in self._axes.values()])
        for fig in self._figs:
            fig.canvas.mpl_connect('scroll_event', self._on_scroll)
            fig.canvas.mpl_connect('motion_notify_event', self._on_mouse)
            fig.canvas.mpl_connect('button_press_event', self._on_mouse)
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

    def set_position(self, x=None, y=None, z=None, v=None):
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
        for key, val in zip('xyz', (x, y, z)):
            if val is not None:
                self._set_viewer_slice(key, val)
                draw = True
        if draw:
            self._update_voxel_levels()
            self._draw()

    def _get_voxel_levels(self):
        """Get levels of the current voxel as a function of volume"""
        # XXX THIS IS WRONG
        #y = self._data[self._idx['x'],
        #               self._idx['y'],
        #               self._idx['z'], :].ravel()
        y = self._data[0, 0, 0, :].ravel()
        y = np.concatenate((y, [y[-1]]))
        return y

    def _update_voxel_levels(self):
        """Update voxel levels in time plot"""
        if self.n_volumes > 1:
            self._volume_ax_objs['step'].set_ydata(self._get_voxel_levels())

    def _set_vol_idx(self, idx):
        """Change which volume is shown"""
        max_ = np.prod(self._volume_dims)
        self._vol = max(min(int(round(idx)), max_ - 1), 0)
        # Must reset what is shown
        self._current_vol_data = self._data[:, :, :, self._vol]
        for key in 'xyz':
            self._ims[key].set_data(self._get_slice_data(key))
        self._volume_ax_objs['patch'].set_x(self._vol - 0.5)

    def _get_slice_data(self, key):
        """Helper to get the current slice image"""
        assert key in ['x', 'y', 'z']
        data = np.take(self._current_vol_data, self._idx[key],
                       axis=self._order[key])
        # saggital: get to S/A
        # coronal: get to S/L
        # axial: get to A/L
        xaxes = dict(x='y', y='x', z='x')
        yaxes = dict(x='z', y='z', z='y')
        if self._order[xaxes[key]] < self._order[yaxes[key]]:
            data = data.T
        if self._flips[xaxes[key]]:
            data = data[:, ::-1]
        if self._flips[yaxes[key]]:
            data = data[::-1]
        return data

    def _set_viewer_slice(self, key, idx):
        """Helper to set a viewer slice number"""
        assert key in ['x', 'y', 'z']
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
        """Handle mpl scroll wheel event"""
        assert event.button in ('up', 'down')
        key = self._in_axis(event)
        if key is None:
            return
        delta = 10 if event.key is not None and 'control' in event.key else 1
        if event.key is not None and 'shift' in event.key:
            if self.n_volumes <= 1:
                return
            key = 'v'  # shift: change volume in any axis
        assert key in ['x', 'y', 'z', 'v']
        idx = self._idx[key] if key != 'v' else self._vol
        idx += delta if event.button == 'up' else -delta
        if key == 'v':
            self._set_vol_idx(idx)
        else:
            self._set_viewer_slice(key, idx)
        self._update_voxel_levels()
        self._draw()

    def _on_mouse(self, event):
        """Handle mpl mouse move and button press events"""
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
        """Handle mpl keypress events"""
        if event.key is not None and 'escape' in event.key:
            self.close()

    def _draw(self):
        """Update all four (or three) plots"""
        for key in 'xyz':
            ax, im = self._axes[key], self._ims[key]
            ax.draw_artist(im)
            for line in self._crosshairs[key].values():
                ax.draw_artist(line)
            ax.figure.canvas.blit(ax.bbox)
        if self.n_volumes > 1 and 'v' in self._axes:  # user might only pass 3
            ax = self._axes['v']
            ax.draw_artist(ax.patch)  # axis bgcolor to erase old lines
            for key in ('step', 'patch'):
                ax.draw_artist(self._volume_ax_objs[key])
            ax.figure.canvas.blit(ax.bbox)
