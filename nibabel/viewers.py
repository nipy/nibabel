""" Utilities for viewing images

Includes version of OrthoSlicer3D code originally written by our own
Paul Ivanov.
"""
from __future__ import division, print_function

import numpy as np
import weakref

from .optpkg import optional_package
from .orientations import aff2axcodes, axcodes2ornt


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
        # Nest imports so that matplotlib.use() has the appropriate
        # effect in testing
        plt, _, _ = optional_package('matplotlib.pyplot')
        mpl_img, _, _ = optional_package('matplotlib.image')
        mpl_patch, _, _ = optional_package('matplotlib.patches')

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
        flips = np.array([c[1] < 0 for c in codes])[order]
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
            #   |  Axial  |     |   Vol   |
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

        # Start midway through each axis, idx is current slice number
        self._ims, self._sizes, self._data_idx = dict(), dict(), dict()

        # set up axis crosshairs
        self._crosshairs = dict()
        r = [self._scalers[self._order['z']] / self._scalers[self._order['y']],
             self._scalers[self._order['z']] / self._scalers[self._order['x']],
             self._scalers[self._order['y']] / self._scalers[self._order['x']]]
        for k in 'xyz':
            self._sizes[k] = self._data.shape[self._order[k]]
        for k, xax, yax, ratio, label in zip('xyz', 'yxx', 'zzy', r,
                                             ('SAIP', 'SLIR', 'ALPR')):
            ax = self._axes[k]
            d = np.zeros((self._sizes[yax], self._sizes[xax]))
            self._ims[k] = self._axes[k].imshow(d, vmin=vmin, vmax=vmax,
                                                aspect=1, cmap=cmap,
                                                interpolation='nearest',
                                                origin='lower')
            vert = ax.plot([0] * 2, [-0.5, self._sizes[yax] - 0.5],
                           color=(0, 1, 0), linestyle='-')[0]
            horiz = ax.plot([-0.5, self._sizes[xax] - 0.5], [0] * 2,
                            color=(0, 1, 0), linestyle='-')[0]
            self._crosshairs[k] = dict(vert=vert, horiz=horiz)
            # add text labels (top, right, bottom, left)
            lims = [0, self._sizes[xax], 0, self._sizes[yax]]
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
            ax.set_aspect(ratio)
            ax.patch.set_visible(False)
            ax.set_frame_on(False)
            ax.axes.get_yaxis().set_visible(False)
            ax.axes.get_xaxis().set_visible(False)
            self._data_idx[k] = 0
        self._data_idx['v'] = -1

        # Set up volumes axis
        if self.n_volumes > 1 and 'v' in self._axes:
            ax = self._axes['v']
            ax.set_axis_bgcolor('k')
            ax.set_title('Volumes')
            y = np.zeros(self.n_volumes + 1)
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

        self._figs = set([a.figure for a in self._axes.values()])
        for fig in self._figs:
            fig.canvas.mpl_connect('scroll_event', self._on_scroll)
            fig.canvas.mpl_connect('motion_notify_event', self._on_mouse)
            fig.canvas.mpl_connect('button_press_event', self._on_mouse)
            fig.canvas.mpl_connect('key_press_event', self._on_keypress)

        # actually set data meaningfully
        self._position = np.zeros(4)
        self._position[3] = 1.  # convenience for affine multn
        self._changing = False  # keep track of status to avoid loops
        self._links = []  # other viewers this one is linked to
        plt.draw()
        for fig in self._figs:
            fig.canvas.draw()
        self._set_volume_index(0, update_slices=False)
        self._set_position(0., 0., 0.)
        self._draw()

    # User-level functions ###################################################
    def show(self):
        """Show the slicer in blocking mode; convenience for ``plt.show()``
        """
        plt, _, _ = optional_package('matplotlib.pyplot')
        plt.show()

    def close(self):
        """Close the viewer figures
        """
        plt, _, _ = optional_package('matplotlib.pyplot')
        for f in self._figs:
            plt.close(f)
        for link in self._links:
            link()._unlink(self)

    @property
    def n_volumes(self):
        """Number of volumes in the data"""
        return int(np.prod(self._volume_dims))

    @property
    def position(self):
        """The current coordinates"""
        return self._position[:3].copy()

    def link_to(self, other):
        """Link positional changes between two canvases

        Parameters
        ----------
        other : instance of OrthoSlicer3D
            Other viewer to use to link movements.
        """
        if not isinstance(other, self.__class__):
            raise TypeError('other must be an instance of %s, not %s'
                            % (self.__class__.__name__, type(other)))
        self._link(other, is_primary=True)

    def _link(self, other, is_primary):
        """Link a viewer"""
        ref = weakref.ref(other)
        if ref in self._links:
            return
        self._links.append(ref)
        if is_primary:
            other._link(self, is_primary=False)
            other.set_position(*self.position)

    def _unlink(self, other):
        """Unlink a viewer"""
        ref = weakref.ref(other)
        if ref in self._links:
            self._links.pop(self._links.index(ref))
            ref()._unlink(self)

    def _notify_links(self):
        """Notify linked canvases of a position change"""
        for link in self._links:
            link().set_position(*self.position[:3])

    def set_position(self, x=None, y=None, z=None):
        """Set current displayed slice indices

        Parameters
        ----------
        x : float | None
            X coordinate to use. If None, do not change.
        y : float | None
            Y coordinate to use. If None, do not change.
        z : float | None
            Z coordinate to use. If None, do not change.
        """
        self._set_position(x, y, z)
        self._draw()

    def set_volume_idx(self, v):
        """Set current displayed volume index

        Parameters
        ----------
        v : int
            Volume index.
        """
        self._set_volume_index(v)
        self._draw()

    def _set_volume_index(self, v, update_slices=True):
        """Set the plot data using a volume index"""
        v = self._data_idx['v'] if v is None else int(round(v))
        if v == self._data_idx['v']:
            return
        max_ = np.prod(self._volume_dims)
        self._data_idx['v'] = max(min(int(round(v)), max_ - 1), 0)
        idx = (slice(None), slice(None), slice(None))
        if self._data.ndim > 3:
            idx = idx + tuple(np.unravel_index(self._data_idx['v'],
                                               self._volume_dims))
        self._current_vol_data = self._data[idx]
        # update all of our slice plots
        if update_slices:
            self._set_position(None, None, None, notify=False)

    def _set_position(self, x, y, z, notify=True):
        """Set the plot data using a physical position"""
        # deal with volume first
        if self._changing:
            return
        self._changing = True
        x = self._position[0] if x is None else float(x)
        y = self._position[1] if y is None else float(y)
        z = self._position[2] if z is None else float(z)

        # deal with slicing appropriately
        self._position[:3] = [x, y, z]
        idxs = np.dot(self._inv_affine, self._position)[:3]
        for key, idx in zip('xyz', idxs):
            self._data_idx[key] = max(min(int(round(idx)),
                                      self._sizes[key] - 1), 0)
        for key in 'xyz':
            # saggital: get to S/A
            # coronal: get to S/L
            # axial: get to A/L
            data = np.take(self._current_vol_data, self._data_idx[key],
                           axis=self._order[key])
            xax = dict(x='y', y='x', z='x')[key]
            yax = dict(x='z', y='z', z='y')[key]
            if self._order[xax] < self._order[yax]:
                data = data.T
            if self._flips[xax]:
                data = data[:, ::-1]
            if self._flips[yax]:
                data = data[::-1]
            self._ims[key].set_data(data)
            # deal with crosshairs
            loc = self._data_idx[key]
            if self._flips[key]:
                loc = self._sizes[key] - loc
            loc = [loc] * 2
            if key == 'x':
                self._crosshairs['z']['vert'].set_xdata(loc)
                self._crosshairs['y']['vert'].set_xdata(loc)
            elif key == 'y':
                self._crosshairs['z']['horiz'].set_ydata(loc)
                self._crosshairs['x']['vert'].set_xdata(loc)
            else:  # key == 'z'
                self._crosshairs['y']['horiz'].set_ydata(loc)
                self._crosshairs['x']['horiz'].set_ydata(loc)

        # Update volume trace
        if self.n_volumes > 1 and 'v' in self._axes:
            idx = [0] * 3
            for key in 'xyz':
                idx[self._order[key]] = self._data_idx[key]
            vdata = self._data[idx[0], idx[1], idx[2], :].ravel()
            vdata = np.concatenate((vdata, [vdata[-1]]))
            self._volume_ax_objs['patch'].set_x(self._data_idx['v'] - 0.5)
            self._volume_ax_objs['step'].set_ydata(vdata)
        if notify:
            self._notify_links()
        self._changing = False

    # Matplotlib handlers ####################################################
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
        if event.key is not None and 'shift' in event.key:
            if self.n_volumes <= 1:
                return
            key = 'v'  # shift: change volume in any axis
        assert key in ['x', 'y', 'z', 'v']
        dv = 10. if event.key is not None and 'control' in event.key else 1.
        dv *= 1. if event.button == 'up' else -1.
        dv *= -1 if self._flips.get(key, False) else 1
        val = self._data_idx[key] + dv
        if key == 'v':
            self._set_volume_index(val)
        else:
            coords = {key: val}
            for k in 'xyz':
                if k not in coords:
                    coords[k] = self._data_idx[k]
            coords = np.array([coords['x'], coords['y'], coords['z'], 1.])
            coords = np.dot(self._affine, coords)[:3]
            self._set_position(coords[0], coords[1], coords[2])
        self._draw()

    def _on_mouse(self, event):
        """Handle mpl mouse move and button press events"""
        if event.button != 1:  # only enabled while dragging
            return
        key = self._in_axis(event)
        if key is None:
            return
        if key == 'v':
            # volume plot directly translates
            self._set_volume_index(event.xdata)
        else:
            # translate click xdata/ydata to physical position
            xax, yax = dict(x='yz', y='xz', z='xy')[key]
            x, y = event.xdata, event.ydata
            x = self._sizes[xax] - x if self._flips[xax] else x
            y = self._sizes[yax] - y if self._flips[yax] else y
            idxs = {xax: x, yax: y, key: self._data_idx[key]}
            idxs = np.array([idxs['x'], idxs['y'], idxs['z'], 1.])
            pos = np.dot(self._affine, idxs)[:3]
            self._set_position(*pos)
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
