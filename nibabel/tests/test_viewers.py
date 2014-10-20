# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import numpy as np
from collections import namedtuple as nt

from ..optpkg import optional_package
from ..viewers import OrthoSlicer3D

from numpy.testing.decorators import skipif

from nose.tools import assert_raises

plt, has_mpl = optional_package('matplotlib.pyplot')[:2]
needs_mpl = skipif(not has_mpl, 'These tests need matplotlib')


@needs_mpl
def test_viewer():
    # Test viewer
    a = np.sin(np.linspace(0, np.pi, 20))
    b = np.sin(np.linspace(0, np.pi*5, 30))
    data = np.outer(a, b)[..., np.newaxis] * a
    viewer = OrthoSlicer3D(data)
    plt.draw()

    # fake some events
    viewer.on_scroll(nt('event', 'button inaxes')('up', None))  # outside axes
    viewer.on_scroll(nt('event', 'button inaxes')('up', plt.gca()))  # in axes
    # "click" outside axes, then once in each axis, then move without click
    viewer.on_mousemove(nt('event', 'xdata ydata inaxes button')(0.5, 0.5,
                                                                 None, 1))
    for im in viewer._ims:
        viewer.on_mousemove(nt('event', 'xdata ydata inaxes button')(0.5, 0.5,
                                                                     im.axes,
                                                                     1))
    viewer.on_mousemove(nt('event', 'xdata ydata inaxes button')(0.5, 0.5,
                                                                 None, None))
    viewer.set_indices(0, 1, 2)
    viewer.close()

    # other cases
    fig, axes = plt.subplots(1, 3)
    plt.close(fig)
    OrthoSlicer3D(data, pcnt_range=[0.1, 0.9], axes=axes)
    assert_raises(ValueError, OrthoSlicer3D, data, aspect_ratio=[1, 2, 3],
                  axes=axes)
