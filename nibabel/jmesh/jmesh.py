# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
# General JMesh Input - Output to and from the filesystem
# Qianqian Fang <q.fang at neu.edu>
##############

__all__ = ['JMesh', 'read', 'write', 'default_header']

from jdata import load as jdload, save as jdsave
import numpy as np
from ..filebasedimages import FileBasedImage

default_header = {
    'JMeshVersion': '0.5',
    'Comment': 'Created by NiPy with NeuroJSON JMesh specification',
    'AnnotationFormat': 'https://neurojson.org/jmesh/draft2',
    'Parser': {
        'Python': ['https://pypi.org/project/jdata', 'https://pypi.org/project/bjdata'],
        'MATLAB': ['https://github.com/NeuroJSON/jnifty', 'https://github.com/NeuroJSON/jsonlab'],
        'JavaScript': 'https://github.com/NeuroJSON/jsdata',
        'CPP': 'https://github.com/NeuroJSON/json',
        'C': 'https://github.com/NeuroJSON/ubj',
    },
}


class JMesh(FileBasedImage):
    """JMesh: a simple data structure representing a brain surface

    * Description - JMesh defines a set of language-neutral JSON annotations for
          storage and exchange of mesh-related data. The details of the specification
          can be found in NeuroJSON's website at https://neurojson.org

    * Child Elements: [NA]
    * Text Content: [NA]

    Attributes
    ----------
    info: a dict
        A dict object storing the metadata (`_DataInfo_`) section of the JMesh
        file
    node : 2-D list or numpy array
        A 2-D numpy.ndarray object to store the vertices of the mesh
    nodelabel : 1-D list or numpy array
        A 1-D numpy.ndarray object to store the label of each vertex
    face : 2-D list or numpy array
        A 2-D numpy.ndarray object to store the triangular elements of the
        mesh; indices start from 1
    facelabel : 1-D list or numpy array
        A 1-D numpy.ndarray object to store the label of each triangle
    raw : a dict
        The raw data loaded from the .jmsh or .bmsh file
    """

    valid_exts = ('.jmsh', '.bmsh')
    files_types = (('image', '.jmsh'), ('image', '.bmsh'))
    makeable = False
    rw = True

    def __init__(self, info=None, node=None, nodelabel=None, face=None, facelabel=None):

        self.raw = {}
        if info is not None:
            self.raw['_DataInfo_'] = info

        if nodelabel is not None:
            self.raw['MeshVertex3'] = {'Data': node, 'Properties': {'Tag': nodelabel}}
            self.node = self.raw['MeshVertex3']['Data']
            self.nodelabel = self.raw['MeshVertex3']['Properties']['Tag']
        else:
            self.raw['MeshVertex3'] = node
            self.node = self.raw['MeshVertex3']

        if facelabel is not None:
            self.raw['MeshTri3'] = {'Data': face, 'Properties': {'Tag': facelabel}}
            self.face = self.raw['MeshTri3']['Data']
            self.facelabel = self.raw['MeshTri3']['Properties']['Tag']
        else:
            self.raw['MeshTri3'] = face
            self.face = self.raw['MeshTri3']

    @classmethod
    def from_filename(self, filename, opt={}, **kwargs):
        self = read(filename, opt, **kwargs)
        return self

    @classmethod
    def to_filename(self, filename, opt={}, **kwargs):
        write(self, filename, opt, **kwargs)


def read(filename, opt={}, **kwargs):
    """Load a JSON or binary JData (BJData) based JMesh file

    Parameters
    ----------
    filename : string
        The JMesh file to open, it has usually ending .gii
    opt: a dict that may contain below option keys
        ndarray: boolean, if True, node/face/nodelabel/facelabel are converted
                 to numpy.ndarray, otherwise, leave those unchanged
    kwargs: additional keyword arguments for `json.load` when .jmsh file is being loaded

    Returns
    -------
    mesh : a JMesh object
        Return a JMesh object containing mesh data fields such as node, face, nodelabel etc
    """
    opt.setdefault('ndarray', True)

    mesh = JMesh
    mesh.raw = jdload(filename, opt, **kwargs)

    # --------------------------------------------------
    # read metadata as `info`
    # --------------------------------------------------
    if '_DataInfo_' in mesh.raw:
        mesh.info = mesh.raw['_DataInfo_']

    # --------------------------------------------------
    # read vertices as `node` and `nodelabel`
    # --------------------------------------------------
    if 'MeshVertex3' in mesh.raw:
        mesh.node = mesh.raw['MeshVertex3']
    elif 'MeshNode' in mesh.raw:
        mesh.node = mesh.raw['MeshNode']
    else:
        raise Exception('JMesh', 'JMesh surface must contain node (MeshVertex3 or MeshNode)')

    if isinstance(mesh.node, dict):
        if ('Properties' in mesh.node) and ('Tag' in mesh.node['Properties']):
            mesh.nodelabel = mesh.node['Properties']['Tag']
        if 'Data' in mesh.node:
            mesh.node = mesh.node['Data']
    if isinstance(mesh.node, np.ndarray) and mesh.node.ndim == 2 and mesh.node.shape[1] > 3:
        mesh.nodelabel = mesh.node[:, 3:]
        mesh.node = mesh.node[:, 0:3]

    # --------------------------------------------------
    # read triangles as `face` and `facelabel`
    # --------------------------------------------------
    if 'MeshTri3' in mesh.raw:
        mesh.face = mesh.raw['MeshTri3']
    elif 'MeshSurf' in mesh.raw:
        mesh.face = mesh.raw['MeshSurf']

    if isinstance(mesh.face, dict):
        if ('Properties' in mesh.face) and ('Tag' in mesh.face['Properties']):
            mesh.facelabel = mesh.face['Properties']['Tag']
        if 'Data' in mesh.face:
            mesh.face = mesh.face['Data']
    if isinstance(mesh.face, np.ndarray) and mesh.face.ndim == 2 and mesh.face.shape[1] > 3:
        mesh.facelabel = mesh.face[:, 3:]
        mesh.face = mesh.face[:, 0:3]

    # --------------------------------------------------
    # convert to numpy ndarray
    # --------------------------------------------------
    if opt['ndarray']:
        if (
            hasattr(mesh, 'node')
            and (mesh.node is not None)
            and (not isinstance(mesh.node, np.ndarray))
        ):
            mesh.node = np.array(mesh.node)

        if (
            hasattr(mesh, 'face')
            and (mesh.face is not None)
            and (not isinstance(mesh.face, np.ndarray))
        ):
            mesh.face = np.array(mesh.face)

        if (
            hasattr(mesh, 'nodelabel')
            and (mesh.nodelabel is not None)
            and (not isinstance(mesh.nodelabel, np.ndarray))
        ):
            mesh.nodelabel = np.array(mesh.nodelabel)

        if (
            hasattr(mesh, 'facelabel')
            and (mesh.facelabel is not None)
            and (not isinstance(mesh.facelabel, np.ndarray))
        ):
            mesh.facelabel = np.array(mesh.facelabel)

    return mesh


def write(mesh, filename, opt={}, **kwargs):
    """Save the current mesh to a new file

    Parameters
    ----------
    mesh : a JMesh object
    filename : string
        Filename to store the JMesh file (.jmsh for JSON based JMesh and
        .bmsh for binary JMesh files)
    opt: a dict that may contain below option keys
        ndarray: boolean, if True, node/face/nodelabel/facelabel are converted
                 to numpy.ndarray, otherwise, leave those unchanged
    kwargs: additional keyword arguments for `json.dump` when .jmsh file is being saved

    Returns
    -------
    None

    We update the mesh related data fields `MeshVetex3`, `MeshTri3` and metadata `_DataInfo_`
    from mesh.node, mesh.face and mesh.info, then save mesh.raw to JData files
    """

    if not hasattr(mesh, 'raw') or mesh.raw is None:
        mesh.raw = {}

    if hasattr(mesh, 'info') and mesh.info is not None:
        mesh.raw['_DataInfo_'] = mesh.info
    if hasattr(mesh, 'node') and mesh.node is not None:
        if hasattr(mesh, 'facelabel') and mesh.nodelabel is not None:
            mesh.raw['MeshVertex3'] = {'Data': mesh.node, 'Properties': {'Tag': mesh.nodelabel}}
        else:
            mesh.raw['MeshVertex3'] = mesh.node

    if hasattr(mesh, 'info') and mesh.face is not None:
        if hasattr(mesh, 'facelabel') and mesh.facelabel is not None:
            mesh.raw['MeshTri3'] = {'Data': mesh.face, 'Properties': {'Tag': mesh.facelabel}}
        else:
            mesh.raw['MeshTri3'] = mesh.face

    return jdsave(mesh.raw, filename, opt, **kwargs)


load = read
save = write
