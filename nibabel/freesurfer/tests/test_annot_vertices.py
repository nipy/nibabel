"""Annotation records address vertices, not positions in the file."""

import itertools
import struct

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from ..io import read_annot, write_annot

# Deliberately not sorted by packed RGB value.
_COLORS = [(1, 2, 3, 0), (255, 255, 255, 127), (1, 0, 0, 255)]
_VALUES = [197121, 16777215, 1]
_NAMES = [b'first', b'second', b'third']


def _annotation_bytes(records, version):
    """Pack wire-format records without using NiBabel's annotation writer."""

    def integer(value):
        return struct.pack('>i', value)

    def string(value):
        return integer(len(value) + 1) + value + b'\x00'

    payload = integer(len(records))
    payload += b''.join(integer(vertex) + integer(value) for vertex, value in records)
    payload += integer(1)
    if version == 1:
        payload += integer(len(_COLORS)) + string(b'NOFILE')
    else:
        payload += integer(-2) + integer(len(_COLORS))
        payload += string(b'NOFILE') + integer(len(_COLORS))
    for index, (name, color) in enumerate(zip(_NAMES, _COLORS)):
        if version == 2:
            payload += integer(index)
        payload += string(name) + b''.join(integer(channel) for channel in color)
    return payload


def _expected_labels(records, orig_ids):
    # Sequential Python integer assignments implement the reference reader,
    # independently of NumPy indexing and duplicate-selection algorithms.
    values = [0] * len(records)
    for vertex, value in records:
        values[vertex] = value
    if orig_ids:
        return values
    return [-1 if value == 0 else _VALUES.index(value) for value in values]


@pytest.mark.parametrize('version', [1, 2])
@pytest.mark.parametrize('orig_ids', [False, True])
@pytest.mark.parametrize('order', list(itertools.permutations(range(4))))
def test_annotation_vertex_order(tmp_path, version, orig_ids, order):
    records = list(enumerate([*_VALUES, 0]))
    records = [records[index] for index in order]
    path = tmp_path / 'permuted.annot'
    path.write_bytes(_annotation_bytes(records, version))
    labels, ctab, names = read_annot(path, orig_ids=orig_ids)
    assert_array_equal(labels, _expected_labels(records, orig_ids))
    assert labels.dtype == np.dtype('>i4')
    assert_array_equal(ctab, [(*color, value) for color, value in zip(_COLORS, _VALUES)])
    assert names == _NAMES


@pytest.mark.parametrize('version', [1, 2])
@pytest.mark.parametrize('orig_ids', [False, True])
@pytest.mark.parametrize('vertices', [[4, 0, 4, 2, 4, 0], [3] * 6, [0, 0, 1, 1, 2, 2]])
def test_annotation_repeated_and_omitted_vertices(tmp_path, version, orig_ids, vertices):
    # Include a final zero to check that removing an annotation also overwrites.
    records = list(zip(vertices, [*_VALUES, *_VALUES[:2], 0]))
    path = tmp_path / 'repeated.annot'
    path.write_bytes(_annotation_bytes(records, version))
    labels, _, _ = read_annot(path, orig_ids=orig_ids)
    assert_array_equal(labels, _expected_labels(records, orig_ids))


@pytest.mark.parametrize('version', [1, 2])
@pytest.mark.parametrize('orig_ids', [False, True])
@pytest.mark.parametrize('records', [[], [(0, 0)], [(0, _VALUES[1])]])
def test_annotation_empty_and_single_vertex(tmp_path, version, orig_ids, records):
    path = tmp_path / 'small.annot'
    path.write_bytes(_annotation_bytes(records, version))
    labels, _, _ = read_annot(path, orig_ids=orig_ids)
    assert_array_equal(labels, _expected_labels(records, orig_ids))


@pytest.mark.parametrize('version', [1, 2])
@pytest.mark.parametrize('orig_ids', [False, True])
@pytest.mark.parametrize('vertex', [-1, -2147483648, 2, 2147483647])
def test_annotation_vertex_out_of_range(tmp_path, version, orig_ids, vertex):
    records = [(0, _VALUES[0]), (vertex, _VALUES[1])]
    path = tmp_path / 'invalid.annot'
    path.write_bytes(_annotation_bytes(records, version))
    with pytest.raises(ValueError, match=r'vertex.*out of range'):
        read_annot(path, orig_ids=orig_ids)


@pytest.mark.parametrize('payload_bytes', [0, 4, 8, 12, 15])
def test_annotation_truncated_records(tmp_path, payload_bytes):
    path = tmp_path / 'truncated.annot'
    records = struct.pack('>4i', 1, _VALUES[0], 0, _VALUES[1])
    path.write_bytes(struct.pack('>i', 2) + records[:payload_bytes])
    with pytest.raises(ValueError, match='reshape'):
        read_annot(path)


@pytest.mark.parametrize('version', [1, 2])
def test_annotation_rewrite_preserves_vertex_labels(tmp_path, version):
    records = [(2, _VALUES[0]), (0, _VALUES[1]), (3, 0), (1, _VALUES[2])]
    path = tmp_path / 'source.annot'
    path.write_bytes(_annotation_bytes(records, version))
    labels, ctab, names = read_annot(path)
    output = tmp_path / 'rewritten.annot'
    write_annot(output, labels, ctab, names)
    reread, _, _ = read_annot(output)
    assert_array_equal(reread, _expected_labels(records, False))
