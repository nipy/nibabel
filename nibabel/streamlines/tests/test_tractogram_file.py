from nose.tools import assert_raises

from ..tractogram import Tractogram
from ..tractogram_file import TractogramFile


def test_subclassing_tractogram_file():

    # Missing 'save' method
    class DummyTractogramFile(TractogramFile):
        @classmethod
        def is_correct_format(cls, fileobj):
            return False

        @classmethod
        def load(cls, fileobj, lazy_load=True):
            return None

    assert_raises(TypeError, DummyTractogramFile, Tractogram())

    # Missing 'load' method
    class DummyTractogramFile(TractogramFile):
        @classmethod
        def is_correct_format(cls, fileobj):
            return False

        def save(self, fileobj):
            pass

    assert_raises(TypeError, DummyTractogramFile, Tractogram())


def test_tractogram_file():
    assert_raises(NotImplementedError, TractogramFile.is_correct_format, "")
    assert_raises(NotImplementedError, TractogramFile.load, "")

    # Testing calling the 'save' method of `TractogramFile` object.
    class DummyTractogramFile(TractogramFile):
        @classmethod
        def is_correct_format(cls, fileobj):
            return False

        @classmethod
        def load(cls, fileobj, lazy_load=True):
            return None

        @classmethod
        def save(self, fileobj):
            pass

    assert_raises(NotImplementedError,
                  super(DummyTractogramFile,
                        DummyTractogramFile(Tractogram)).save, "")
