
class StreamlineFile:
    @staticmethod
    def get_magic_number():
        raise NotImplementedError()

    @staticmethod
    def is_correct_format(cls, fileobj):
        raise NotImplementedError()

    def get_header(self):
        raise NotImplementedError()

    def get_streamlines(self, as_generator=False):
        raise NotImplementedError()

    def get_scalars(self, as_generator=False):
        raise NotImplementedError()

    def get_properties(self, as_generator=False):
        raise NotImplementedError()

    @classmethod
    def load(cls, fileobj):
        raise NotImplementedError()

    def save(self, filename):
        raise NotImplementedError()

    def __iter__(self):
        raise NotImplementedError()


class DynamicStreamlineFile(StreamlineFile):
    def append(self, streamlines):
        raise NotImplementedError()

    def __iadd__(self, streamlines):
        return self.append(streamlines)       