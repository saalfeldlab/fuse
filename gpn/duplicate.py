import logging

from gunpowder import BatchFilter
from gunpowder.array import ArrayKey

logger = logging.getLogger(__name__)

class Duplicate(BatchFilter):
    '''Add a constant intensity padding around arrays of another batch
    provider. This is useful if your requested batches can be larger than what
    your source provides.

    Args:

        kwargs: dictionary mapping arrays to be duplicated to new array keys
    '''

    def __init__(self, to_duplicate):

        self.to_duplicate = {}
        self.to_duplicate.update(to_duplicate)

    def setup(self):

        for key, value in self.to_duplicate.items():
            assert key in self.spec, ("Asked to duplicate %s, but is not provided upstream."%key)
            assert value not in self.spec, ("Asked to duplicate %s, but target %s is already in upstream."%(key, value))

            spec = self.spec[key].copy()
            self.spec[value] = spec

    def prepare(self, request):
        pass

    def process(self, batch, request):

        for _, key in self.to_duplicate.items():

            if key not in request:
                continue

            assert isinstance(key, ArrayKey), "Can only duplicate array data"

            array = batch.arrays[key]
            array.data = batch.arrays[_].data.copy()
            array.spec.roi = request[_].roi.copy()

