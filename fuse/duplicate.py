import logging

import copy

from gunpowder import BatchFilter
from gunpowder.array import ArrayKey, Array

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
            logger.debug("Set spec for key %s to be the same as for key %s: %s", value, key, spec)

    def prepare(self, request):
        pass

    def process(self, batch, request):

        for old_key, new_key in self.to_duplicate.items():

            if old_key not in request:
                continue

            assert new_key not in batch.arrays, "key {} already present in batch".format(new_key)
            assert isinstance(new_key, ArrayKey), "Can only duplicate array data"

            array_ = batch.arrays[old_key]
            array  = Array(array_.data.copy(), spec=array_.spec, attrs=copy.deepcopy(array_.attrs))
            batch.arrays[new_key] = array

