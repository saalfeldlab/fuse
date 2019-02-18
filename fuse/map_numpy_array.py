from __future__ import division

import logging

import numpy as np

from gunpowder import BatchFilter

logger = logging.getLogger(__name__)


class MapNumpyArray(BatchFilter):
    def __init__(
            self,
            mapping,
            *keys):
        super(BatchFilter, self).__init__()
        self.keys    = keys
        self.mapping = mapping

    def setup(self):
        pass

    def prepare(self, request):
        pass

    def process(self, batch, request):

        for key in self.keys:
            if not key in request:
                logger.debug('Ignoring key %s: not in request %s', key, request)
                continue

            assert key in batch.arrays, 'Requested key %s not in batch arrays %s' % (key, batch)

            array = batch.arrays[key]
            logger.debug('Array data dtype for key %s before mapping: %s', key, array.data.dtype)
            mapped_data = self.mapping(array.data)
            assert mapped_data.shape == array.data.shape, 'Mapping must not change shape of data: Original shape is {}, mapped shape is {}'.format(array.data.shape, mapped_data.shape)
            array.data = mapped_data
            logger.debug('Array data dtype for key %s after mapping: %s', key, batch.arrays[key].data.dtype)
