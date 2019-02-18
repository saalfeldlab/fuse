from __future__ import division

import logging

import numpy as np

from gunpowder import BatchFilter

logger = logging.getLogger(__name__)


class NumpyRequire(BatchFilter):
    def __init__(
            self,
            *keys,
            dtype=None,
            requirements=None):
        super(BatchFilter, self).__init__()
        self.keys         = keys
        self.dtype        = dtype
        self.requirements = requirements

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
            logger.debug('Array data dtype for key %s before requiring: %s', key, array.data.dtype)
            array.data = np.require(array.data, dtype=self.dtype, requirements=self.requirements)
            logger.debug('Array data dtype for key %s after requiring: %s', key, batch.arrays[key].data.dtype)
