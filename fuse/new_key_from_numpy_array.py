from __future__ import division

import logging

from gunpowder import Array, BatchFilter

logger = logging.getLogger(__name__)


class NewKeyFromNumpyArray(BatchFilter):
    def __init__(
            self,
            mapping,
            key,
            mapped_key
            ):
        super(BatchFilter, self).__init__()
        self.key           = key
        self.mapped_key    = mapped_key
        self.mapping       = mapping
        self.original_spec = None

    def setup(self):

        self.provides(self.mapped_key, self.spec[self.key].copy())

    def prepare(self, request):
        assert self.key in request and self.mapped_key in request, 'Need to request both key %s and mapped key %s' % (self.key, self.mapped_key)
        # need to update the upstream request and remember the original request for self.key
        # crop the arrays appropriately in self.process
        self.original_spec = request[self.key].copy()
        request[self.key].roi = request[self.key].roi.union(request[self.mapped_key].roi)

    def process(self, batch, request):

        if not self.mapped_key in request:
            logger.debug('Ignoring key %s: not in request %s', self.mapped_key, request)
            return

        assert self.key in batch.arrays, 'Requested key %s not in batch arrays %s' % (self.key, batch)

        array = batch.arrays[self.key]
        logger.debug('Array data dtype for key %s before mapping: %s', self.key, array.data.dtype)
        mapped_data = self.mapping(self._cutout(array, request[self.mapped_key].roi))

        mapped_array = Array(data=mapped_data, spec=array.spec.copy(), attrs=array.attrs)
        mapped_array.spec.roi = request[self.mapped_key].roi
        batch.arrays[self.mapped_key] = mapped_array

        cutout_data = self._cutout(array, self.original_spec.roi)
        array.data = cutout_data
        array.spec = self.original_spec
        request[self.key] = self.original_spec
        logger.debug('Array data dtype for key %s after mapping: %s', self.mapped_key, batch.arrays[self.mapped_key].data.dtype)

    def _cutout(self, array, requested_roi):
        in_local_coordinates = array.spec.roi / array.spec.voxel_size
        requested_roi_in_local_coordinates = requested_roi / array.spec.voxel_size
        offset = requested_roi_in_local_coordinates.get_begin() - in_local_coordinates.get_begin()
        shape  = requested_roi_in_local_coordinates.get_shape()
        slicing = [slice(None)] * array.data.ndim
        for i, (o, s) in enumerate(zip(offset, shape)):
            slicing[-3 + i] = slice(o, o + s)
        return array.data[slicing]


