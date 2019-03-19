import numpy as np

from gunpowder import BatchFilter

import logging
logger = logging.getLogger(__name__)


class MaskOutLabels(BatchFilter):

    def __init__(self, label_key, mask_key, ids_to_be_masked=(0,)):
        self.label_key        = label_key
        self.mask_key         = mask_key
        self.ids_to_be_masked = tuple(ids_to_be_masked)

    def prepare(self, request):
        assert self.label_key in request and self.mask_key in request, 'Need to request both label key %s and mask key %s' % (self.label_key, self.mask_key)
        assert request[self.label_key] == request[self.mask_key]

    def process(self, batch, request):

        assert self.label_key in request and self.mask_key in request, 'Need to request both label key %s and mask key %s' % (self.label_key, self.mask_key)
        assert request[self.label_key] == request[self.mask_key]

        assert self.label_key in batch and self.mask_key in batch, 'Need to request both label key %s and mask key %s' % (self.label_key, self.mask_key)
        assert batch[self.label_key].spec == batch[self.mask_key].spec

        mask = batch[self.mask_key]
        labels = batch[self.label_key]

        assert mask.data.shape == labels.data.shape, 'Incompatible shapes: %s -- %s' % (mask.data.shape, labels.data.shape)

        mask.data *= np.isin(labels.data, test_elements=self.ids_to_be_masked, invert=True, assume_unique=False)