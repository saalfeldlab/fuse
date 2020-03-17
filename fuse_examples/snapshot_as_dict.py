import copy
import logging

from gunpowder import Array, BatchFilter
from gunpowder.batch_request import BatchRequest

logger = logging.getLogger(__name__)

class SnapshotAsDict(BatchFilter):
    '''Save a passing batch in an HDF file.

    Args:

        every (``int``):

            How often to save a batch. ``every=1`` indicates that every batch
            will be stored, ``every=2`` every second and so on. By default,
            every batch will be stored.

        additional_request (:class:`BatchRequest`):

            An additional batch request to merge with the passing request, if a
            snapshot is to be made. If not given, only the arrays that are in
            the batch anyway are recorded. This is useful to request additional
            arrays like loss gradients for visualization that are otherwise not
            needed.

        '''

    def __init__(
            self,
            every=1,
            additional_request=None,
            ignore_key=lambda key: False):
        self.every = max(1,every)
        self.additional_request = BatchRequest() if additional_request is None else additional_request
        self.n = 0
        self.snapshots = {}
        self.ignore_key = ignore_key

    def prepare(self, request):

        self.record_snapshot = self.n%self.every == 0

        # append additional array requests, don't overwrite existing ones
        for array_key, spec in self.additional_request.array_specs.items():
            if array_key not in request.array_specs:
                request.array_specs[array_key] = spec

    def process(self, batch, _):

        if self.record_snapshot:

            snapshot_at = {}

            for (array_key, array) in batch.arrays.items():

                if self.ignore_key(array_key):
                    continue

                snapshot_at[array_key] = Array(data=array.data.copy(), spec=array.spec.copy(), attrs=copy.deepcopy(array.attrs))

            snapshot_at['loss'] = batch.loss
            self.snapshots[self.n] = snapshot_at

        self.n += 1

