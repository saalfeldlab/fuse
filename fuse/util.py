from fuse.snapshot_as_dict import SnapshotAsDict
from gunpowder import ArrayKey, BatchRequest, build, RandomProvider, ArraySpec, logging

RAW       = ArrayKey('RAW')
GT_LABELS = ArrayKey('GT_LABELS')

logger = logging.getLogger(__name__)

def run_augmentations(
        data_providers,
        roi,
        keys=(),
        augmentations=(),
        voxel_size=lambda key: None):

    request = BatchRequest()
    for key in keys:
        request[key] = ArraySpec(roi(key).snap_to_grid(voxel_size(key)), voxel_size=voxel_size(key))

    logger.debug('Requesting batch with request %s', request)

    data_sources = tuple(provider for provider in data_providers)

    snapshot = SnapshotAsDict()

    pipeline = data_sources + RandomProvider() + snapshot

    for augmentation in augmentations:
        pipeline += augmentation

    with build(pipeline) as b:
        logging.info("submitting request %s", request)
        batch = b.request_batch(request)

    logger.debug("Got snapshots from request %s: %s", request, snapshot.snapshots)
    return batch, snapshot.snapshots[0]
