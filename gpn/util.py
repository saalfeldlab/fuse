from gpn.duplicate import Duplicate
from gunpowder import ArrayKey, BatchRequest, build, RandomLocation, RandomProvider, ArraySpec

RAW       = ArrayKey('RAW')
GT_LABELS = ArrayKey('GT_LABELS')

def run_augmentations(
        data_providers,
        roi,
        keys_with_sizes=(),
        augmentations=(),
        duplicate_key_mapping=lambda key: '{}-original'.format(key.identifier)):

    request = BatchRequest()
    duplicates = {}
    for key, size in keys_with_sizes:
        request[key] = ArraySpec(roi(key))
        duplicates[key] = ArrayKey(duplicate_key_mapping(key))

    data_sources = tuple(provider + RandomLocation() for provider in data_providers)

    pipeline = data_sources + RandomProvider() + Duplicate(duplicates)

    for augmentation in augmentations:
        pipeline += augmentation

    with build(pipeline) as b:
        return b.request_batch(request)
