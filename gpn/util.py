import os

from gunpowder import ArrayKey, BatchRequest, Normalize, Pad, Crop, \
    build, Hdf5Write, RandomLocation, RandomProvider

RAW       = ArrayKey('RAW')
GT_LABELS = ArrayKey('GT_LABELS')


def run_augmentations(
        data_providers,
        roi,
        keys_with_sizes=(),
        augmentations=(),
        store_inputs_node=None,
        output_node=Hdf5Write(dataset_names={RAW:'volumes/raw'}, output_dir=os.path.join(os.path.expanduser("~"), "tmp"), output_filename="test-nodes.h5")):



    request = BatchRequest()
    for key, size in keys_with_sizes:
        request.add(key, roi(key).get_shape())

    data_sources = tuple(provider + RandomLocation() for provider in data_providers)

    pipeline = data_sources + RandomProvider()
    if store_inputs_node:
        pipeline += store_inputs_node
    for augmentation in augmentations:
        pipeline += augmentation
    pipeline += output_node
    print(request)
    print(pipeline)

    with build(pipeline) as b:
        b.request_batch(request)