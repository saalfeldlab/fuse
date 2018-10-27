import os

from gunpowder import ArrayKey, BatchRequest, Normalize, Pad, Crop, \
    build, Hdf5Write

RAW       = ArrayKey('RAW')
GT_LABELS = ArrayKey('GT_LABELS')


def run_augmentations(
        data_providers,
        keys_with_rois=(),
        augmentations=(),
        store_inputs_node=None,
        output_node=Hdf5Write(dataset_names={RAW:'volumes/raw'}, output_dir=os.path.join(os.path.expanduser("~"), "tmp"), output_filename="test-nodes.h5")):



    # TODO why is GT_AFFINITIES three-dimensional? compare to
    # TODO https://github.com/funkey/gunpowder/blob/master/examples/cremi/train.py#L35
    # specifiy which Arrays should be requested for each batch
    request = BatchRequest()
    for key, roi in keys_with_rois:
        request.add(key, roi.get_shape())
    total_roi = keys_with_rois[0][1]
    for k, r in keys_with_rois[1:]:
        print(total_roi, k, r)
        total_roi = total_roi.union(r)

    # create a tuple of data sources, one for each HDF file
    data_sources = tuple(_get_data_sources(provider, [k for (k, r) in keys_with_rois], total_roi) for provider in data_providers)

    pipeline = data_sources
    if store_inputs_node:
        pipeline += store_inputs_node
    for augmentation in augmentations:
        pipeline += augmentation
    pipeline += output_node

    with build(pipeline) as b:
        b.request_batch(request)

def _get_data_sources(provider, keys, total_roi):
    source = provider
    for k in keys:
        if k == RAW:
            source += Normalize(RAW)
        source += Pad(k, None)
        source += Crop(k, roi=total_roi)
    return source