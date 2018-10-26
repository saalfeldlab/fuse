import os

from gunpowder import ArrayKey, BatchRequest, Normalize, Pad, Crop, \
    build, Hdf5Write

RAW       = ArrayKey('RAW')
GT_LABELS = ArrayKey('GT_LABELS')


def run_augmentations(
        data_providers,
        input_roi,
        augmentations=(),
        output_node=Hdf5Write(dataset_names={RAW:'volumes/raw'}, output_dir=os.path.join(os.path.expanduser("~"), "tmp"), output_filename="test-nodes.h5")):



    # TODO why is GT_AFFINITIES three-dimensional? compare to
    # TODO https://github.com/funkey/gunpowder/blob/master/examples/cremi/train.py#L35
    # specifiy which Arrays should be requested for each batch
    request = BatchRequest()
    print(input_roi, type(input_roi))
    request.add(RAW, input_roi.get_shape())

    # create a tuple of data sources, one for each HDF file
    data_sources = tuple(
        provider +
        Normalize(RAW) + # ensures RAW is in float in [0, 1]
        Pad(RAW, None) +
        Crop(RAW, roi=input_roi)
        for provider in data_providers)

    pipeline = data_sources
    for augmentation in augmentations:
        pipeline += augmentation
    pipeline += output_node

    with build(pipeline) as b:
        b.request_batch(request)