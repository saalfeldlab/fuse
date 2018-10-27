import glob
import os

import gpn.util
from gunpowder import Hdf5Source, Roi, Hdf5Write, SimpleAugment

import numpy as np
import time

import jnius_config

RAW       = gpn.util.RAW
GT_LABELS = gpn.util.GT_LABELS

import argparse
import h5py
parser = argparse.ArgumentParser()
parser.add_argument("--max-heap-size", default="80g")
args = parser.parse_args()

data_providers = []
defect_dir = "/groups/saalfeld/saalfeldlab/larissa/data/cremi-2017/"
# data_dir = "/groups/saalfeld/home/hanslovskyp/data/cremi/training"
data_dir = '/groups/saalfeld/home/hanslovskyp/experiments/quasi-isotropic/data'
# for data in glob.glob(os.path.join(data_dir, '*.h5')):
file_pattern = 'sample_A_padded_20160501-2-additional-sections-fixed-offset.h5'
for data in glob.glob(os.path.join(data_dir, file_pattern)):
    print(data)
    # TODO add masks later on
    h5_source = Hdf5Source(
        data,
        datasets={
            RAW: 'volumes/raw',
            GT_LABELS: 'volumes/labels/neuron_ids-downsampled',
            # ArrayKeys.GT_MASK: 'volumes/masks/groundtruth',
            # ArrayKeys.TRAINING_MASK: 'volumes/masks/training'
        },
        array_specs={
            # ArrayKeys.GT_MASK: ArraySpec(interpolatable=False)
        }
    )
    data_providers.append(h5_source)

input_resolution = (360, 36, 36)

input_roi = Roi(offset=(0, 0, 0), shape=(7200, 720, 720))
output_dir = output_dir=os.path.join(os.path.expanduser("~"), "tmp")
output_filename = "test-nodes.h5"
dataset_names = {RAW: 'volumes/raw-augmented'}
input_dataset_names = {RAW: 'volumes/raw'}

output_node = Hdf5Write(
    dataset_names=dataset_names,
    output_dir=output_dir,
    output_filename=output_filename)

store_input_node = Hdf5Write(
    dataset_names=input_dataset_names,
    output_dir=output_dir,
    output_filename=output_filename)

gpn.util.run_augmentations(
    data_providers=data_providers,
    augmentations=(SimpleAugment(transpose_only=[1,2]),),
    input_roi=input_roi,
    output_node=output_node,
    store_inputs_node=store_input_node)

jnius_config.add_options('-Xmx{}'.format(args.max_heap_size))

import payntera.jfx
import imglyb
from jnius import autoclass
payntera.jfx.init_platform()

PainteraBaseView = autoclass('org.janelia.saalfeldlab.paintera.PainteraBaseView')
viewer = PainteraBaseView.defaultView()
pbv = viewer.baseView
scene, stage = payntera.jfx.start_stage(viewer.paneWithStatus.getPane())


with h5py.File(os.path.join(output_dir, output_filename)) as f:
    raw = f[dataset_names[RAW]].value
    raw_augmented = f[input_dataset_names[RAW]].value

raw_img           = imglyb.to_imglib(raw)
raw_augmented_img = imglyb.to_imglib(raw_augmented)

raw_state = pbv.addSingleScaleRawSource(raw_img, input_resolution[::-1], input_roi.get_begin()[::-1], np.min(raw), np.max(raw), 'raw')
raw_augmented_state = pbv.addSingleScaleRawSource(raw_augmented_img, input_resolution[::-1], input_roi.get_begin()[::-1], np.min(raw_augmented), np.max(raw_augmented), 'raw-augmented')
viewer.keyTracker.installInto(scene)
scene.addEventFilter(autoclass('javafx.scene.input.MouseEvent').ANY, viewer.mouseTracker)

hidden_state = payntera.jfx.WaitForHidden(stage=stage)

while not hidden_state.is_hidden:
    time.sleep(0.1)