import glob
import os
import time

import numpy as np

import gpn.util
import jnius_config
from gunpowder import Hdf5Source, Roi, Hdf5Write, Coordinate

RAW       = gpn.util.RAW
GT_LABELS = gpn.util.GT_LABELS

import argparse
import h5py
parser = argparse.ArgumentParser()
parser.add_argument("--max-heap-size", default="80g")
args = parser.parse_args()

data_providers = []
data_dir = '/groups/saalfeld/home/hanslovskyp/experiments/quasi-isotropic/data'
file_pattern = 'sample_A_padded_20160501-2-additional-sections-fixed-offset.h5'



for data in glob.glob(os.path.join(data_dir, file_pattern)):
    print(data)
    # TODO add masks later on
    h5_source = Hdf5Source(
        data,
        datasets={
            RAW: 'volumes/raw',
            GT_LABELS: 'volumes/labels/neuron_ids-downsampled',
        }
    )
    data_providers.append(h5_source)

input_resolution  = (360, 36, 36)
output_resolution = (120, 108, 108)
offset = (13640, 10932, 10932)

input_roi = Roi(offset=(0, 0, 0), shape=Coordinate((200, 3072, 3072)) * input_resolution)
output_roi = Roi(offset=(120, -36, -36), shape=Coordinate((373, 1250, 1250)) * output_resolution)
roi = Roi(offset=(0, 0, 0), shape=Coordinate((12, 10, 10)) * output_resolution)#shape=(7200, 648, 648))
output_dir = os.path.join(os.path.expanduser("~"), "tmp")
output_filename = "test-nodes.h5"
dataset_names = {RAW: 'volumes/raw-augmented', GT_LABELS: 'volumes/labels/neuron_ids-downsampled-augmented'}
input_dataset_names = {RAW: 'volumes/raw', GT_LABELS: 'volumes/labels/neuron_ids-downsampled'}

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
    roi=lambda key: roi,
    augmentations=(),
    keys_with_sizes=((RAW, input_roi), (GT_LABELS, output_roi)),
    output_node=output_node,
    store_inputs_node=None)

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
    raw = f[input_dataset_names[RAW]].value
    raw_augmented = f[dataset_names[RAW]].value
    labels = f[input_dataset_names[GT_LABELS]].value
    labels_augmented = f[dataset_names[GT_LABELS]].value

raw_img           = imglyb.to_imglib(raw)
raw_augmented_img = imglyb.to_imglib(raw_augmented)

labels_img           = imglyb.to_imglib(labels)
labels_augmented_img = imglyb.to_imglib(labels_augmented)

max_label = np.max(labels)

raw_state = pbv.addSingleScaleRawSource(raw_img, input_resolution[::-1], input_roi.get_begin()[::-1], np.min(raw), np.max(raw), 'raw')
label_state = pbv.addSingleScaleLabelSource(labels_img, output_resolution[::-1], output_roi.get_begin()[::-1], max_label+1, 'labels')
raw_augmented_state = pbv.addSingleScaleRawSource(raw_augmented_img, input_resolution[::-1], input_roi.get_begin()[::-1], np.min(raw_augmented), np.max(raw_augmented), 'raw-augmented')
label_augmented_state = pbv.addSingleScaleLabelSource(labels_augmented_img, output_resolution[::-1], output_roi.get_begin()[::-1], max_label+1, 'labels-augmented')
viewer.keyTracker.installInto(scene)
scene.addEventFilter(autoclass('javafx.scene.input.MouseEvent').ANY, viewer.mouseTracker)

hidden_state = payntera.jfx.WaitForHidden(stage=stage)

while not hidden_state.is_hidden:
    time.sleep(0.1)