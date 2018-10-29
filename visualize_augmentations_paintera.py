import glob
import os
import time

import numpy as np

import gpn.util
import jnius_config
from gunpowder import Hdf5Source, Roi, Coordinate

RAW       = gpn.util.RAW
GT_LABELS = gpn.util.GT_LABELS

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--max-heap-size", default="16g")
args = parser.parse_args()

data_providers = []
data_dir = '/groups/saalfeld/home/hanslovskyp/experiments/quasi-isotropic/data'
file_pattern = 'sample_A_padded_20160501-2-additional-sections-fixed-offset.h5'



for data in glob.glob(os.path.join(data_dir, file_pattern)):
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
roi = Roi(offset=(0, 0, 0), shape=Coordinate((120, 100, 100)) * output_resolution)#shape=(7200, 648, 648))

batch = gpn.util.run_augmentations(
    data_providers=data_providers,
    roi=lambda key: roi,
    augmentations=(),
    keys_with_sizes=((RAW, input_roi), (GT_LABELS, output_roi)))

jnius_config.add_options('-Xmx{}'.format(args.max_heap_size))

import payntera.jfx
import imglyb
from jnius import autoclass
payntera.jfx.init_platform()

PainteraBaseView = autoclass('org.janelia.saalfeldlab.paintera.PainteraBaseView')
viewer = PainteraBaseView.defaultView()
pbv = viewer.baseView
scene, stage = payntera.jfx.start_stage(viewer.paneWithStatus.getPane())
payntera.jfx.invoke_on_jfx_application_thread(lambda: pbv.orthogonalViews().setScreenScales([0.3, 0.1, 0.03]))


raw = batch[RAW]
labels = batch[GT_LABELS]

raw_img    = imglyb.to_imglib(raw.data)
labels_img = imglyb.to_imglib(labels.data)

max_label = np.max(labels.data)

raw_state = pbv.addSingleScaleRawSource(raw_img, input_resolution[::-1], input_roi.get_begin()[::-1], np.min(raw.data), np.max(raw.data), 'raw-augmented')
label_state = pbv.addSingleScaleLabelSource(labels_img, output_resolution[::-1], output_roi.get_begin()[::-1], max_label+1, 'labels-augmented')
viewer.keyTracker.installInto(scene)
scene.addEventFilter(autoclass('javafx.scene.input.MouseEvent').ANY, viewer.mouseTracker)

while not stage.isHidden():
    time.sleep(0.1)
