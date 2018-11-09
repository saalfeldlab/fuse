# uncomment to see debug output
import logging

from gpn import Misalign

logging.getLogger('gpn.elastic_augment').setLevel(logging.DEBUG)
logging.getLogger('gpn.misalign').setLevel(logging.DEBUG)
logging.getLogger('gpn.util').setLevel(logging.DEBUG)

import glob
import os
import time
import numpy as np

import gpn.util
import jnius_config
from gpn import ElasticAugment
from gunpowder import Hdf5Source, Roi, Coordinate

RAW       = gpn.util.RAW
GT_LABELS = gpn.util.GT_LABELS


def add_to_viewer(batch_or_snapshot, keys, name=lambda key: key.identifier, is_label=lambda key, array: array.data.dtype == np.uint64):
    states = {}
    for key in keys:
        if not key in batch_or_snapshot:
            continue

        data       = batch_or_snapshot[key]
        data_img   = imglyb.to_imglib(data.data)
        voxel_size = data.spec.voxel_size[::-1]
        offset     = data.spec.roi.get_begin()[::-1]

        if is_label(key, data):
            state = pbv.addSingleScaleLabelSource(
                data_img,
                voxel_size,
                offset,
                np.max(data.data) + 1,
                name(key))
        else:
            state = pbv.addSingleScaleRawSource(
                data_img,
                voxel_size,
                offset,
                np.min(data.data),
                np.max(data.data),
                name(key))
        states[key] = state

    return states

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--max-heap-size", default="16g")
args = parser.parse_args()

data_providers = []
data_dir = '/groups/saalfeld/home/hanslovskyp/experiments/quasi-isotropic/data'
# data_dir = os.path.expanduser('~/Dropbox/cremi-upsampled/')
file_pattern = 'sample_A_padded_20160501-2-additional-sections-fixed-offset.h5'
file_pattern = 'sample_B_padded_20160501-2-additional-sections-fixed-offset.h5'
file_pattern = 'sample_C_padded_20160501-2-additional-sections-fixed-offset.h5'



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
output_resolution = Coordinate((120, 108, 108))
offset = (13640, 10932, 10932)

output_shape = Coordinate((60, 100, 100)) * output_resolution
output_offset = (13320 + 3600, 32796 + 36 + 10800, 32796 + 36 + 10800)

overhang = Coordinate((360, 108, 108)) * 16

input_shape = output_shape + overhang * 2
input_offset = Coordinate(output_offset) - overhang


output_roi = Roi(offset=output_offset, shape=output_shape)
input_roi  = Roi(offset=input_offset, shape=input_shape)

augmentations = (
    ElasticAugment(
        voxel_size=(360, 36, 36),
        control_point_spacing=(4, 40, 40),
        jitter_sigma=(0, 1 * 2 * 36, 1 * 2 * 36),
        rotation_interval=(0 * np.pi / 8, 0*2*np.pi),
        subsample=8,
        seed=100),
    Misalign(z_resolution=360, prob_slip=0.2, prob_shift=0.0, max_misalign=(3600, 0), seed=100),
)

keys = (RAW, GT_LABELS)[:]

batch, snapshot = gpn.util.run_augmentations(
    data_providers=data_providers,
    roi=lambda key: output_roi.copy() if key == GT_LABELS else input_roi.copy(),
    augmentations=augmentations,
    keys=keys,
    voxel_size=lambda key: input_resolution if key == RAW else (output_resolution if key == GT_LABELS else None))

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

snapshot_states = add_to_viewer(snapshot, keys=keys, name=lambda key: '%s-snapshot'%key.identifier)
states          = add_to_viewer(batch, keys=keys)

viewer.keyTracker.installInto(scene)
scene.addEventFilter(autoclass('javafx.scene.input.MouseEvent').ANY, viewer.mouseTracker)

while stage.isShowing():
    time.sleep(0.1)
