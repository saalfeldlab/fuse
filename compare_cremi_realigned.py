import argparse
import h5py
import numpy as np

import logging
logging.basicConfig()
_logger = logging.getLogger(__name__)

def compare(sample):

    container1 = '/groups/saalfeld/saalfeldlab/lauritzen/cremi_fixed/sample%s.h5' % sample
    container2 = '/groups/saalfeld/home/hanslovskyp/experiments/quasi-isotropic/data/realigned/sample_%s_realigned-2-additional-sections-fixed-offset.h5' % sample
    container3 = '/groups/saalfeld/saalfeldlab/lauritzen/cremi_fixed/sample%s.merged.h5' % sample

    dataset1 = 'volumes/labels/merged_ids'
    dataset2 = 'volumes/labels/neuron_ids'
    dataset3 = 'volumes/labels/merged_ids'

    stride1 = 1
    stride2 = 3
    stride3 = 1

    _logger.info('Loading dataset %s from container %s', dataset1, container1)
    with h5py.File(container1, 'r') as f:
        data1 = f[dataset1].value[::stride1, ...]

    _logger.info('Loading dataset %s from container %s', dataset2, container2)
    with h5py.File(container2, 'r') as f:
        data2 = f[dataset2].value[::stride2, ...]

    _logger.info('Loading dataset %s from container %s', dataset2, container3)
    with h5py.File(container3, 'r') as f:
        data3 = f[dataset3].value[::stride3, ...]

    shape_equals12 = data1.shape == data2.shape
    _logger.info('Comparing shapes of %s/%s and %s/%s: %s', container1, dataset1, container2, dataset2, shape_equals12)
    assert shape_equals12
    shape_equals13 = data1.shape == data3.shape
    _logger.info('Comparing shapes of %s/%s and %s/%s: %s', container1, dataset1, container3, dataset3, shape_equals13)
    assert data1.shape == data3.shape

    contents_equals12 = np.all(data1 == data2)
    _logger.info('Comparing contents of %s/%s and %s/%s: %s', container1, dataset1, container2, dataset2, contents_equals12)
    assert contents_equals12, 'Contents of %s/%s and %s/%s differ!' % (container1, dataset1, container2, dataset2)
    contents_equals13 = np.all(data1 == data3)
    _logger.info('Comparing contents of %s/%s and %s/%s: %s', container1, dataset1, container3, dataset3, contents_equals13)
    assert contents_equals13, 'Contents of %s/%s and %s/%s differ!' % (container1, dataset1, container3, dataset3)

parser = argparse.ArgumentParser(description='Compare cremi volumes for label equality.')
parser.add_argument('samples', metavar='SAMPLE_NAMES', type=str, nargs='+', choices=tuple('ABC'), help='Sample identifier, one of {A, B, C}')
parser.add_argument('--log-level', metavar='LOG_LEVEL', type=str, choices=('DEBUG', 'WARN', 'INFO', 'ERROR'), default='WARN')
args = parser.parse_args()
_logger.setLevel(logging.getLevelName(args.log_level))

for sample in args.samples:
    _logger.info('Comparing for sample identifier %s', sample)
    try:
        compare(sample)
    except AssertionError as e:
        _logger.error('Inconsistency detected for sample identifier %s: %s', sample, e.args)
