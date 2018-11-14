import argparse
import functools
import h5py
import numpy as np

import logging
logging.basicConfig()
_logger = logging.getLogger(__name__)

def mask(container, dataset, mask_pattern, ignore_values, overwrite=False):
    mask_dataset = mask_pattern.format(dataset.split('/')[-1])
    if  mask_dataset == dataset:
        raise Exception("Mask dataset {} and dataset {} are the same (pattern={})!".format(mask_dataset, dataset, mask_pattern))

    with h5py.File(container, 'r+') as f:
        if mask_dataset in f:
            if overwrite:
                _logger.warning('Overwriting mask dataset %s', mask_dataset)
                del f[mask_dataset]
            else:
                raise Exception('Data set {} already exists!'.format(mask_dataset))
        data = f[dataset].value
    mask = functools.reduce(lambda x, y: x & y,  map(lambda v: data != v, ignore_values)).astype(np.float32)
    with h5py.File(container, 'r+') as f:
        ds = f.create_dataset(mask_dataset, data=mask, chunks=f[dataset].chunks)
        for k, v in f[dataset].attrs.items():
            ds.attrs[k] = v





parser = argparse.ArgumentParser(description='Compare cremi volumes for label equality.')
parser.add_argument('container', metavar='CONTAINER', type=str, nargs=None, help='HDF5 container')
parser.add_argument('datasets', metavar='DATASET', type=str, nargs='+', help='Data sets for which to create mask')
parser.add_argument('--mask-pattern', '-m', metavar='MASK_PATTERN', type=str, default='/volumes/masks/{}', help='Mask dataset will be created from this pattern: MASK_PATTERN.format(DATASET.split("/")[-1])')
parser.add_argument('--log-level', metavar='LOG_LEVEL', type=str, choices=('DEBUG', 'WARN', 'INFO', 'ERROR'), default='WARN')
parser.add_argument('--ignore-values', metavar='IGNORE_VALUES', type=int, default=(0,), nargs='+', help='Values that will be zeroed out in mask.')
parser.add_argument('--overwrite-existing', default=False, action='store_true', help='Overwrite existing data sets.')
args = parser.parse_args()
_logger.setLevel(logging.getLevelName(args.log_level))
_logger.debug('args=%s', args)

for dataset in args.datasets:
    _logger.info('Creating mask for data set %s with background values %s', dataset, args.ignore_values)
    mask(args.container, dataset, args.mask_pattern, args.ignore_values, overwrite=args.overwrite_existing)
