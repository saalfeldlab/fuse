from __future__ import division

import logging
import math

import numpy as np
import scipy.ndimage

from gunpowder import BatchFilter, Roi, ArrayKey, Coordinate

logger = logging.getLogger(__name__)


def _spatial_roi(roi, spatial_dims):
    return Roi(
        roi.get_begin()[-spatial_dims:],
        roi.get_shape()[-spatial_dims:]
    )


class Misalign(BatchFilter):
    """
    Misalign serial sections by randomly shifting along yx-coordinate axes (1, 2).
    Arrays can have different voxel sizes but the z-component of each voxel size
    has to be integer multiple of z_resolution. In practice, use the lowest resolution
    along z of all arrays requested in the roi.

    Args:

        z_resolution (``int``):

            Resolution at which to generate shifts. Note that, for all specs in the request,
            spec.voxel_size[0] is integer multiple of z_resolution


        prob_slip (``float``):

            Probability of a section to "slip", i.e., be independently moved in
            x-y.

        prob_shift (``float``):

            Probability of a section and all following sections to move in x-y. This
            is a conditional probability for the case that slip == False. If slip == True,
            no shift will occur for a section.

        max_misalign (``tuple`` of two ``floats``):

            Maximal displacement to shift in x and y. Samples will be drawn
            uniformly.

        ignore_keys_for_slip (``tuple`` of keys):

            Only apply shifts (but not slips) to any key in this ``tuple``.
    """

    def __init__(
            self,
            z_resolution,
            prob_slip=0,
            prob_shift=0,
            max_misalign=(0, 0),
            ignore_keys_for_slip=(),
            seed=None):
        super(BatchFilter, self).__init__()
        self.z_resolution = Coordinate((z_resolution,))
        self.prob_slip = prob_slip
        self.prob_shift = prob_shift
        self.max_misalign = max_misalign
        self.ignore_keys_for_slip = ignore_keys_for_slip
        self.seed = seed

        logger.debug('initialized with parameters '
                     'prob_slip=%f '
                     'prob_shift=%f '
                     'max_misalign=%s '
                     'ignore_keys_for_slip=%s '
                     'seed=%d',
                     self.prob_slip,
                     self.prob_shift,
                     self.max_misalign,
                     self.ignore_keys_for_slip,
                     self.seed)

        self.translations = {}
        self.target_rois = {}

    def setup(self):
        pass

    '''
    prepare:
        target_roi  := requested roi (world space)
        b           := begin of requested roi
        t           := array of translations per section (world space)
        roi         := roi for upstream request (world space), deducted from target_roi:
                         - begin: shift b by min(s)
                         - shape: extend shape of target_roi by max(t) - min(t)
                         - snap to voxel grid
        b_tilde     := begin of roi

    process:
        o           := offset between target_ roi and roi: b - b_tilde
        shifts      := array of shifts (offset into voxel array of roi) per section indexed by i:
                           shifts[i] = o + t[i] / voxel_size
                       use shifts for offset parameter in scipy affine transform
    '''
    def prepare(self, request):
        logger.debug('%s preparing request %s with z_resolution %s', type(self).__name__, request, self.z_resolution)

        self._sanity_check(request)

        total_roi  = request.get_total_roi()
        master_roi = self._z_roi(total_roi)

        if self.seed is not None:
            np.random.seed(self.seed)

        master_roi_snapped = master_roi.snap_to_grid(self.z_resolution, mode='grow')
        master_roi_voxels  = master_roi_snapped // self.z_resolution
        master_shifts, master_slips = map(np.asarray, self._misalign(master_roi_voxels.get_shape()[0]))
            # np.asarray(self._misalign(master_roi_voxels.get_shape()[0]))

        self.translations.clear()
        self.target_rois.clear()

        for key, spec in request.items():
            assert isinstance(key, ArrayKey), 'Only ArrayKey supported but got %s in request'%type(key)
            z_resolution = Coordinate((spec.voxel_size[0],))
            z_roi = self._z_roi(spec.roi)
            z_roi_voxels = z_roi / z_resolution
            z_roi_snapped_voxels = ( master_roi_snapped ) / z_resolution
            voxel_size_ratio = int((self.z_resolution / z_resolution)[0])
            half_voxel_size_diff = (self.z_resolution - z_resolution)[0] / 2
            logger.debug('prepare key %s: half voxel size diff=%s', key, half_voxel_size_diff)
            assert half_voxel_size_diff.is_integer() and (half_voxel_size_diff / z_resolution[0]).is_integer(), \
                'half of voxel size diff %f must be integer multiple of z_resolution %s'%(half_voxel_size_diff, z_resolution)
            logger.debug('prepare key %s: voxel size ratio=%s', key, voxel_size_ratio)
            offset = (z_roi.get_begin() - master_roi_snapped.get_begin())
            voxel_offset = int((z_roi_voxels.get_begin() - z_roi_snapped_voxels.get_begin())[0])
            logger.debug('prepare key %s: offset=%s voxel_offset=%s voxel_size=%s', key, offset, voxel_offset, spec.voxel_size)
            start = voxel_offset + int(half_voxel_size_diff / z_resolution[0])
            logger.debug('prepare key %s: voxel_offset=%s start=%s', key, voxel_offset, start)
            stop = start + int(z_roi_voxels.get_shape()[0])
            master_translations = master_shifts if key in self.ignore_keys_for_slip else master_shifts + master_slips
            translations = np.repeat(master_translations, voxel_size_ratio, axis=0)[start:stop]
            # logger.debug('prepare key %s: translations=%s', key, translations)
            m = np.min(translations, axis=0)
            M = np.max(translations, axis=0)
            d = M - m
            slice_roi = Roi(offset = m + np.asarray(spec.roi.get_begin()[-2:]), shape = d + np.asarray(spec.roi.get_shape()[-2:]))
            slice_roi = slice_roi.snap_to_grid(spec.voxel_size[1:])

            self.translations[key] = translations
            # remember roi of key in original request
            self.target_rois[key] = spec.roi
            # if all translation are > 0, new roi.begin might be larger than original roi.begin, which is ok
            # new roi need not contain all of original roi (target roi)
            spec.roi = Roi(
                spec.roi.get_begin()[:-2] + slice_roi.get_begin(),
                spec.roi.get_shape()[:-2] + slice_roi.get_shape())



    def process(self, batch, request):

        for key, _ in request.items():
            logger.debug('process key %s', key)
            assert key in batch.arrays, 'only arrays supported but got %s'%key
            array = batch.arrays[key]
            voxel_size = np.asarray(array.spec.voxel_size)
            # target_roi is roi in original request
            target_roi = self.target_rois[key]
            target_roi_voxels = _spatial_roi(target_roi, 3) / array.spec.voxel_size
            roi_voxels = _spatial_roi(array.spec.roi, 3) / array.spec.voxel_size
            # offset can be negative, thus use in64 instead of uin64
            offset_voxels = np.asarray(target_roi_voxels.get_begin() - roi_voxels.get_begin())[1:].astype(np.int64)
            slice_shape = np.asarray(target_roi_voxels.get_shape()[1:]).astype(np.int64)
            data = np.empty(shape=target_roi.get_shape()[:-3] + target_roi_voxels.get_shape(), dtype=array.data.dtype)
            interpolate = array.spec.interpolatable
            for index, translation in enumerate(self.translations[key]):
                translation_in_voxels = translation / voxel_size[1:]
                current_slice = array.data[..., index, :, :]
                if np.all(translation_in_voxels == 0):
                    start = offset_voxels
                    stop = start + slice_shape
                    data[..., index, :, :] = current_slice[..., start[0]:stop[0], start[1]:stop[1]]
                else:
                    shift  = offset_voxels + translation_in_voxels
                    source = np.reshape(current_slice, (-1,) + current_slice.shape[-2:])
                    target = np.reshape(data[..., index, :, :], (-1,) + tuple(map(int, slice_shape)))
                    matrix = np.ones((2,))
                    order  = 1 if interpolate else 0
                    for s, t in zip(source, target):
                        # output_shape has to be specified even if output is provided, soooo annoying to figure out
                        # from the scipy doc, offset is wrt input:
                        """
                        Apply an affine transformation.

                        Given an output image pixel index vector ``o``, the pixel value
                        is determined from the input image at position
                        ``np.dot(matrix, o) + offset``."""
                        scipy.ndimage.interpolation.affine_transform(input=s, output=t, output_shape=t.shape, matrix=matrix, offset=shift, order=order)

            array.spec.roi = target_roi
            array.data = data

    def _z_roi(self, roi):
        return Roi(
            roi.get_begin()[-3:-2],
            roi.get_shape()[-3:-2]
        )

    def _misalign(self, num_sections):
        """

        :param num_sections: number of sections
        :return: (shifts, slips)
        """

        slips  = [Coordinate((0,0))]*num_sections
        shifts = [Coordinate((0,0))]*num_sections
        for z in range(num_sections):

            r = np.random.random()

            if r <= self.prob_slip:

                slips[z] = self._random_offset()

            elif r <= self.prob_slip + self.prob_shift:

                offset = self._random_offset()
                for zp in range(z, num_sections):
                    shifts[zp] += offset

        logger.debug("misaligning sections with " + str(shifts))


        return shifts, slips

    def _sanity_check(self, request):

        for key, spec in request.items():

            logger.debug('Sanity checking key=%s spec=%s', key, spec)

            assert key is not None, 'Key is none'
            assert spec is not None, 'Spec is None for key %s'%key
            assert spec.voxel_size is not None, 'Voxel size is None for key %s'%key
            assert spec.roi is not None, 'Roi is None for key %s'%key
            assert spec.roi.get_begin(), 'Offset is None for key %s'%key
            assert spec.roi.get_shape(), 'Shape is None for key %s'%key
            assert int(self.z_resolution[0]) % spec.voxel_size[0] == 0, \
                'z_resolution is not integer multiple of z resolution for key %s'%key

    def _get_source_roi(self, transformation):

        dims = transformation.shape[0]

        # get bounding box of needed data for transformation
        bb_min = Coordinate(int(math.floor(transformation[d].min())) for d in range(dims))
        bb_max = Coordinate(int(math.ceil(transformation[d].max())) + 1 for d in range(dims))

        # from the scipy doc for affine transform, offset is wrt input:
        '''
        Apply an affine transformation.

        Given an output image pixel index vector ``o``, the pixel value
        is determined from the input image at position
        ``np.dot(matrix, o) + offset``.'''

        # create roi sufficiently large to feed transformation
        source_roi = Roi(
                bb_min,
                bb_max - bb_min
        )

        return source_roi

    def _random_offset(self):
        return Coordinate(tuple(ma - np.random.rand() * 2 * ma for ma in self.max_misalign))
