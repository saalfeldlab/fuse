from __future__ import division

import logging
import math

import scipy.ndimage

import numpy as np

import augment

from gunpowder import BatchFilter, Roi, ArrayKey, ArraySpec, Coordinate

logger = logging.getLogger(__name__)

def _create_identity_transformation(shape, voxel_size=None, offset=None, subsample=1):

    dims = len(shape)

    if voxel_size is None:
        voxel_size = Coordinate((1,) * dims)

    if offset is None:
        offset = Coordinate((0,) * dims)
    subsample_shape = tuple(max(1,int(s/subsample)) for s in shape)
    step_width = tuple(float(shape[d]-1)/(subsample_shape[d]-1) if subsample_shape[d] > 1 else 1 for d in range(dims))
    step_width = tuple(s*vs for s, vs in zip(step_width, voxel_size))

    axis_ranges = (
            np.arange(subsample_shape[d], dtype=np.float32)*step_width[d] + offset[d]
            for d in range(dims)
    )
    return np.array(np.meshgrid(*axis_ranges, indexing='ij'), dtype=np.float32)


def _upscale_transformation(transformation, output_shape, interpolate_order=1, dtype=np.float32):

    input_shape = transformation.shape[1:]

    dims = len(output_shape)
    scale = tuple(float(s)/c for s,c in zip(output_shape, input_shape))

    scaled = np.empty((dims,)+output_shape, dtype=dtype)
    for d in range(dims):
        scipy.ndimage.zoom(transformation[d], zoom=scale, output=scaled[d], order=interpolate_order, mode='nearest')

    return scaled


class ElasticAugment(BatchFilter):
    """
    jitter_sigma in world space
    """

    def __init__(
            self,
            voxel_size,
            control_point_spacing,
            jitter_sigma,
            rotation_interval,
            subsample=1,
            prob_slip=0,
            prob_shift=0,
            max_misalign=0,
            spatial_dims=3,
            seed=None):
        super(BatchFilter, self).__init__()
        self.voxel_size = voxel_size
        self.control_point_spacing = control_point_spacing
        self.jitter_sigma = jitter_sigma
        self.rotation_start = rotation_interval[0]
        self.rotation_max_amount = rotation_interval[1] - rotation_interval[0]
        self.subsample = subsample
        self.prob_slip = prob_slip
        self.prob_shift = prob_shift
        self.max_misalign = max_misalign
        self.spatial_dims = spatial_dims
        self.seed = seed

        logger.debug('initialized with parameters '
                     'voxel_size=%s '
                     'control_point_spacing=%s '
                     'jitter_sigma=%s '
                     'rotation_start=%f '
                     'rotation_max_amount=%f '
                     'subsample=%f '
                     'prob_slip=%f '
                     'prob_shift=%f '
                     'max_misalign=%f '
                     'spatial_dims=%d '
                     'seed=%d',
                     self.voxel_size,
                     self.control_point_spacing,
                     self.jitter_sigma,
                     self.rotation_start,
                     self.rotation_max_amount,
                     self.subsample,
                     self.prob_slip,
                     self.prob_shift,
                     self.max_misalign,
                     self.spatial_dims,
                     self.seed)

        assert isinstance(self.subsample, int), 'subsample has to be integer'
        assert self.subsample >= 1, 'subsample has to be strictly positive'

        self.transformations = {}
        self.target_rois = {}

    def setup(self):
        pass

    def prepare(self, request):
        logger.debug('%s preparing request %s', type(self).__name__, request)

        self.__sanity_check(request)

        total_roi  = request.get_total_roi()
        master_roi = self.__spatial_roi(total_roi)
        logger.debug("master roi is %s with voxel size %s", master_roi, self.voxel_size)

        if self.seed is not None:
            np.random.seed(self.seed)

        # create displacements in world space.
        # TODO displacement is inverse look up?
        master_roi_snapped = master_roi.snap_to_grid(self.voxel_size, mode='grow')
        master_roi_voxels  = master_roi_snapped // self.voxel_size
        master_transform   = self.__create_transformation(master_roi_voxels.get_shape(), offset=master_roi_snapped.get_begin())

        self.transformations.clear()
        self.target_rois.clear()

        logger.debug('preparing %s with voxel size %s', type(self).__name__, self.voxel_size)

        for key, spec in request.items():

            assert isinstance(key, ArrayKey), 'Only ArrayKey supported but got %s in request'%type(key)

            logger.debug('preparing key %s with spec %s', key, spec)

            voxel_size            = spec.voxel_size if isinstance(spec, ArraySpec) else self.voxel_size
            logger.debug('voxel size is %s for key %s', voxel_size, key)
            # Todo we could probably remove snap_to_grid, we already check spec.roi % voxel_size == 0
            target_roi            = self.__spatial_roi(spec.roi).snap_to_grid(voxel_size)
            self.target_rois[key] = target_roi
            target_roi_voxels     = target_roi // voxel_size

            logger.debug('target roi is %s, request roi is %s, for key %s with voxel size %s', target_roi, spec.roi, key, voxel_size)

            # get scale and offset to transform/interpolate master displacement to current spec
            vs_ratio     = np.array([vs1/vs2 for vs1, vs2 in zip(voxel_size, self.voxel_size)])
            offset_world = target_roi.get_begin() - master_roi_snapped.get_begin()
            scale        = vs_ratio
            offset       = offset_world / self.voxel_size

            logger.debug('scale %s and offset %s for key %s', scale, offset, key)

            # need to pass inverse transform, hence -offset
            transform    = self.__affine(master_transform, scale, offset, target_roi_voxels)
            logger.debug('key %s transform statistics %s %s %s %s', key, tuple(map(np.mean, transform)), tuple(map(np.std, transform)), tuple(map(np.min, transform)), tuple(map(np.max, transform)))
            source_roi = self.__get_source_roi(transform).snap_to_grid(voxel_size)
            logger.debug('source roi for key %s is %s', key, source_roi)
            logger.debug('min/max of transform for key %s is %s %s', key, tuple(map(np.min, transform)), tuple(map(np.max, transform)))
            self.__shift_transformation(-target_roi.get_begin(), transform)
            logger.debug('key %s shifted transform statistics %s %s %s %s', key, tuple(map(np.mean, transform)), tuple(map(np.std, transform)), tuple(map(np.min, transform)), tuple(map(np.max, transform)))
            logger.debug('key %s transform statistics after shift %s %s', key, tuple(map(np.mean, transform)), tuple(map(np.std, transform)))
            logger.debug('source roi vs target roi %s %s', source_roi, target_roi)
            for d, (vs, b1, b2) in enumerate(zip(voxel_size, target_roi.get_begin(), source_roi.get_begin())):
                pixel_offset = (b1 - b2) / vs
                logger.debug('pixel offset for dimension %d: %f', d, pixel_offset)
                transform[d] = transform[d] / vs + pixel_offset

            logger.debug('min/max of transform for key %s is %s %s', key, tuple(map(np.min, transform)), tuple(map(np.max, transform)))

            self.transformations[key] = transform

            # update upstream request
            spec.roi = Roi(
                spec.roi.get_begin()[:-self.spatial_dims] + source_roi.get_begin()[-self.spatial_dims:],
                spec.roi.get_shape()[:-self.spatial_dims] + source_roi.get_shape()[-self.spatial_dims:])

    def process(self, batch, request):

        for key, _ in request.items():
            assert key in batch.arrays, 'only arrays supported but got %s'%key
            array = batch.arrays[key]

            # for arrays, the target ROI and the requested ROI should be the
            # same in spatial coordinates
            assert \
                self.target_rois[key].get_begin() == request[key].roi.get_begin()[-self.spatial_dims:], \
                'inconsistent offsets {} -- {} for key {}'.format(
                    self.target_rois[key].get_begin(),
                    request[key].roi.get_begin()[-self.spatial_dims:],
                    key)
            assert (self.target_rois[key].get_shape() == request[key].roi.get_shape()[-self.spatial_dims:])

            # reshape array data into (channels,) + spatial dims
            shape = array.data.shape
            data = array.data.reshape((-1,) + shape[-self.spatial_dims:])
            logger.debug('key %s transform statistics %s %s', key, tuple(map(np.mean, self.transformations[key])), tuple(map(np.std, self.transformations[key])))

            # apply transformation on each channel
            data = np.array([
                augment.apply_transformation(
                    data[c],
                    self.transformations[key],
                    interpolate=self.spec[key].interpolatable)
                for c in range(data.shape[0])
            ])

            data_roi = request[key].roi / self.spec[key].voxel_size
            array.data = data.reshape(data_roi.get_shape())

            # restore original ROIs
            array.spec.roi = request[key].roi

    def __create_transformation(self, target_shape, offset):

        logger.debug('creating displacement for shape %s, subsample %d', target_shape, self.subsample)

        if self.subsample > 1:
            identity_shape = tuple(max(1, int(s / self.subsample)) for s in target_shape)
        else:
            identity_shape = target_shape

        # transformation = np.zeros((len(target_shape),) + identity_shape, dtype=np.float32)
        # needs control points in world coordinates
        # TODO add these transformations as well
        transformation = _create_identity_transformation(target_shape, subsample=self.subsample, voxel_size=self.voxel_size, offset=offset)
        if np.any(np.asarray(self.jitter_sigma) > 0):
            logger.debug('Jittering with sigma=%s and spacing=%s', self.jitter_sigma, self.control_point_spacing)
            elastic = augment.create_elastic_transformation(
                    target_shape,
                    self.control_point_spacing,
                    self.jitter_sigma,
                    subsample=self.subsample)
            logger.debug('elastic mean=%s std=%s', np.mean(elastic.reshape(elastic.shape[0], -1), axis=-1), np.std(elastic.reshape(elastic.shape[0], -1), axis=-1))
            transformation += elastic
        rotation = np.random.random()*self.rotation_max_amount + self.rotation_start
        logger.debug('rotation is %f', rotation)
        if rotation != 0:
            logger.debug('rotating with rotation=%f', rotation)
            transformation += augment.create_rotation_transformation(
                    target_shape,
                    rotation,
                    subsample=self.subsample)

        if self.subsample > 1:
            logger.debug('upscaling subsampled transformation: %d', self.subsample)
            logger.debug('tf before upscale mean=%s std=%s', np.mean(transformation.reshape(transformation.shape[0], -1), axis=-1), np.std(transformation.reshape(transformation.shape[0], -1), axis=-1))
            transformation = _upscale_transformation(
                    transformation,
                    target_shape)
            logger.debug('tf after upscale mean=%s std=%s', np.mean(transformation.reshape(transformation.shape[0], -1), axis=-1), np.std(transformation.reshape(transformation.shape[0], -1), axis=-1))

        if self.prob_slip + self.prob_shift > 0:
            logger.debug('misaligning')
            self.__misalign(transformation)

        return transformation

    def __spatial_roi(self, roi):
        return Roi(
            roi.get_begin()[-self.spatial_dims:],
            roi.get_shape()[-self.spatial_dims:]
        )

    def __affine(self, array, scale, offset, target_roi, dtype=np.float32, order=1):
        '''taken from the scipy 0.18.1 doc:
https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.ndimage.affine_transform.html#scipy.ndimage.affine_transform

Apply an affine transformation.
The given matrix and offset are used to find for each point in the output the corresponding coordinates in the input by
an affine transformation. The value of the input at those coordinates is determined by spline interpolation of the
requested order. Points outside the boundaries of the input are filled according to the given mode.

Given an output image pixel index vector o, the pixel value is determined from the input image at position
np.dot(matrix,o) + offset.

A diagonal matrix can be specified by supplying a one-dimensional array-like to the matrix parameter, in which case a
more efficient algorithm is applied.

Changed in version 0.18.0: Previously, the exact interpretation of the affine transformation depended on whether the
matrix was supplied as a one-dimensional or two-dimensional array. If a one-dimensional array was supplied to the matrix
parameter, the output pixel value at index o was determined from the input image at position matrix * (o + offset).
        '''
        ndim   = array.shape[0]
        output = np.empty((ndim,) + target_roi.get_shape(), dtype=dtype)
        logger.debug(
            'Transforming array %s with scale %s and offset %s into target_roi%s',
            array.shape,
            scale,
            offset,
            target_roi)
        for d in range(ndim):
            scipy.ndimage.affine_transform(
                input=array[d],
                matrix=scale,
                offset=offset,
                output=output[d],
                output_shape=output[d].shape,
                order=order,
                mode='nearest')
        return output

    def __get_minimal_containing_roi(self, transformation):

        dims = transformation.shape[0]

        # get bounding box of needed data for transformation
        bb_min = Coordinate(int(math.floor(transformation[d].min())) for d in range(dims))
        bb_max = Coordinate(int(math.ceil(transformation[d].max())) + 1 for d in range(dims))

        logger.debug('Got bb_min=%s and bb_max=%s', bb_min, bb_max)

        # create roi sufficiently large to feed transformation
        source_roi = Roi(
                bb_min,
                bb_max - bb_min
        )

        return source_roi

    def __shift_transformation(self, shift, transformation):
        logger.debug('shifting transform by %s', shift)
        for d in range(transformation.shape[0]):
            transformation[d] += shift[d]

    def __get_displacement(self, transform, dtype=np.float32):
        logger.debug('getting displacment from absolute transform with shape %s', transform.shape)
        displacement = np.empty(transform.shape, dtype=dtype)
        positions = np.meshgrid(*[np.arange(d) for d in transform.shape[1:]], indexing='ij')
        for d in range(transform.shape[0]):
            logger.debug(
                'getting displacement from absolute transform for dim %d: transform shape %s positions shape %s',
                d,
                transform[d, ...].shape,
                positions[d].shape)
            displacement[d, ...] = transform[d, ...] - positions[d]
        return displacement

    def __scale_displacements(self, displacements, voxel_size):
        for d in range(displacements.shape[0]):
            displacements[d, ...] = displacements[d, ...] / voxel_size[d]

    def __as_absolute_positions_in_voxels(self, displacements, dtype=np.float32):
        positions = np.meshgrid(*[np.arange(d) for d in displacements.shape[1:]], indexing='ij')
        absolute  = np.empty(displacements.shape, dtype=dtype)
        for d in range(displacements.shape[0]):
            absolute[d, ...] = positions[d] + displacements[d, ...]
        return absolute

    def __misalign(self, transformation):

        assert transformation.shape[0] == 3, (
            "misalign can only be applied to 3D volumes")

        num_sections = transformation[0].shape[0]

        shifts = [Coordinate((0,0,0))]*num_sections
        for z in range(num_sections):

            r = np.random.random()

            if r <= self.prob_slip:

                shifts[z] = self.__random_offset()

            elif r <= self.prob_slip + self.prob_shift:

                offset = self.__random_offset()
                for zp in range(z, num_sections):
                    shifts[zp] += offset

        logger.debug("misaligning sections with " + str(shifts))

        for z in range(num_sections):
            transformation[1][z,:,:] += shifts[z][1]
            transformation[2][z,:,:] += shifts[z][2]


    def __random_offset(self):
        return Coordinate((0,) + tuple(self.max_misalign - np.random.randint(0, 2*int(self.max_misalign)) for d in range(2)))


    def __sanity_check(self, request):

        for key, spec in request.items():

            logger.debug('Sanity checking key=%s spec=%s', key, spec)

            assert key is not None, 'Key is none'
            assert spec is not None, 'Spec is None for key %s'%key
            assert spec.voxel_size is not None, 'Voxel size is None for key %s'%key
            assert spec.roi is not None, 'Roi is None for key %s'%key
            assert spec.roi.get_begin(), 'Offset is None for key %s'%key
            assert spec.roi.get_shape(), 'Shape is None for key %s'%key
            assert np.all(np.mod(self.__spatial_roi(spec.roi).get_begin(), spec.voxel_size) == 0), \
                'begin of roi %s not snapped to voxel size %s for key %s'%(spec.roi.get_begin(), spec.voxel_size, key)
            assert np.all(np.mod(self.__spatial_roi(spec.roi).get_shape(), spec.voxel_size) == 0), \
                'shape of roi %s not snapped to voxel size %s for key %s'%(spec.roi.get_shape(), spec.voxel_size, key)

    def __get_source_roi(self, transformation):

        dims = transformation.shape[0]

        # get bounding box of needed data for transformation
        bb_min = Coordinate(int(math.floor(transformation[d].min())) for d in range(dims))
        bb_max = Coordinate(int(math.ceil(transformation[d].max())) + 1 for d in range(dims))
        logger.debug('getting source roi with min=%s and max=%s', bb_min, bb_max)

        # create roi sufficiently large to feed transformation
        source_roi = Roi(
                bb_min,
                bb_max - bb_min
        )

        return source_roi
