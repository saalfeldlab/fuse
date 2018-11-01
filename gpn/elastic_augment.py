from __future__ import division

import logging
import math

import scipy.ndimage

import numpy as np

import augment

from gunpowder import BatchFilter, Roi, ArrayKey, ArraySpec, Coordinate

logger = logging.getLogger(__name__)

class ElasticAugment(BatchFilter):

    def __init__(
            self,
            voxel_size,
            control_point_spacing,
            jitter_sigma,
            rotation_interval,
            prob_slip=0,
            prob_shift=0,
            max_misalign=0,
            spatial_dims=3):
        super(BatchFilter, self).__init__()
        self.voxel_size = voxel_size
        self.control_point_spacing = control_point_spacing
        self.jitter_sigma = jitter_sigma
        self.rotation_start = rotation_interval[0]
        self.rotation_max_amount = rotation_interval[1] - rotation_interval[0]
        self.prob_slip = prob_slip
        self.prob_shift = prob_shift
        self.max_misalign = max_misalign
        self.spatial_dims = spatial_dims

        self.transformations = {}
        self.target_rois = {}

    def setup(self):
        pass

    def prepare(self, request):

        total_roi  = request.get_total_roi()
        master_roi = self.__spatial_roi(total_roi)
        logger.debug("master roi is %s", master_roi)

        master_roi_snapped  = master_roi.snap_to_grid(self.voxel_size, mode='grow')
        master_roi_voxels   = master_roi_snapped // self.voxel_size
        master_tf           = self.__create_transformation(master_roi_voxels.get_shape())
        master_displacement = self.__get_displacement(master_tf)
        displacement_world  = np.empty(master_displacement.shape, dtype=np.float32)
        for d in range(displacement_world.shape[0]):
            displacement_world[d, ...] = master_displacement[d, ...] * self.voxel_size[d]

        self.transformations.clear()
        self.target_rois.clear()

        logger.debug('preparing %s with voxel size %s', type(self).__name__, self.voxel_size)

        for key, spec in request.items():

            assert isinstance(key, ArrayKey), 'Only ArrayKey supported but got %s in request'%type(key)

            logger.debug('preparing key %s with spec %s', key, spec)

            voxel_size            = spec.voxel_size if isinstance(spec, ArraySpec) else self.voxel_size
            target_roi            = self.__spatial_roi(spec.roi).snap_to_grid(voxel_size)
            self.target_rois[key] = target_roi
            target_roi_voxels     = target_roi // voxel_size

            logger.debug('voxel size for key %s: %s', key, voxel_size)

            vs_ratio     = np.array([vs1/vs2 for vs1, vs2 in zip(voxel_size, self.voxel_size)])
            offset_world = (target_roi - master_roi_snapped.get_begin()).get_begin()
            scale        = vs_ratio#1.0 / vs_ratio
            offset       = -offset_world / self.voxel_size#offset_world / voxel_size

            logger.debug('scale %s and offset %s for key %s', scale, offset, key)

            displacements = self.__affine(displacement_world, scale, offset, target_roi_voxels)
            self.__scale_displacements(displacements, voxel_size)
            absolute_positions = self.__as_absolute_positions_in_voxels(displacements)
            minimal_containing_roi_voxels = self.__get_minimal_containing_roi(absolute_positions)
            source_roi = minimal_containing_roi_voxels * voxel_size + target_roi.get_begin()

            self.transformations[key] = absolute_positions

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



    def __create_transformation(self, target_shape):

        transformation = augment.create_identity_transformation(
                target_shape,
                subsample=1)
        # needs control points in world coordinates
        # TODO add these transformations as well
        if sum(self.jitter_sigma) > 0:
            transformation += augment.create_elastic_transformation(
                    target_shape,
                    self.control_point_spacing,
                    self.jitter_sigma,
                    subsample=1)
        rotation = np.random.random()*self.rotation_max_amount + self.rotation_start
        if rotation != 0:
            transformation += augment.create_rotation_transformation(
                    target_shape,
                    rotation,
                    subsample=1)

        if self.prob_slip + self.prob_shift > 0:
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

        dims = 3
        bb_min = tuple(int(math.floor(transformation[d].min())) for d in range(dims))
        bb_max = tuple(int(math.ceil(transformation[d].max())) + 1 for d in range(dims))
        logger.debug("min/max of transformation: " + str(bb_min) + "/" + str(bb_max))

        for z in range(num_sections):
            transformation[1][z,:,:] += shifts[z][1]
            transformation[2][z,:,:] += shifts[z][2]

        bb_min = tuple(int(math.floor(transformation[d].min())) for d in range(dims))
        bb_max = tuple(int(math.ceil(transformation[d].max())) + 1 for d in range(dims))
        logger.debug("min/max of transformation after misalignment: " + str(bb_min) + "/" + str(bb_max))

    def __random_offset(self):

        return Coordinate((0,) + tuple(self.max_misalign - np.random.randint(0, 2*int(self.max_misalign)) for d in range(2)))