from __future__ import division

import logging
import math

import scipy.ndimage

import numpy as np

import augment

from gunpowder import BatchFilter, Roi, ArrayKey, ArraySpec, Coordinate

logger = logging.getLogger(__name__)

class ElasticAugment(BatchFilter):

    def __init__(self, voxel_size, spatial_dims=3):
        super(BatchFilter, self).__init__()
        self.voxel_size = voxel_size
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

        logging.debug('preparing %s with voxel size %s', type(self).__name__, self.voxel_size)

        for key, spec in request.items():

            assert isinstance(key, ArrayKey), 'Only ArrayKey supported but got %s in request'%type(key)

            voxel_size            = spec.voxel_size if isinstance(spec, ArraySpec) else self.voxel_size
            target_roi            = self.__spatial_roi(spec.roi).snap_to_grid(voxel_size)
            self.target_rois[key] = target_roi
            target_roi_voxels     = target_roi // voxel_size

            logging.debug('voxel size for key %s: %s', key, voxel_size)

            vs_ratio     = np.array([vs1/vs2 for vs1, vs2 in zip(voxel_size, self.voxel_size)])
            offset_world = (target_roi - master_roi_snapped.get_begin()).get_begin()
            scale        = 1.0 / vs_ratio
            offset       = offset_world / voxel_size

            logging.debug('scale %s and offset %s for key %s', scale, offset, key)

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
            assert (self.target_rois[key].get_begin() == request[key].roi.get_begin()[-self.spatial_dims:])
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
        # if sum(self.jitter_sigma) > 0:
        #     transformation += augment.create_elastic_transformation(
        #             target_shape,
        #             self.control_point_spacing,
        #             self.jitter_sigma,
        #             subsample=self.subsample)
        # rotation = np.random()*self.rotation_max_amount + self.rotation_start
        # if rotation != 0:
        #     transformation += augment.create_rotation_transformation(
        #             target_shape,
        #             rotation,
        #             subsample=self.subsample)
        #
        # if self.subsample > 1:
        #     transformation = augment.upscale_transformation(
        #             transformation,
        #             target_shape)
        #
        # if self.prob_slip + self.prob_shift > 0:
        #     self.__misalign(transformation)

        return transformation

    def __spatial_roi(self, roi):
        return Roi(
            roi.get_begin()[-self.spatial_dims:],
            roi.get_shape()[-self.spatial_dims:]
        )

    def __affine(self, array, scale, offset, target_roi, dtype=np.float32, order=1):
        ndim   = array.shape[0]
        output = np.empty((ndim,) + target_roi.get_shape(), dtype=dtype)
        logging.debug('Transforming array %s with scale %s and offset %s into target_roi%s', array.shape, scale, offset, target_roi)
        for d in range(ndim):
            scipy.ndimage.affine_transform(input=array[d], matrix=scale, offset=offset, output=output[d], order=order)
        return output

    def __get_minimal_containing_roi(self, transformation):

        dims = transformation.shape[0]

        # get bounding box of needed data for transformation
        bb_min = Coordinate(int(math.floor(transformation[d].min())) for d in range(dims))
        bb_max = Coordinate(int(math.ceil(transformation[d].max())) + 1 for d in range(dims))

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