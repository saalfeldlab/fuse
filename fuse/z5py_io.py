import os

from .ext import z5py

from gunpowder.nodes.hdf5like_source_base import Hdf5LikeSource


class Z5Source(Hdf5LikeSource):
    '''A `zarr <https://github.com/zarr-developers/zarr>`_ data source.

    Provides arrays from zarr datasets. If the attribute ``resolution`` is set
    in a zarr dataset, it will be used as the array's ``voxel_size``. If the
    attribute ``offset`` is set in a dataset, it will be used as the offset of
    the :class:`Roi` for this array. It is assumed that the offset is given in
    world units.

    Args:

        filename (``string``):

            The zarr directory.

        datasets (``dict``, :class:`ArrayKey` -> ``string``):

            Dictionary of array keys to dataset names that this source offers.

        array_specs (``dict``, :class:`ArrayKey` -> :class:`ArraySpec`, optional):

            An optional dictionary of array keys to array specs to overwrite
            the array specs automatically determined from the data file. This
            is useful to set a missing ``voxel_size``, for example. Only fields
            that are not ``None`` in the given :class:`ArraySpec` will be used.
    '''

    def __init__(self, filename, datasets, array_specs=None, revert=None):
        super().__init__(filename, datasets, array_specs)
        self.revert = revert

    def _get_array_attribute(self, dataset, attribute, fallback_value, revert=False):
        val = dataset.attrs[attribute] if attribute in dataset.attrs else [fallback_value] * 3
        return val[::-1] if revert else val

    def _revert(self):
        if self.revert is None:
            self.revert = self.filename.endswith('.n5')
        return self.revert

    def _get_voxel_size(self, dataset):
        return Coordinate(self._get_array_attribute(dataset, 'resolution', 1, revert=self._revert()))

    def _get_offset(self, dataset):
        return Coordinate(self._get_array_attribute(dataset, 'offset', 0, revert=self._revert()))

    def _open_file(self, filename):
        return z5py.File(ensure_str(filename), mode='r')


from gunpowder.nodes.hdf5like_write_base import Hdf5LikeWrite
from gunpowder.coordinate import Coordinate
from gunpowder.compat import ensure_str


class Z5Write(Hdf5LikeWrite):
    '''Assemble arrays of passing batches in one zarr container. This is useful
    to store chunks produced by :class:`Scan` on disk without keeping the
    larger array in memory. The ROIs of the passing arrays will be used to
    determine the position where to store the data in the dataset.

    Args:

        dataset_names (``dict``, :class:`ArrayKey` -> ``string``):

            A dictionary from array keys to names of the datasets to store them
            in.

        output_dir (``string``):

            The directory to save the zarr container. Will be created, if it does
            not exist.

        output_filename (``string``):

            The output filename of the container. Will be created, if it does
            not exist, otherwise data is overwritten in the existing container.

        compression_type (``string`` or ``int``):

            Compression strategy.  Legal values are ``gzip``, ``szip``,
            ``lzf``. If an integer between 1 and 10, this indicates ``gzip``
            compression level.

        dataset_dtypes (``dict``, :class:`ArrayKey` -> data type):

            A dictionary from array keys to datatype (eg. ``np.int8``). If
            given, arrays are stored using this type. The original arrays
            within the pipeline remain unchanged.
    '''

    def __init__(
            self,
            dataset_names,
            output_dir='.',
            output_filename='output.hdf',
            compression_type=None,
            dataset_dtypes=None,
            revert=None):
        super().__init__(dataset_names, output_dir, output_filename, compression_type, dataset_dtypes)
        self.revert = revert

    def _get_array_attribute(self, dataset, attribute, fallback_value, revert=False):
        val = dataset.attrs[attribute] if attribute in dataset.attrs else [fallback_value] * 3  # len(dataset.shape)
        return val[::-1] if revert else val

    def _revert(self):
        if self.revert is None:
            self.revert = os.path.join(self.output_dir, self.output_filename).endswith('.n5')
        return self.revert

    def _get_voxel_size(self, dataset):
        return Coordinate(self._get_array_attribute(dataset, 'resolution', 1, revert=self._revert()))

    def _get_offset(self, dataset):
        return Coordinate(self._get_array_attribute(dataset, 'offset', 0, revert=self._revert()))

    def _set_voxel_size(self, dataset, voxel_size):

        if self.output_filename.endswith('.n5'):
            dataset.attrs['resolution'] = voxel_size[::-1]
        else:
            dataset.attrs['resolution'] = voxel_size

    def _set_offset(self, dataset, offset):

        if self.output_filename.endswith('.n5'):
            dataset.attrs['offset'] = offset[::-1]
        else:
            dataset.attrs['offset'] = offset

    def _open_file(self, filename):
        return z5py.File(ensure_str(filename), mode='a')