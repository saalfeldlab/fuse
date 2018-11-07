import augment
import numpy as np

shape = (10, 5)
test_img = np.random.rand(*shape)
subsample = 3
voxel_size = (3.0, 2.5)
jitter_sigma        = (5.0,) * len(shape)
jitter_sigma_voxels = tuple(j/vs for j, vs in zip(jitter_sigma, voxel_size))

identity     = augment.create_identity_transformation(shape=shape, subsample=subsample)
np.random.seed(100)
elastic      = augment.create_elastic_transformation(shape=shape, control_point_spacing=(1, 1), jitter_sigma=jitter_sigma_voxels, subsample=subsample)
np.random.seed(100)
elastic_w    = augment.create_elastic_transformation(shape=shape, control_point_spacing=(1, 1), jitter_sigma=jitter_sigma, subsample=subsample)
transform    = identity + elastic
print(transform)
print('-----')
print(elastic)
print(elastic_w)
print(elastic_w / elastic)
displacement = augment.upscale_transformation(elastic, shape)
transform    = augment.upscale_transformation(transform, shape)
displace_w   = augment.upscale_transformation(elastic_w, shape)


tf1 = transform
tf2 = displacement + augment.create_identity_transformation(shape=shape, subsample=1)

print()
print('-----')
print(tf1)
print('-----')
print(tf2)

d1 = transform - augment.create_identity_transformation(shape=shape, subsample=1)
d2 = displacement
print()
print('-----')
print(d1)
print('-----')
print(d2)

print(np.abs(tf1 - tf2))
