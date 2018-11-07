import augment
import numpy as np

shape = (10,)
test_img = np.random.rand(*shape)
subsample = 3

identity     = augment.create_identity_transformation(shape=shape, subsample=subsample)
elastic      = augment.create_elastic_transformation(shape=shape, control_point_spacing=(1, 1), jitter_sigma=2.0, subsample=subsample)
transform    = identity + elastic
print(transform)
displacement = augment.upscale_transformation(elastic, shape)
transform    = augment.upscale_transformation(transform, shape)


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
