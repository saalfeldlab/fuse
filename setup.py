from setuptools import setup
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md')) as f:
    fuse_long_description = f.read()

setup(
    name='fuse',
    version='0.1.0.dev0',
    author='Philipp Hanslovsky',
    author_email='hanslovskyp@janelia.hhmi.org',
    description='Fuse to get gunpowder started',
    long_description=fuse_long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/hanslovsky/fuse',
    license='bsd-2',
    packages=['fuse', 'fuse.ext'],
    install_requires=['gunpowder', 'numpy', 'scipy', 'h5py', 'augment']
)
