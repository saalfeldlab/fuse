from setuptools import setup
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md')) as f:
    gpn_long_description = f.read()

setup(
    name='gunpowder nodes',
    version='0.1.0.dev0',
    author='Philipp Hanslovsky',
    author_email='hanslovskyp@janelia.hhmi.org',
    description='Some gunpowder nodes I use..',
    long_description=gpn_long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/hanslovsky/gunpowder-nodes',
    packages=['gpn'],
    install_requires=['gunpowder', 'numpy', 'scipy', 'h5py', 'augment']
)
