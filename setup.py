import os
from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'README.md')) as f:
    fuse_long_description = f.read()

install_requires = [
    'pip>=18.1',
    'numpy',
    'scipy',
    'h5py',
    'augment-nd',
    'gunpowder @ git+https://github.com/funkey/gunpowder@d49573f53e8f23d12461ed8de831d0103acb2715'
]

# will need python 3.6 for f-strings in fuse.version_info
# https://realpython.com/python-f-strings/#f-strings-a-new-and-improved-way-to-format-strings-in-python
# https://www.python.org/dev/peps/pep-0498/

name = 'fuse'

version_info = {}
with open(os.path.join(here, name, 'version_info.py')) as fp:
    exec(fp.read(), version_info)
version = version_info['_version']

setup(
    name='fuse',
    version=version.version(),
    author='Philipp Hanslovsky',
    author_email='hanslovskyp@janelia.hhmi.org',
    description='Fuse to get gunpowder started',
    long_description=fuse_long_description,
    long_description_content_type='text/markdown',
    url=f'https://github.com/hanslovsky/{name}',
    license='bsd-2',
    packages=[name, f'{name}.ext'],
    install_requires=install_requires,
    python_requires='>=3.6'
)
