from __future__ import print_function
import logging
import traceback

import sys

logger = logging.getLogger(__name__)


class NoSuchModule(object):
    def __init__(self, name):
        self.__name = name
        self.__traceback_str = traceback.format_tb(sys.exc_info()[2])
        errtype, value = sys.exc_info()[:2]
        self.__exception = errtype(value)

    def __getattr__(self, item):
        raise self.__exception

try:
    import z5py
except ImportError:
    z5py = NoSuchModule('z5py')