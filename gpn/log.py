from __future__ import division

import logging

import numpy as np

from gunpowder import BatchFilter

_logger = logging.getLogger(__name__)

class _LazyStatisticsString(object):

    def __init__(self, array, *statistics_or_meta):
        self.array = array
        self.statistics_or_meta = statistics_or_meta

    def __str__(self):
        stats = {som: _LazyStatisticsString._run_or_return(getattr(self.array, som, lambda : 'N/A')) for som in self.statistics_or_meta}
        return ' '.join('{}={}'.format(k, v) for k, v in stats.items())

    @staticmethod
    def _run_or_return(maybe_callable, *args):
        return maybe_callable(*args) if callable(maybe_callable) else maybe_callable


class Log(BatchFilter):
    def __init__(
            self,
            prepare_logger = lambda batch: _logger.debug('No logger provided for prepare'),
            process_logger = lambda batch, request: _logger.debug('No logging function provided for process')):
        super(BatchFilter, self).__init__()
        self.prepare_logger = prepare_logger
        self.process_logger = process_logger


    def setup(self):
        pass

    def prepare(self, request):

        self.prepare_logger(request)

    def process(self, batch, request):

        self.process_logger(batch, request)

    @staticmethod
    def log_numpy_array_stats_after_process(array_key, *stats_or_meta, logging_prefix='', logger=_logger, level='debug'):

        def process_logger(batch, _):
            if array_key in batch.arrays:
                getattr(logger, level, None)('%s%s', logging_prefix, _LazyStatisticsString(batch.arrays[array_key].data, *stats_or_meta))
            else:
                getattr(logger, level, None)('Key %s not in batch %s', array_key, batch)

        return Log(prepare_logger=lambda batch: None, process_logger=process_logger)



