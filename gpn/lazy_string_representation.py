class LazyStringRepresentation(object):

    def __init__(self, obj, pattern=None, *funcs):

        super(LazyStringRepresentation, self).__init__()
        self.obj = obj
        self.pattern = ' '.join(('%s',) * len(funcs)) if pattern is None else pattern
        self.funcs = funcs

        self.string_representation = None

    def __str__(self):
        if self.string_representation is None:
            self.string_representation = self.pattern % tuple(func(self.obj) for func in self.funcs)
        return self.string_representation
