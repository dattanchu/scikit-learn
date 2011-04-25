from ..base import BaseEstimator
"""
Base class for all ensemble classes
"""


class BaseEnsemble(BaseEstimator):

    def __init__(self, estimator, **params):

        self.estimator = estimator
        if not issubclass(estimator, BaseEstimator):
            raise TypeError("estimator must be a subclass of BaseEstimator")
        self.params = params
        self.estimators = []

    def __nonzero__(self):

        return len(self) > 0
    
    def __len__(self):

        return len(self.estimators)

    def __getitem__(self, index):

        return self.estimators[index]

    def __setitem__(self, index, thing):

        self.estimators[index] = thing

    def __delitem__(self, index):

        del self.estimators[index]

    def append(self, thing):

        return self.estimators.append(thing)

    def __getattr__(self, name):
        
        try:
            return super(BaseEnsemble, self).__getattr__(self, name)
        except AttributeError: pass
        if not self:
            raise AttributeError("%s has no attribute %s"% \
                (self.estimator.__name__, name))
        def func(self, *args, **kwargs):
            norm = 0.
            _return = None
            for weight, estimator in self:
                norm += weight
                if _return is None:
                    _return = weight * estimator.__getattr__(name)(*args, **kwargs)
                else:
                    _return += weight * estimator.__getattr__(name)(*args, **kwargs)
            return _return / norm if norm > 0 else None
        return func
