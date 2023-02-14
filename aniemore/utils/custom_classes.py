"""
Custom classes for aniemore
"""
import torch
import gc
from contextlib import contextmanager

class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class BaseClass:
    def __enter__(self):
        return self

    def __del__(self):
        del self
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__del__()

    def close(self):
        self.__del__()


