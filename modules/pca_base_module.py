import numpy as np
import matplotlib.pyplot as plt

from functools import reduce
from utils import tools

class PCABaseModule(object):
    '''
    Base module for PCA-based experiments
    '''
    def __init__(
            self,
            datagenerator,
            model,
            params: dict
        ):
        super(PCABaseModule, self).__init__()
        self.dg = datagenerator
        self.model = model
        self.hparams = params

    def fit_pca(self):
        for batch_nb, batch_in in enumerate(self.dg.create_generator('train')):
            self.model.partial_fit(self._flatten_batch(batch_in))
            print(batch_nb)

    def transform_pca(self):
        for batch_in in self.dg.create_generator('train'):
            self.model.transform(self._flatten_batch(batch_in))

    def _flatten_batch(self, batch_in):
        return batch_in.reshape(
            batch_in.shape[0],
            reduce(lambda x, y: x*y, batch_in.shape[1:])
        )

