import numpy as np
import matplotlib.pyplot as plt

from functools import reduce
from sklearn import model_selection
from utils import tools, losses


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
        self.dg = datagenerator
        self.model = model
        self.hparams = params

    def fit_pca(self, fast_dev_run: int=0):
        '''
        Function. Uses the class's datagenerator to fit the PCA model on the
        entire dataset.
        Returns self
        '''
        # TODO: Consider doing cross validation (say, 10-fold) on each batch
        for batch_nb, batch_tr_in in enumerate(self.dg.trainval_generator('train')):
            print(f'[BATCH {batch_nb}] Fitting...')
            self.model.partial_fit(self._flatten_batch(batch_tr_in))

            running_novelty_score = 0
            for batch_vl_nb, batch_vl_in in enumerate(self.dg.trainval_generator('val')):
                batch_vl_rd = self.model.transform(self._flatten_batch(batch_vl_in))
                batch_vl_rc = self.model.inverse_transform(batch_vl_rd).reshape(batch_vl_in.shape)

                # Run validation on batch
                for x_hat, x in zip(batch_vl_rc, batch_vl_in):
                    assert len(x.shape) == 3 and len(x_hat.shape), 'Only accepts H,W,C image-data'
                    running_novelty_score += losses.squared_error(
                        x,
                        x_hat,
                        show_plot=(True if batch_nb % 10 == 9 else False),
                        return_map=False
                    )
            print(f'[BATCH {batch_nb}] Validation novelty score {running_novelty_score/(len(batch_vl_in)*batch_vl_nb)}')

            if fast_dev_run > 0 and batch_nb == (fast_dev_run - 1):
                return self
                break
        return self

    def transform_pipeline(self, fast_dev_run: int=0):
        novelty_scores = []
        for batch_nb, (batch_rd, batch_in, batch_label) in enumerate(self.transform_generator()):
            print('Transform: ', batch_nb)
            # Compute batch-wise reconstruction
            batch_rc = self.inverse_transform(batch_rd).reshape(batch_in.shape)
            # Operate on each image-pair independently
            for x_nb, (x_hat, x) in enumerate(zip(batch_rc, batch_in)):
                assert len(x.shape) == 3, 'Can only operate on H,W,C image-data'
                # TODO: Consider using alternative losses here
                image_novelty_score = losses.squared_error(
                    x,
                    x_hat,
                    show_plot=(True if all([batch_nb % 10 == 0, x_nb == 0]) else False),
                    return_map=False
                )
                novelty_scores.append(image_novelty_score)

            if fast_dev_run > 0 and batch_nb == (fast_dev_run - 1):
                break

        # TODO: This is an ideal place to return a BatchStatistics object in the future
        return np.array(novelty_scores)

    def transform_generator(self):
        """
        Generator. Generates data and transforms it with a trained PCA model
        on the fly. Useful for iterating through the results and working with the transformations

        Returns a tuple generator (reduced data--flattened, input data--image shape)
        that can ce called as an iterable.
        """
        for batch_in, batch_label in self.dg.test_generator():
            yield self.model.transform(self._flatten_batch(batch_in)), batch_in, batch_label

    def inverse_transform(self, reduced_batch_in):
        '''
        Function.
        reduced_batch_in: Reduced data. Available after running .transform method on
        sklearn estimator. For this wrapper, it's used to convert a single batch
        back to the original data space in conjunction with the transform generator.
        '''
        return self.model.inverse_transform(reduced_batch_in)

    @property
    def components(self):
        return self.model.components_

    @property
    def explained_variance(self):
        return self.model.explained_variance_

    @property
    def explained_variance_ratio(self):
        return self.model.explained_variance_ratio_

    @property
    def meanvar(self):
        return self.model.meanvar_

    def _flatten_batch(self, batch_in):
        # Keep the batch dimension, take product of other three dimensions
        return batch_in.reshape(
            batch_in.shape[0],
            reduce(lambda x, y: x * y, batch_in.shape[1:])
        )