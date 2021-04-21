"""
Defines base module for PCA experiments.
Class must be instantiated with an sklearn model that has the following 
methods: .partial_fit, .transform, .inverse_transform
"""
import numpy as np

from utils import losses, tools


class PCABaseModule(object):
    def __init__(self, model):
        self.model = model

    def fit_pipeline(self, datamodule, fast_dev_run: int=0):
        """
        Uses the class's training and validation generator to iteratively fit/validate the PCA model.
        """
        if 'module' in datamodule.name.lower():
            X, y = datamodule.split(train=True)
            self.model.fit(self._reshape_batch(X))

        elif 'generator' in datamodule.name.lower():
            novelty_scores = []

            # Grab a batch from the training set
            for batch_nb, batch_tr_in in enumerate(self.dg.trainval_generator('train')):
                print(f'[BATCH {batch_nb}] Fitting...')
                self.model.partial_fit(self._reshape_batch(batch_tr_in))

                batch_novelty_scores = []
                # Grab a batch from the validation set
                for batch_vl_in in self.dg.trainval_generator('val'):
                    batch_vl_rd = self.model.transform(self._reshape_batch(batch_vl_in))
                    batch_vl_rc = self.model.inverse_transform(batch_vl_rd).reshape(batch_vl_in.shape)

                    # Run validation on batch
                    for x_nb, (x_rc, x_in) in enumerate(zip(batch_vl_rc, batch_vl_in)):
                        assert len(x_rc.shape) == 3, 'Only accepts H,W,C image-data'
                        image_novelty_score = losses.squared_error(
                            x_in,
                            x_rc,
                            show_plot=(True if all([batch_nb % 10 == 0, x_nb == 4]) else False),
                            return_map=False
                        )
                        batch_novelty_scores.append(image_novelty_score)

                print(f'[BATCH {batch_nb}] Validation novelty score {np.mean(batch_novelty_scores)}')
                novelty_scores.append(batch_novelty_scores)

                if fast_dev_run > 0 and batch_nb == (fast_dev_run - 1):
                    break

        else:
            raise ValueError('Something went wrong when loading the datamodule...')

    def transform_pipeline(self, datamodule, fast_dev_run: int=0):
        """
        Uses the class's test generator to iteratively transform and evaluate novelty scores for the
        trained PCA model.
        """
        if 'module' in datamodule.name.lower():
            x_test_in, y_test_in = datamodule.split(train=False)
            x_test_reduced = self.model.transform(self._reshape_batch(x_test_in))
            x_test_rc = self.model.inverse_transform(x_test_reduced).reshape(x_test_in.shape)

            novelty_scores = []
            for x_nb, (x_in, x_rc) in enumerate(zip(x_test_in, x_test_rc)):
                novelty_score = losses.squared_error(x_in, x_rc, show_plot=False, return_map=False)
                novelty_scores.append(novelty_score)

            return novelty_scores

        elif 'generator' in datamodule.name.lower():
            novelty_scores = []
            novelty_labels = []
            for batch_nb, (batch_in, batch_lbl) in enumerate(self.dg.test_generator()):

                # Compute batch-wise reduction and reconstruction
                batch_rd = self.model.transform(self._reshape_batch(batch_in))
                batch_rc = self.model.inverse_transform(batch_rd).reshape(batch_in.shape)

                # Operate on each image-pair independently, looping over the batch
                for x_nb, (x_rc, x_in, x_lbl) in enumerate(zip(batch_rc, batch_in, batch_lbl)):
                    assert len(x_rc.shape) == 3, 'Can only operate on H,W,C image-data'
                    # TODO: Consider using alternative losses here
                    x_novelty_score = losses.squared_error(
                        x_in,
                        x_rc,
                        show_plot=(True if all([batch_nb % 10 == 0, x_nb == 0]) else False),
                        return_map=False
                    )
                    novelty_scores.append(x_novelty_score)
                    novelty_labels.append(x_lbl)

                print(f'[BATCH {batch_nb}] Transforming...')

                if fast_dev_run > 0 and batch_nb == (fast_dev_run - 1):
                    break

            return novelty_scores, novelty_labels

    def _reshape_batch(self, batch_in):
        # Keep the batch/n_sample dimension, take product of other remaining dimensions
        return batch_in.reshape(batch_in.shape[0], -1)