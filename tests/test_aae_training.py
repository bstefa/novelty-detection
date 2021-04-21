import os
import glob
import unittest
import pytorch_lightning as pl

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

from pathlib import Path
from functools import reduce
from utils import tools, callbacks
from modules.aae_base_module import AAEBaseModule
from datasets import supported_datamodules
from models import supported_models


class TestCAETraining(unittest.TestCase):

    def test_aae_config_compatability(self):

        def test_training_pipeline(config):
            # Change log_dir for testing
            config['experiment-parameters']['log_dir'] = os.path.join('tests', 'test_logs')

            datamodule = supported_datamodules[config['experiment-parameters']['datamodule']](
                **config['data-parameters'])
            datamodule.prepare_data()
            datamodule.setup('train')

            model = supported_models[config['experiment-parameters']['model']](
                in_nodes=reduce(lambda x, y: x*y, datamodule.data_shape),
                latent_nodes=config['module-parameters']['latent_nodes'])

            # Initialize experimental module
            module = AAEBaseModule(model, batch_size=datamodule.batch_size, **config['module-parameters'])

            # Initialize loggers to monitor training and validation
            logger = pl.loggers.TensorBoardLogger(
                config['experiment-parameters']['log_dir'],  # Temp location for dummy logs
                name=os.path.join(config['experiment-parameters']['datamodule'], config['experiment-parameters']['model']))

            # Initialize the Trainer object
            trainer = pl.Trainer(
                gpus=1,
                logger=logger,
                max_epochs=1,
                weights_summary=None,
                callbacks=[
                    pl.callbacks.early_stopping.EarlyStopping(
                        monitor='val_r_loss',
                        patience=5 if config['experiment-parameters']['patience'] is None else config['experiment-parameters']['patience']),
                    pl.callbacks.GPUStatsMonitor(),
                    pl.callbacks.ModelCheckpoint(
                        monitor='val_r_loss',
                        filename='{val_r_loss:.2f}-{epoch}',
                        save_last=True),
                    callbacks.AAEVisualization()
                ])

            # Train the model
            trainer.fit(module, datamodule)

            # Remove try-except block for testing
            tools.save_object_to_version(
                config, version=module.version, filename='configuration.yaml', **config['experiment-parameters'])
            tools.save_object_to_version(
                str(model), version=module.version, filename='model_summary.txt', **config['experiment-parameters'])

            return module

        config_paths = glob.glob('configs/aae/*mnist*')
        for pth in config_paths:
            logging.info(f"Testing training for: {pth}")
            config = tools.load_config(pth)
            module = test_training_pipeline(config)

            log_path = Path('tests') / \
                'test_logs' / \
                config['experiment-parameters']['datamodule'] / \
                config['experiment-parameters']['model'] / \
                f'version_{module.version}'
            logging.info(log_path)

            self.assertTrue( (log_path / 'checkpoints').is_dir() )
            self.assertTrue( (log_path / 'configuration.yaml').is_file() )
            self.assertTrue( (log_path / 'model_summary.txt').is_file() )


if __name__ == '__main__':
    unittest.main()
