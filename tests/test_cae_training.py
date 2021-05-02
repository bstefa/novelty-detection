import os
import glob
import unittest
import pytorch_lightning as pl

from pathlib import Path
from utils import tools, callbacks, supported_preprocessing_transforms
from modules.cae_base_module import CAEBaseModule
from datasets import supported_datamodules
from models import supported_models


class TestCAETraining(unittest.TestCase):

    def test_cae_config_compatability(self):

        config_paths = glob.glob('configs/cae/**')
        for pth in config_paths:
            print(f'Testing training for: {pth}')
            config = tools.load_config(pth)

            module = _test_training_pipeline(config)

            log_path = Path('tests') / \
                'test_logs' / \
                config['experiment-parameters']['datamodule'] / \
                config['experiment-parameters']['model'] / \
                f'version_{module.version}'

            self.assertTrue( (log_path / 'checkpoints').is_dir() )
            self.assertTrue( (log_path / 'configuration.yaml').is_file() )
            self.assertTrue( (log_path / 'model_summary.txt').is_file() )


def _test_training_pipeline(config):
    # Change log_dir for testing
    config['experiment-parameters']['log_dir'] = os.path.join('tests', 'test_logs')

    # Set up preprocessing routine
    preprocessing_transforms = supported_preprocessing_transforms[config['data-parameters']['preprocessing']]

    datamodule = supported_datamodules[config['experiment-parameters']['datamodule']](
        data_transforms=preprocessing_transforms,
        **config['data-parameters'])
    datamodule.prepare_data()
    datamodule.setup('train')

    model = supported_models[config['experiment-parameters']['model']](datamodule.data_shape)

    # Initialize experimental module
    module = CAEBaseModule(model, **config['module-parameters'])

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
            pl.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5 if config['experiment-parameters']['patience'] is None else config['experiment-parameters']['patience']),
            pl.callbacks.GPUStatsMonitor(),
            pl.callbacks.ModelCheckpoint(
                monitor='val_loss',
                filename='{val_loss:.2f}-{epoch}',
                save_last=True),
            callbacks.VisualizationCallback()
        ]
    )

    # Always run lr_finder for testing
    lr, lr_finder_fig = callbacks.learning_rate_finder(trainer, module, datamodule, num_training=25)
    module.lr = lr
    config['module-parameters']['learning_rate'] = module.lr

    # Train the model
    trainer.fit(module, datamodule)

    # Remove try-except block for testing
    tools.save_object_to_version(
        config, version=module.version, filename='configuration.yaml', **config['experiment-parameters'])
    tools.save_object_to_version(
        str(model), version=module.version, filename='model_summary.txt', **config['experiment-parameters'])
    if 'lr_finder_fig' in locals():
        tools.save_object_to_version(
            lr_finder_fig, version=module.version, filename='lr-find.eps', **config['experiment-parameters'])

    return module


if __name__ == '__main__':
    unittest.main()
