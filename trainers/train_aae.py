import torch
import time
import os
import pytorch_lightning as pl

from models import supported_models
from datasets import supported_datamodules
from modules.aae_base_module import AAEBaseModule
from utils import tools, callbacks, supported_preprocessing_transforms
from functools import reduce

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    # Load configs
    config = tools.load_config(DEFAULT_CONFIG_FILE)
    # Unpack configuration
    exp_params = config['experiment-parameters']
    data_params = config['data-parameters']
    module_params = config['module-parameters']

    # Catch a few early common bugs
    assert ('AAE' in exp_params['model']), \
        'Only accepts AAE-type models for training, check your configuration file.'
    if 'RegionExtractor' in data_params['preprocessing']:
        assert (data_params['use_nre_collation'] is True)

    # Set up preprocessing routine
    preprocessing_transforms = supported_preprocessing_transforms[data_params['preprocessing']]

    # Initialize datamodule
    datamodule = supported_datamodules[exp_params['datamodule']](
        data_transforms=preprocessing_transforms,
        **data_params)
    datamodule.prepare_data()
    datamodule.setup('train')

    # Initialize model with number of nodes equal to number of input pixels
    model = supported_models[exp_params['model']](
        datamodule.data_shape,
        latent_nodes=module_params['latent_nodes'])

    # Initialize module
    module = AAEBaseModule(
        model,
        batch_size=datamodule.batch_size,
        **module_params)

    # Initialize loggers to monitor training and validation
    logger = pl.loggers.TensorBoardLogger(
        exp_params['log_dir'],
        name=os.path.join(exp_params['datamodule'], exp_params['model']))

    # Initialize the Trainer object
    print('[INFO] Initializing trainer..')
    trainer = pl.Trainer(
        weights_summary=None,
        gpus=1,
        logger=logger,
        max_epochs=1000,
        check_val_every_n_epoch=1,
        callbacks=[
            pl.callbacks.early_stopping.EarlyStopping(
                monitor='val_r_loss',
                patience=5 if exp_params['patience'] is None else exp_params['patience']),
            pl.callbacks.GPUStatsMonitor(),
            pl.callbacks.ModelCheckpoint(
                monitor='val_r_loss',
                filename='{val_r_loss:.2f}-{epoch}',
                save_last=True),
            callbacks.AAEVisualization()
        ])

    # Train the model
    print('[INFO] Training model...')
    trainer.fit(module, datamodule)

    # Some final saving once training is complete
    tools.save_object_to_version(config, version=module.version, filename='configuration.yaml', **exp_params)
    tools.save_object_to_version(str(model), version=module.version, filename='model_summary.txt', **exp_params)


if __name__ == '__main__':
    DEFAULT_CONFIG_FILE = 'configs/aae/aae_simple_mnist.yaml'

    start = time.time()
    main()
    print(f'Training took {time.time() - start:.3f}s')
