"""
Script used to run novelty detection experiment using reference Convolutional
Autoencoder (See ISAIRAS 2020 paper for details). This script trains and
serializes the model for future evaluation.

Uses:
    Module: CAEBaseModule
    Model: ReferenceCAE
    Dataset: LunarAnalogueDataGenerator
    Configuration: cae_baseline_lunar_analogue.yaml
"""
import os
import pytorch_lightning as pl
import logging

from utils import tools, callbacks
from modules.cae_base_module import CAEBaseModule
from datasets import supported_datamodules
from models import supported_models


def main():
    # Set defaults
    config = tools.load_config(DEFAULT_CONFIG_FILE)
    # Unpack configuration
    exp_params = config['experiment-parameters']
    data_params = config['data-parameters']
    module_params = config['module-parameters']
    assert ('CAE' in exp_params['model']), \
        'Only accepts CAE-type models for training, check your configuration file.'

    # What I should do is set up the data transforms directly in the training script, this can
    # be outsourced to a configuration file later.
    preprocessing_transforms = Compose([

    ])

    # Initialize datamodule (see datasets/__init__.py for details)
    datamodule = supported_datamodules[exp_params['datamodule']](**data_params)
    datamodule.prepare_data()
    datamodule.setup('train')
    logging.debug(datamodule.data_shape)

    # Note that Pytorch uses the convention of shaping data as [..., C, H, W] as opposed
    # to [..., H, W, C]. When using a region extractor, the shape of the returned data may
    # be [n_regions, C, H, W]. The BaseDataModule class always returns the last three channels
    # when calling '.data_shape', the actual batch shape discrepancy is is handled implicitly
    # in the DataIntegrityCallback
    model = supported_models[exp_params['model']](datamodule.data_shape)

    # Initialize experimental module
    module = CAEBaseModule(model, **module_params)

    # Initialize loggers to monitor training and validation
    logger = pl.loggers.TensorBoardLogger(
        exp_params['log_dir'],
        name=os.path.join(exp_params['datamodule'], exp_params['model']))

    # Initialize the Trainer object
    trainer = pl.Trainer(
        gpus=1,
        logger=logger,
        max_epochs=1000,
        weights_summary=None,
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5 if exp_params['patience'] is None else exp_params['patience']),
            pl.callbacks.GPUStatsMonitor(),
            pl.callbacks.ModelCheckpoint(
                monitor='val_loss',
                filename='{val_loss:.2f}-{epoch}',
                save_last=True),
            callbacks.VisualizationCallback()
        ]
    )
    # Find learning rate and set values
    if module_params['learning_rate'] is None:
        lr, lr_finder_fig = callbacks.learning_rate_finder(trainer, module, datamodule)
        module.lr = lr
        config['module-parameters']['learning_rate'] = module.lr

    # Train the model
    trainer.fit(module, datamodule)

    # Some final saving once training is complete
    try:
        tools.save_object_to_version(config, version=module.version, filename='configuration.yaml', **exp_params)
        tools.save_object_to_version(str(model), version=module.version, filename='model_summary.txt', **exp_params)
        if 'lr_finder_fig' in locals():
            tools.save_object_to_version(lr_finder_fig, version=module.version, filename='lr-find.eps', **exp_params)
    except TypeError as e:
        print(e)
        pass


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    DEFAULT_CONFIG_FILE = 'configs/cae/cae_baseline_lunar_analogue.yaml'
    main()
