"""
Script used to run novelty detection experiment using reference Convolutional
Autoencoder (See ISAIRAS 2020 paper for details). This script trains and
serializes the model for future evaluation.

Uses:
    Module: CAEBaseModule
    Model: ReferenceCAE
    Dataset: LunarAnalogueDataGenerator
    Configuration: reference_cae_lunar_analogue.yaml
"""
import os
import pytorch_lightning as pl

from utils import tools, callbacks
from modules.cae_base_module import CAEBaseModule
from models.reference_cae import ReferenceCAE
from datasets import supported_datamodules


def main():
    # Set defaults
    config = tools.config_from_command_line(DEFAULT_CONFIG_FILE)
    # Unpack configuration
    exp_params = config['experiment-parameters']
    data_params = config['data-parameters']
    module_params = config['module-parameters']

    # Initialize datamodule (see datasets/__init__.py for details)
    datamodule = supported_datamodules[exp_params['datamodule']](**data_params)
    datamodule.prepare_data()
    datamodule.setup('train')
    print(datamodule.shape)

    # Initialize model with the number of channels in the data (note that torch uses
    # the convention of shaping data as [C, H, W] as opposed to the usual [H, W, C]
    model = ReferenceCAE(in_shape=datamodule.shape)

    # Initialize experimental module
    module = CAEBaseModule(model, **module_params)

    # Initialize loggers to monitor training and validation
    logger = pl.loggers.TensorBoardLogger(
        exp_params['log_dir'],
        name=os.path.join(exp_params['name'], exp_params['datamodule']))

    # Initialize the Trainer object
    trainer = pl.Trainer(
        gpus=1,
        logger=logger,
        max_epochs=100,
        callbacks=[
            pl.callbacks.EarlyStopping(monitor='val_loss', patience=4),
            pl.callbacks.GPUStatsMonitor(),
            pl.callbacks.ModelCheckpoint(monitor='val_loss', filename='{val_loss:.2f}-{epoch}', save_last=True),
            callbacks.VisualizationCallback()
        ]
    )
    # Find learning rate and set values
    if module_params['learning_rate'] is None:
        lr, lr_finder_fig = callbacks.learning_rate_finder(trainer, module, datamodule)
        module.lr = lr
        config['module-parameters']['learning_rate'] = module.lr
        # lr_finder = trainer.tuner.lr_find(module, datamodule)
        # module.lr = lr_finder.suggestion()
        # print('[INFO] Using learning rate: ', module.lr)
        # config['module-parameters']['learning_rate'] = module.lr  # Replace 'auto' with actual learning rate
        # lr_finder_fig = lr_finder.plot(suggest=True, show=False)

    # Train the model
    trainer.fit(module, datamodule)

    # Some final saving once training is complete
    if 'lr_finder_fig' in locals():
        tools.save_object_to_version(lr_finder_fig, version=module.version, filename='lr-find.eps', **exp_params)
    tools.save_object_to_version(config, version=module.version, filename='configuration.yaml', **exp_params)


if __name__ == '__main__':
    DEFAULT_CONFIG_FILE = 'configs/cae/reference_cae_lunar_analogue.yaml'
    main()
