"""
Script used to run novelty detection experiment using reference Convolutional
Autoencoder (See ISAIRAS 2020 paper for details). This script trains and
serializes the model for future evaluation.

Uses:
    Module: CAEBaseModule
    Model: ReferenceCAE
    Dataset: LunarAnalogueDataGenerator
    Configuration: reference_cae.yaml
"""
import pytorch_lightning as pl

from utils import tools, callbacks
from modules.cae_base_module import CAEBaseModule
from models.reference_cae import ReferenceCAE
from datasets.lunar_analogue import LunarAnalogueDataModule


def main():
    # Set defaults
    DEFAULT_CONFIG_FILE = 'configs/reference_cae.yaml'
    config = tools.config_from_command_line(default_config_file)
    # Unpack configuration
    exp_params = config['experiment-parameters']
    data_params = config['data-parameters']
    module_params = config['module-parameters']

    # Initialize datamodule
    datamodule = LunarAnalogueDataModule(**data_params)
    datamodule.setup('train')

    # Initialize model
    model = ReferenceCAE()

    # Initialize experimental module
    module = CAEBaseModule(model, **module_params)

    # Initialize loggers to monitor training and validation
    logger = pl.loggers.TensorBoardLogger(
        exp_params['log_dir'],
        name=exp_params['name']
    )

    # Initialize the Trainer object
    trainer = pl.Trainer(
        gpus=1,
        logger=logger,
        max_epochs=100,
        callbacks=[
            pl.callbacks.EarlyStopping(monitor='val_loss', patience=4),
            pl.callbacks.GPUStatsMonitor(),
            pl.callbacks.ModelCheckpoint(monitor='val_loss', filename='{val_loss:.2f}-{epoch}', save_last=True),
            # callbacks.SimpleHyperparameterSaver(exp_params['log_dir'], exp_params['name'], 'hyperparameters.yaml'),
            callbacks.VisualizationCallback()
        ]
    )
    # Find learning rate
    if module_params['learning_rate'] == 'auto':
        lr_finder = trainer.tuner.lr_find(module, datamodule)
        module.lr = lr_finder.suggestion()
        print('[INFO] Using learning rate: ', module.lr)
        config['module-parameters']['learning_rate'] = module.lr  # Set learning rate to config for reference
        lr_finder_fig = lr_finder.plot(suggest=True, show=False)

    # Train the model
    trainer.fit(module, datamodule)

    # Some final saving once training is complete
    tools.save_object_to_version(lr_finder_fig, version=module.version, filename='lr-find.eps', **exp_params)
    tools.save_object_to_version(config, version=module.version, filename='configuration.yaml', **exp_params)


if __name__ == '__main__':
    main()
