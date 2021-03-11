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
    default_config_file = 'configs/reference_cae.yaml'
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
        auto_lr_find=(True if module_params['learning_rate'] == 0. else False),
        callbacks=[
            pl.callbacks.EarlyStopping(monitor='val_loss', patience=4),
            pl.callbacks.GPUStatsMonitor(),
            pl.callbacks.ModelCheckpoint(monitor='val_loss', filename='{val_loss:.2f}-{epoch}', save_last=True),
            callbacks.SimpleHyperparameterSaver(exp_params['log_dir'], exp_params['name'], 'hyperparameters.yaml'),
            callbacks.VisualizationCallback()
        ]
    )
    # Find the learning rate automatically
    trainer.tune(module, datamodule)

    # Train the model
    trainer.fit(module, datamodule)

    # Save the imported configuration only after the model has completed training
    tools.save_dictionary_to_current_version(
        exp_params['log_dir'],
        exp_params['name'],
        'configuration.yaml',
        config
    )


if __name__ == '__main__':
    main()
