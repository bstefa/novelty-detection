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

from utils import tools
from modules.cae_base_module import CAEBaseModule
from models.reference_cae import ReferenceCAE
from datasets.lunar_analogue import LunarAnalogueDataModule


def main():
    # Set defaults
    default_config_file = 'configs/reference_cae.yaml'
    config = tools.config_from_command_line(default_config_file)

    # Initialize datamodule
    datamodule = LunarAnalogueDataModule(config)
    datamodule.setup('fit')

    # Add information about the dataset to the experimental parameters
    config['num_train_samples'] = datamodule.num_train_samples
    config['num_val_samples'] = datamodule.num_val_samples
    config['num_train_batches'] = int(config['num_train_samples'] / config['batch_size'])

    # Initialize model
    model = ReferenceCAE(config)

    # Initialize experimental module
    module = CAEBaseModule(datamodule, model, config)

    # Initialize loggers to monitor training and validation
    logger = pl.loggers.TensorBoardLogger(
        config['log_directory'],
        name=config['experiment_name']
    )

    # Initialize the Trainer object
    trainer = pl.Trainer(
        gpus=1,
        logger=logger,
        max_epochs=50,
        # auto_lr_find=(True if config['learning_rate'] is None else False),
        callbacks=[
            pl.callbacks.early_stopping.EarlyStopping(monitor='val_loss', patience=10)
        ]
    )

    # lr_finder = trainer.tuner.lr_find(module)
    # module.hparams.learning_rate = lr_finder.suggestion()
    # print('Learning rate: ', module.hparams.learning_rate)
    # fig = lr_finder.plot(suggest=True)
    # fig.show()
    #
    # tuner = pl.tuner.tuning.Tuner(trainer)
    # module.hparams.batch_size = tuner.scale_batch_size(module)
    # print('Batch size: ', module.hparams.batch_size)

    # Train the model
    trainer.fit(module)


if __name__ == '__main__':
    main()
