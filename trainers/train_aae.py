import torch
import os
import pytorch_lightning as pl

from models.aae_simple import SimpleAAE
from datasets import supported_datamodules
from modules.aae_base_module import AAEBaseModule
from utils import tools, callbacks
from functools import reduce


def main():
    # Load configs
    print('[INFO] Loading configs...')
    config = tools.config_from_command_line(DEFAULT_CONFIG_FILE)
    # Unpack configuration
    exp_params = config['experiment-parameters']
    data_params = config['data-parameters']
    module_params = config['module-parameters']

    # Initialize datamodule
    print('[INFO] Initializing datamodule..')
    datamodule = supported_datamodules[exp_params['datamodule']](**data_params)
    datamodule.prepare_data()
    datamodule.setup('train')

    # Initialize model
    print('[INFO] Initializing model..')
    model = SimpleAAE(reduce(lambda x, y: x*y, datamodule.shape), **module_params)

    # Initialize module
    print('[INFO] Initializing module..')
    module = AAEBaseModule(
        model,
        train_size=datamodule.train_size,
        val_size=datamodule.val_size,
        batch_size=datamodule.batch_size,
        **module_params)

    # Initialize loggers to monitor training and validation
    print('[INFO] Initializing logger..')
    logger = pl.loggers.TensorBoardLogger(
        exp_params['log_dir'],
        name=os.path.join(exp_params['name'], exp_params['datamodule']))

    # Initialize the Trainer object
    print('[INFO] Initializing trainer..')
    trainer = pl.Trainer(
        weights_summary=None,
        gpus=1,
        logger=logger,
        max_epochs=100,
        check_val_every_n_epoch=1,
        callbacks=[
            pl.callbacks.early_stopping.EarlyStopping(monitor='val_r_loss', patience=5),
            pl.callbacks.ModelCheckpoint(monitor='val_r_loss', filename='{val_r_loss:.2f}-{epoch}', save_last=True),
            callbacks.AAEVisualization()
        ])

    # Train the model
    print('[INFO] Training model...')
    trainer.fit(module, datamodule)

    # Some final saving once training is complete
    tools.save_object_to_version(config, version=module.version, filename='configuration.yaml', **exp_params)


if __name__ == '__main__':
    DEFAULT_CONFIG_FILE = 'configs/aae/aae_simple_mnist.yaml'
    main()
