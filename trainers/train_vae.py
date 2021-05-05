import os
import time
import logging
import torch
import pytorch_lightning as pl

from models import supported_models
from datasets import supported_datamodules
from modules.vae_base_module import VAEBaseModule
from utils import tools, callbacks, supported_preprocessing_transforms
# from torchsummary import summary


def main():
    # Load configs
    print('[INFO] Loading configs...')
    config = tools.load_config(DEFAULT_CONFIG_FILE)
    # Unpack configuration
    exp_params = config['experiment-parameters']
    data_params = config['data-parameters']
    module_params = config['module-parameters']

    # Catch few early, common bugs
    assert ('VAE' in exp_params['model']), \
        'Only accepts VAE-type models for training, check your configuration file.'
    if 'RegionExtractor' in data_params['preprocessing']:
        assert (data_params['use_nre_collation'] is True)
    else:
        assert (data_params['use_nre_collation'] is False)

    # Set up preprocessing routine
    preprocessing_transforms = supported_preprocessing_transforms[data_params['preprocessing']]

    # Initialize datamodule
    print('[INFO] Initializing datamodule..')
    datamodule = supported_datamodules[exp_params['datamodule']](
        data_transforms=preprocessing_transforms,
        **data_params)
    datamodule.prepare_data()
    datamodule.setup('train')

    # Initialize model
    print('[INFO] Initializing model..')
    model = supported_models[exp_params['model']](datamodule.data_shape, **module_params)

    # Initialize module
    print('[INFO] Initializing module..')
    module = VAEBaseModule(
        model,
        train_size=datamodule.train_size,
        val_size=datamodule.val_size,
        batch_size=datamodule.batch_size,
        **module_params)

    # Initialize loggers to monitor training and validation
    print('[INFO] Initializing logger..')
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
            pl.callbacks.EarlyStopping(
                monitor='val_elbo_loss',
                patience=5 if exp_params['patience'] is None else exp_params['patience']),
            pl.callbacks.ModelCheckpoint(
                monitor='val_elbo_loss',
                filename='{val_elbo_loss:.2f}-{epoch}',
                save_last=True),
            pl.callbacks.GPUStatsMonitor(),
            callbacks.VAEVisualization()
        ])

    # Find learning rate
    if module_params['learning_rate'] is None:
        lr, lr_finder_fig = callbacks.learning_rate_finder(trainer, module, datamodule)
        module.lr = lr
        config['module-parameters']['learning_rate'] = module.lr

    # Train the model
    print('[INFO] Training model...')
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
    DEFAULT_CONFIG_FILE = 'configs/vae/vae_simple_mnist.yaml'

    start = time.time()
    main()
    print(f'Training took {time.time() - start:.3f}s')
