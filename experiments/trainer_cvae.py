import torch
import pytorch_lightning as pl

from models.cvae import VariationalAutoEncoder
from datasets.mnist import MNISTDataModule
from datasets.emnist import EMNISTDataModule
from modules.cvae_base_module import CVAEBaseModule
from utils import tools
from torchsummary import summary

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
    datamodule = EMNISTDataModule(**data_params)
    datamodule.prepare_data()
    datamodule.setup('train')

    # Initialize model
    print('[INFO] Initializing model..')
    model = VariationalAutoEncoder(
        input_channels=1,
        input_height=28,
        input_width=28,
        latent_dims=10)

    # View a summary of the model
    summary(model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')), (1, 28, 28))

    # Initialize module
    print('[INFO] Initializing module..')
    module = CVAEBaseModule(
        model,
        train_size=datamodule.train_size,
        val_size=datamodule.val_size,
        batch_size=datamodule.batch_size,
        **module_params
    )
    # # Initialize loggers to monitor training and validation
    # print('[INFO] Initializing logger..')
    logger = pl.loggers.TestTubeLogger(
        exp_params['log_dir'],
        name=exp_params['name'],
        debug=False,
        create_git_tag=False
    )

    # Initialize the Trainer object
    print('[INFO] Initializing trainer..')
    trainer = pl.Trainer(
        weights_summary=None,
        gpus=1,
        logger=logger,
        max_epochs=100,
        check_val_every_n_epoch=1,
        callbacks=[
            pl.callbacks.early_stopping.EarlyStopping(monitor='loss', patience=5)
        ]
    )

    # Find learning rate
    if module_params['learning_rate'] == 'auto':
        lr_finder = trainer.tuner.lr_find(module, datamodule)
        module.lr = lr_finder.suggestion()
        print('[INFO] Using learning rate: ', module.lr)
        lr_finder_fig = lr_finder.plot(suggest=True, show=False)

    # Train the model
    print('[INFO] Training model...')
    trainer.fit(module, datamodule)

    # Some final saving once training is complete
    tools.save_object_to_version(lr_finder_fig, version=module.version, filename='lr-find.eps', **exp_params)
    tools.save_object_to_version(config, version=module.version, filename='configuration.yaml', **exp_params)


if __name__ == '__main__':
    DEFAULT_CONFIG_FILE = 'configs/cvae.yaml'
    main()
