import torch
import pytorch_lightning as pl
from models.cvae import VariationalAutoEncoder
from datasets.mnist import MNISTDataModule
from modules.cvae_base_module import CVAEBaseModule
from utils import tools
from torchsummary import summary
from pytorch_lightning.loggers import TestTubeLogger

def main():
    # Load configs
    print("Loading configs...")
    DEFAULT_CONFIG_FILE = 'configs/cvae_trainer.yaml'
    config = tools.config_from_command_line(DEFAULT_CONFIG_FILE)

    # Initialize device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Initialize datamodule
    print("Initializing datamodule..")
    datamodule = MNISTDataModule(config)
    datamodule.setup()

    # Initialize model
    print("Initializing model..")
    model = VariationalAutoEncoder(input_channels=1, 
                                   input_height=28, 
                                   input_width=28, 
                                   latent_dims=10)

    model.to(device)

    summary(model, (1, 28, 28))

    # Initialize module
    print("Initializing module..")
    module = CVAEBaseModule(datamodule, model, config)

    # Initialize loggers to monitor training and validation
    print("Initializing logger..")
    # tb_logger = pl.loggers.TensorBoardLogger(
    #     config['log_directory'],
    #     name=config['experiment_name']
    # )

    tt_logger = TestTubeLogger(
        save_dir=config['log_directory'],
        name=config['experiment_name'],
        debug=False,
        create_git_tag=False,
    )
    # Initialize the Trainer object
    print("Initializing trainer..")
    trainer = pl.Trainer(
        gpus=1,
        logger=tt_logger,
        max_epochs=10,
        # auto_lr_find=(True if config['learning_rate'] is None else False),
        callbacks=[
            pl.callbacks.early_stopping.EarlyStopping(monitor='loss', patience=10)
        ]
    )

    # Find learning rate
    print("Finding optimal learning rate and batch size..")
    lr_finder = trainer.tuner.lr_find(module)
    module.hparams.learning_rate = lr_finder.suggestion()
    print('Learning rate: ', module.hparams.learning_rate)
    fig = lr_finder.plot(suggest=True)
    fig.show()

    tuner = pl.tuner.tuning.Tuner(trainer)
    module.hparams.batch_size = tuner.scale_batch_size(module)
    print('Batch size: ', module.hparams.batch_size)

    # Train the model
    print("Training model")
    trainer.fit(module)

if __name__ == '__main__':
    main()