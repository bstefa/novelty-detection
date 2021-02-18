import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision

from utils.tools import *

class CAEBaseModule(pl.LightningModule):
    '''
    The lighting module helps enforce best practices
    by keeping your code modular and abstracting the
    'engineering code' or boilerplate that new model
    require.
    '''
    def __init__(
            self,
            datamodule,
            model,
            params: dict
        ):
        super(CAEBaseModule, self).__init__()

        # Use datamodule and model
        self.dm = datamodule
        self.model = model

        # Set hparams (anything set as an hparam is automatically saved)
        self.hparams = params

        # Return a callable torch.nn.XLoss object
        # e.g. self._loss_function = self._handle_loss_function(params['loss_function'])
        self._loss_function = nn.MSELoss()

    def forward(self, x):
        return self.model.forward(x)

    def train_dataloader(self):
        return self.dm.train_dataloader()

    def val_dataloader(self):
        return self.dm.val_dataloader()

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.hparams.learning_rate
        )

    def on_train_start(self):
        print(f'Initializing with parameters:\n{self.hparams}\n')


    def on_epoch_start(self):
        random_integers = torch.randint(
            low=0, 
            high=self.hparams.num_train_batches,
            size=(3,)
        )
        self._random_train_steps = self.global_step + random_integers
        print(f'\nLogging images from batches: {self._random_train_steps.tolist()}\n')

    def training_step(self, x_in, batch_idx):

        x_out = self(x_in)
        loss = self._loss_function(x_out, x_in)

        images = {
            'x_in': x_in.detach(), # Tensor
            'x_out': x_out.detach() # Tensor
        }
        result = {
            'loss': loss
        }

        # Log some data
        if any([x == self.global_step for x in self._random_train_steps]):
            self._handle_image_logging(images, session='train')

        self.log_dict(result)
        return result # The returned object must contain a 'loss' key

    def validation_step(self, x_in, batch_idx):

        x_out = self(x_in)
        loss = self._loss_function(x_out, x_in)

        result = {
            'val_loss': loss
        }

        self.log_dict(result)
        return result

    def _handle_image_logging(self, images: dict, session: str='train'):

        if self.logger.version is None:
            return
        else:
            compute = {
                'x_in_01': unstandardize_batch(images['x_in']),
                'x_out_01': unstandardize_batch(images['x_out']),
                'error_map': get_error_map(images['x_in'], images['x_out'])
            }

            self._log_to_tensorboard(images, compute)
            self._log_images(compute)
        return


    def _log_to_tensorboard(self, result: dict, compute: dict):
        self.logger.experiment.add_image(
            f'x_in-{self.global_step}', 
            result['x_in'], 
            global_step=self.global_step, 
            dataformats='NCHW'
        )
        self.logger.experiment.add_image(
            f'x_in_unstandardized-{self.global_step}', 
            compute['x_in_01'], 
            global_step=self.global_step, 
            dataformats='NCHW'
        )
        self.logger.experiment.add_image(
            f'x_out-{self.global_step}', 
            result['x_out'], 
            global_step=self.global_step, 
            dataformats='NCHW'
        )
        self.logger.experiment.add_image(
            f'x_out_unstandardized-{self.global_step}', 
            compute['x_out_01'], 
            global_step=self.global_step, 
            dataformats='NCHW'
        )
        self.logger.experiment.add_image(
            f'metric_squared_error-{self.global_step}', 
            compute['error_map'], 
            global_step=self.global_step, 
            dataformats='NCHW'
        )
    
    def _log_images(self, compute: dict):

        logger_save_path = os.path.join(
            self.logger.save_dir, 
            self.logger.name, 
            f'version_{self.logger.version}'
        )

        if not os.path.exists(os.path.join(logger_save_path, 'images')):
            os.mkdir(os.path.join(logger_save_path, 'images'))

        rint = torch.randint(0, len(compute['x_in_01']), size=())
        for key in compute:
            torchvision.utils.save_image(
                compute[key][rint], 
                os.path.join(
                    logger_save_path,
                    'images',
                    f'{self.current_epoch}-{self.global_step-(self.hparams.num_train_batches*self.current_epoch)}-{key}.png'
                )
            )
