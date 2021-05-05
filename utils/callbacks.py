#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 12:36:29 2020

@author: brahste
"""
import os
import torch
import torchvision
import pytorch_lightning as pl
import torchvision.utils as vutils
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

from utils import tools


def learning_rate_finder(trainer, module, datamodule, **kwargs):
    lr_finder = trainer.tuner.lr_find(module, datamodule, **kwargs)
    suggested_lr = lr_finder.suggestion()
    lr_finder_fig = lr_finder.plot(suggest=True, show=False)

    print('[INFO] Using learning rate: ', suggested_lr)
    return suggested_lr, lr_finder_fig


def _log_to_tensorboard(result: dict, compute: dict, pl_module):
    pl_module.logger.experiment.add_image(
        f'batch_in-{pl_module.global_step}',
        result['batch_in'],
        global_step=pl_module.global_step,
        dataformats='NCHW'
    )
    pl_module.logger.experiment.add_image(
        f'batch_in_unstandardized-{pl_module.global_step}',
        compute['batch_in_01'],
        global_step=pl_module.global_step,
        dataformats='NCHW'
    )
    pl_module.logger.experiment.add_image(
        f'batch_rc-{pl_module.global_step}',
        result['batch_rc'],
        global_step=pl_module.global_step,
        dataformats='NCHW'
    )
    pl_module.logger.experiment.add_image(
        f'batch_rc_unstandardized-{pl_module.global_step}',
        compute['batch_rc_01'],
        global_step=pl_module.global_step,
        dataformats='NCHW'
    )
    pl_module.logger.experiment.add_image(
        f'metric_squared_error-{pl_module.global_step}',
        compute['error_map'],
        global_step=pl_module.global_step,
        dataformats='NCHW'
    )


def _log_images(compute: dict, pl_module):
    logger_save_path = os.path.join(
        pl_module.logger.save_dir,
        pl_module.logger.name,
        f'version_{pl_module.logger.version}'
    )

    if not os.path.exists(os.path.join(logger_save_path, 'images')):
        os.mkdir(os.path.join(logger_save_path, 'images'))

    rint = torch.randint(0, len(compute['batch_in_01']), size=())
    for key in compute:
        torchvision.utils.save_image(
            compute[key][rint],
            os.path.join(
                logger_save_path,
                'images',
                f'epoch={pl_module.current_epoch}-step={pl_module.global_step}-{key}.png'
            )
        )


def _handle_image_logging(images: dict, pl_module) -> None:
    assert pl_module.logger.version is not None, 'Logging cannot proceed without a version number.'

    batch_in_01 = tools.unstandardize_batch(images['batch_in'])
    batch_rc_01 = tools.unstandardize_batch(images['batch_rc'])

    compute = {
        'batch_in_01': batch_in_01,
        'batch_rc_01': batch_rc_01,
        'error_map': tools.get_error_map(batch_in_01, batch_rc_01)
    }

    _log_to_tensorboard(images, compute, pl_module)
    _log_images(compute, pl_module)


class NREDataShapeHandlerCallback(pl.callbacks.base.Callback):
    """This callback manages the shapes for data with a region dimension"""
    def __init__(self):
        pass

    @staticmethod
    def _handle_batch(batch):
        """Conducts an inplace operation to merge regions and batch size if necessary"""
        batch_in, _ = batch
        assert any(len(batch_in.shape) == s for s in (4, 5)), \
            f'Batch must have 4 or 5 dims, got {len(batch_in.shape)}'
        if len(batch_in.shape) == 5:
            batch_in = batch_in.view(-1, *batch_in.shape[2:])
        batch = (batch_in, _)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        self._handle_batch(batch)

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        self._handle_batch(batch)


class VisualizationCallback(pl.callbacks.base.Callback):
    def __init__(self):
        self._save_at_train_step = 0

    def on_epoch_start(self, trainer, pl_module):
        # Save images at second training step every epoch
        self._save_at_train_step = pl_module.global_step + 1

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if self._save_at_train_step == pl_module.global_step:
            batch_in, _ = pl_module.handle_batch(batch)
            batch_in = batch_in.detach()
            batch_rc = pl_module.forward(batch_in.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
            if trainer.datamodule.name == 'CuriosityDataModule':
                batch_in = batch_in[:, [2, 0, 1]]
                batch_rc = batch_rc[:, [2, 0, 1]]
            images = {
                'batch_in': batch_in,
                'batch_rc': batch_rc
            }
            _handle_image_logging(images, pl_module)


class AAEVisualization(pl.callbacks.base.Callback):
    def __init__(self):
        self._save_at_train_step = 0

    def on_epoch_start(self, trainer, pl_module):
        # Save images at second training step every epoch
        # (uses +4 because there are two dummy steps at the beginning of each epoch)
        self._save_at_train_step = pl_module.global_step + 4

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if self._save_at_train_step == pl_module.global_step:

            batch_in, _ = pl_module.handle_batch_shape(batch)
            batch_in = batch_in.detach()
            image_shape = batch_in.shape

            batch_in = batch_in.view(image_shape[0], -1)
            batch_lt = pl_module.encoder(batch_in.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
            batch_rc = pl_module.decoder(batch_lt)

            batch_in = batch_in.reshape(image_shape)
            batch_rc = batch_rc.reshape(image_shape)

            if trainer.datamodule.name == 'CuriosityDataModule':
                batch_in = batch_in[:, [2, 0, 1]]
                batch_rc = batch_rc[:, [2, 0, 1]]

            images = {
                'batch_in': batch_in,
                'batch_rc': batch_rc
            }
            _handle_image_logging(images, pl_module)


class VAEVisualization(pl.callbacks.base.Callback):
    def __init__(self):
        pass

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if batch_idx == 1 and pl_module.logger.version is not None:
            batch_in, _ = pl_module.handle_batch_shape(batch)
            batch_in = batch_in.detach().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

            batch_rc = pl_module.model.generate(batch_in)

            samples = pl_module.model.sample(num_samples=144)

            if trainer.datamodule.name == 'CuriosityDataModule':
                batch_in = batch_in[:, [2, 0, 1]]
                batch_rc = batch_rc[:, [2, 0, 1]]
                samples = samples[:, [2, 0, 1]]

            index = torch.randint(len(batch_in), size=(6,))
            self.sample_images(batch_in[index], batch_rc[index], samples, pl_module)

    @staticmethod
    def sample_images(batch_in, batch_rc, samples, pl_module):
        logger_save_path = os.path.join(
            pl_module.logger.save_dir,
            pl_module.logger.name,
            f'version_{pl_module.logger.version}',
            'images'
        )

        if not os.path.exists(logger_save_path):
            os.mkdir(logger_save_path)

        grid = torch.cat((batch_in, batch_rc))

        vutils.save_image(
            grid,
            os.path.join(
                logger_save_path,
                f'epoch={pl_module.current_epoch}-step={pl_module.global_step}-recons.png'),
            normalize=True,
            nrow=6)

        vutils.save_image(
            samples,
            os.path.join(
                logger_save_path,
                f'epoch={pl_module.current_epoch}-step={pl_module.global_step}-samples.png'),
            normalize=True,
            nrow=12)
