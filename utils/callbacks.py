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

from utils import tools


def learning_rate_finder(trainer, module, datamodule):
    lr_finder = trainer.tuner.lr_find(module, datamodule)
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


def _handle_image_logging(images: dict, pl_module):
    assert pl_module.logger.version is not None, 'Logging cannot proceed without a verison number.'

    batch_in_01 = tools.unstandardize_batch(images['batch_in'])
    batch_rc_01 = tools.unstandardize_batch(images['batch_rc'])
    compute = {
        'batch_in_01': batch_in_01,
        'batch_rc_01': batch_rc_01,
        'error_map': tools.get_error_map(batch_in_01, batch_rc_01)
    }

    _log_to_tensorboard(images, compute, pl_module)
    _log_images(compute, pl_module)


class VisualizationCallback(pl.callbacks.base.Callback):
    def __init__(self):
        self._save_at_train_step = 0

    def on_epoch_start(self, trainer, pl_module):
        # Save images at second training step every epoch
        self._save_at_train_step = pl_module.global_step + 1

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if self._save_at_train_step == pl_module.global_step:
            batch_in, _ = batch
            batch_in.detach_()
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
        self._save_at_train_step = pl_module.global_step + 4

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if self._save_at_train_step == pl_module.global_step:
            batch_in, _ = batch
            image_shape = batch_in.shape

            batch_in = batch_in.view(image_shape[0], -1)
            batch_rc = pl_module.decoder(pl_module.encoder(batch_in.to(pl_module.device)))

            images = {
                'batch_in': batch_in.detach().view(*image_shape).transpose(2, 3),
                'batch_rc': batch_rc.detach().view(*image_shape).transpose(2, 3)
            }
            _handle_image_logging(images, pl_module)

