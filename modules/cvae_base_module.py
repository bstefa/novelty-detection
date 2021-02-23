import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision

from utils.dtypes import *

class CVAEBaseModule(pl.LightningModule):
	def __init__(
			self,
			data_module,
			model,
			params: dict):
		super(CVAEBaseModule, self).__init__()

		self.dm = data_module
		self.model = model
		self.params = params
		self.curr_device = None

	def forward(self, x: Tensor, **kwargs) -> Tensor:
		return self.model(x, **kwargs)

	def training_step(self, batch, batch_idx, optimizer_idx=0):

		images, labels = batch
		self.curr_device = images.device

		results = self.forward(images, labels=labels)
		train_loss = self.model.loss_function(*results,
											M_N = self.params['batch_size']/ self.num_val_imgs,
                                            optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)
		
		self.logger.experiment.log({key: val.item() for key, val in train_loss.items()})

        return train_loss