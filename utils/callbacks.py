#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 12:36:29 2020

@author: brahste
"""

import torch
import pytorch_lightning as pl

class SaveStateDictCallback(pl.callbacks.base.Callback):
    '''Callback for saving and visualizing autoencoder training process'''
    def on_train_end(self, trainer, pl_module):
        print('training has ended!')
        torch.save(pl_module.state_dict(), 'logs/CAE_.pt')        