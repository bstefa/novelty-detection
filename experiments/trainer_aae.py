# import sys
# sys.path.append('.')

import torch
import pytorch_lightning as pl

from models.simple_aae import SimpleAAE
from datasets import supported_datamodules
from modules.cvae_base_module import CVAEBaseModule
from utils import tools
from torchsummary import summary
from functools import reduce


DEFAULT_CONFIG_FILE = 'configs/aae/aae_emnist.yaml'


# Load configs
print('[INFO] Loading configs...')
config = tools.config_from_command_line(DEFAULT_CONFIG_FILE)
# Unpack configuration
exp_params = config['experiment-parameters']
data_params = config['data-parameters']
module_params = config['module-parameters']

#%%

# Initialize datamodule
print('[INFO] Initializing datamodule..')
datamodule = supported_datamodules[exp_params['datamodule']](**data_params)
datamodule.prepare_data()
datamodule.setup('train')

print(reduce(lambda x, y: x*y, datamodule.shape))
#%%

# Initialize model
print('[INFO] Initializing model..')
model = SimpleAAE(datamodule.shape, **module_params)

# View a summary of the model
summary(model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')), datamodule.shape)

# Initialize module
print('[INFO] Initializing module..')
module = AAEBaseModule(
    model,
    train_size=datamodule.train_size,
    val_size=datamodule.val_size,
    batch_size=datamodule.batch_size,
    **module_params)

# Initialize the Trainer object
print('[INFO] Initializing trainer..')
trainer = pl.Trainer(
    weights_summary=None,
    gpus=1,
    max_epochs=100,
    check_val_every_n_epoch=1,
    callbacks=[
        pl.callbacks.early_stopping.EarlyStopping(monitor='loss', patience=5)
    ])

# Train the model
print('[INFO] Training model...')
trainer.fit(module, datamodule)

