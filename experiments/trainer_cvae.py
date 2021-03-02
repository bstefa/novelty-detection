import pytorch_lightning as pl
from models.cvae import VariationalAutoEncoder
from datasets.mnist import MNISTDataModule
from modules.cvae_base_module import CVAEBaseModule
from utils import tools

def main():
	# Set defaults
    DEFAULT_CONFIG_FILE = 'configs/trainer_cae_reference.yaml'
    config = tools.config_from_command_line(DEFAULT_CONFIG_FILE)

    datamodule = MNISTDataModule(config)

    model = VariationalAutoEncoder(input_channels=1, 
    							   input_height=28, 
    							   input_width=28, 
    							   latent_dims=10, 
    							   hidden_dims=[6, 12, 24])

    module = CVAEBaseModule(datamodule, model, config)

    #TODO: figure out configs

if __name__ == '__main__':
	main()