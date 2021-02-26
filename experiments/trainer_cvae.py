import pytorch_lightning as pl
from models.cvae import VariationalAutoEncoder
from dataset.mnist import MNISTDataModule
from modules.cvae_base_module import CVAEBaseModule

def main():
	# Set defaults
    DEFAULT_CONFIG_FILE = 'configs/trainer_cae_reference.yaml'
    config = tools.config_from_command_line(DEFAULT_CONFIG_FILE)

    datamodule = MNISTDataModule()

    model = VariationalAutoEncoder(1, 10, [6, 12, 24], 28, 28)

    module = CVAEBaseModule(datamodule, model, config)

    #TODO: figure out configs

if __name__ == '__main__':
	main()