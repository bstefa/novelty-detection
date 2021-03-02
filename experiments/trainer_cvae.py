import pytorch_lightning as pl
from models.cvae import VariationalAutoEncoder
from datasets.mnist import MNISTDataModule
from modules.cvae_base_module import CVAEBaseModule
from utils import tools

def main():
	# Set defaults
    DEFAULT_CONFIG_FILE = 'configs/trainer_cae_reference.yaml'
    config = tools.config_from_command_line(DEFAULT_CONFIG_FILE)

    datamodule = MNISTDataModule()

    model = VariationalAutoEncoder(1, 28, 28, 10, [6, 12, 24])

    print(model)

    module = CVAEBaseModule(datamodule, model, config)

    #TODO: figure out configs

if __name__ == '__main__':
	main()