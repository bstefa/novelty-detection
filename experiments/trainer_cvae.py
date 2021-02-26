import pytorch_lightning as pl
from models.cvae import VariationalAutoEncoder

def main():
	# Set defaults
    DEFAULT_CONFIG_FILE = 'configs/trainer_cae_reference.yaml'
    config = tools.config_from_command_line(DEFAULT_CONFIG_FILE)

if __name__ == '__main__':
	main()