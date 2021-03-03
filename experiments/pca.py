"""
Script used to run novelty detection experiment using Principal
Component Analysis.

Uses:
    Module: PCABaseModule
    Model: IncrementalPCA
    Dataset: LunarAnalogueDataGenerator
"""
import pickle

from utils import tools
from datasets.lunar_analogue import LunarAnalogueDataGenerator
from models.incremental_pca import IncrementalPCA
from modules.pca_base_module import PCABaseModule


def main():
    default_config_file = 'configs/pca.yaml'
    config = tools.config_from_command_line(default_config_file)

    # Initialize datagenerator
    datagenerator = LunarAnalogueDataGenerator(config)

    # Initialize model. With n_component=None the number of features will
    # autoscale to the batch size
    model = IncrementalPCA(n_components=None)

    # Initialize experimental module
    module = PCABaseModule(datagenerator, model, config)

    # Incrementally fit training set with PCA
    datagenerator.setup('train')
    module.fit_pca(fast_dev_run=1)

    with open('models/saved-models/IncrementalPCA-LunarAnalogue-WholeImage.p', 'wb') as f:
        pickle.dump(module.model, f)


if __name__ == '__main__':
    main()
