"""
Script used to run novelty detection experiment using incremental Principal
Component Analysis. This script trains and serializes the model for future
evaluation.

Uses:
    Module: PCABaseModule
    Model: IncrementalPCA
    Dataset: LunarAnalogueDataGenerator
    Configuration: incremental_pca.yaml
"""
import pickle

from utils import tools
from datasets.lunar_analogue import LunarAnalogueDataGenerator
from models.incremental_pca import IncrementalPCA
from modules.pca_base_module import PCABaseModule


def main():
    default_config_file = 'configs/incremental_pca.yaml'
    # Optionally accepts configuration file specified on command line
    config = tools.config_from_command_line(default_config_file)

    # Initialize datagenerator
    datagenerator = LunarAnalogueDataGenerator(config, stage='train')

    # Initialize model. With n_component=None the number of features will autoscale to the batch size
    model = IncrementalPCA(n_components=None)

    # Initialize experimental module
    module = PCABaseModule(datagenerator, model, config)

    # Incrementally fit training set with PCA
    validation_novelty_scores = module.fit_pipeline()

    # Do some serializing
    with open('logs/IncrementalPCA/LunarAnalogue-WholeImage-ValidationNoveltyScores.p', 'wb') as f:
        pickle.dump(validation_novelty_scores, f)

    with open('models/saved-models/IncrementalPCA/LunarAnalogue-WholeImage.p', 'wb') as f:
        pickle.dump(module.model, f)


if __name__ == '__main__':
    main()
