"""
Script used to run novelty detection experiment using incremental Principal
Component Analysis. This script trains and serializes the model for future
evaluation.

Uses:
    Module: PCABaseModule
    Model: IncrementalPCA
    Dataset: LunarAnalogueDataGenerator
    Configuration: pca_incremental_lunar_analogue-OLD.yaml
"""
import pickle

from pathlib import Path
from utils import tools, supported_preprocessing_transforms
from datasets import supported_datamodules
from models import supported_models
from modules.pca_base_module import PCABaseModule


def main():
    config = tools.load_config(DEFAULT_CONFIG_FILE)
    # Unpack configuration
    exp_params = config['experiment-parameters']
    data_params = config['data-parameters']
    module_params = config['module-parameters']

    # Catch a few early, common bugs
    assert ('PCA' in exp_params['model']), \
        'Only accepts PCA-type models for training, check your configuration file.'

    # Set up preprocessing routine
    preprocessing_transforms = supported_preprocessing_transforms[data_params['preprocessing']]

    # Initialize datagenerator
    datamodule = supported_datamodules[exp_params['datamodule']](
        data_transforms=preprocessing_transforms,
        **data_params)

    # Initialize model. With n_component=None the number of features will autoscale to the batch size
    model = supported_models[exp_params['model']](**module_params)

    # Initialize experimental module
    module = PCABaseModule(model)

    # Fit training set with PCA
    module.fit_pipeline(datamodule)

    # Do some serializing
    log_path, version_str, version_nb = tools.prepare_log_path(**exp_params)
    tools.save_object_to_version(config, version=version_nb, filename='configuration.yaml', **exp_params)
    with open(log_path / version_str / 'fitted_model.p', 'wb') as f:
        pickle.dump(module.model, f)


if __name__ == '__main__':
    DEFAULT_CONFIG_FILE = 'configs/pca/pca_standard_mnist.yaml'
    main()
