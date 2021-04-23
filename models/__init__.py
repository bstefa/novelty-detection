from .aae_simple import SimpleAAE
from .vae import SimpleVAE, BaselineVAE
from .cae_baseline import BaselineCAE
from .cae_compression import CompressionCAEMidCapacity, CompressionCAEHighCapacity
from .pca import IncrementalPCA, StandardPCA

supported_models = {
    'SimpleAAE': SimpleAAE,
    'SimpleVAE': SimpleVAE,
    'BaselineVAE': BaselineVAE,
    'BaselineCAE': BaselineCAE,
    'CompressionCAEMidCapacity': CompressionCAEMidCapacity,
    'CompressionCAEHighCapacity': CompressionCAEHighCapacity,
    'StandardPCA': StandardPCA,
    'IncrementalPCA': IncrementalPCA
}