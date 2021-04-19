from .aae_simple import SimpleAAE
from .vae_simple import SimpleVAE
from .cae_baseline import BaselineCAE
from .cae_compression import CompressionCAEMidCapacity, CompressionCAEHighCapacity
from .pca_standard import StandardPCA
from .pca_incremental import IncrementalPCA

supported_models = {
    'SimpleAAE': SimpleAAE,
    'SimpleVAE': SimpleVAE,
    'BaselineCAE': BaselineCAE,
    'CompressionCAEMidCapacity': CompressionCAEMidCapacity,
    'CompressionCAEHighCapacity': CompressionCAEHighCapacity,
    'StandardPCA': StandardPCA,
    'IncrementalPCA': IncrementalPCA
}