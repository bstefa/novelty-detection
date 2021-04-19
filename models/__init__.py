from .aae_simple import SimpleAAE
from .cae_baseline import BaselineCAE
from .cae_compression import CompressionCAEMidCapacity, CompressionCAEHighCapacity

supported_models = {
    'SimpleAAE': SimpleAAE,
    'BaselineCAE': BaselineCAE,
    'CompressionCAEMidCapacity': CompressionCAEMidCapacity,
    'CompressionCAEHighCapacity': CompressionCAEHighCapacity
}