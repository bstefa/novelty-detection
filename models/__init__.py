from .cae_baseline import BaselineCAE
from .cae_compression import CompressionCAEMidCapacity, CompressionCAEHighCapacity

supported_models = {
    'BaselineCAE': BaselineCAE,
    'CompressionCAEMidCapacity': CompressionCAEMidCapacity,
    'CompressionCAEHighCapacity': CompressionCAEHighCapacity
}