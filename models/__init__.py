from .cae_baseline import BaselineCAE
from .cae_compression import CompressionCAE

supported_models = {
    'BaselineCAE': BaselineCAE,
    'CompressionCAE': CompressionCAE
}