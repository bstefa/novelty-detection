"""
A mix and match of useful tools for the repo
"""
import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F
import sys
import yaml

from pathlib import Path
from pprint import pprint

from utils.dtypes import *
from utils import dtypes


def config_from_command_line(default_config: str):
    """Use this function to read configurations for training scripts"""
    if len(sys.argv) == 1:
        # Then the file being called is the only argument so return the default configuration
        config_file = default_config
    elif len(sys.argv) == 2:
        # Then a specific configuration file has been used so load it
        config_file = str(sys.argv[1])
    elif all([len(sys.argv) == 3, sys.argv[1] == '-f']):
        config_file = default_config
    else:
        print(sys.argv)
        raise ValueError('CLI only accepts 0 args (default) or 1 arg (path/to/config).')

    with open(str(config_file), 'r') as f:
        y = yaml.full_load(f)
        print(f'Experimental parameters\n------')
        pprint(y)
        return y


def config_from_file(config_file: str):
    """Use this function to read configurations in Jupyter Notebooks"""
    with open(str(config_file)) as f:
        y = yaml.full_load(f)
        print(f'Experimental parameters\n------')
        pprint(y)
        return y


def save_object_to_version(
        obj,
        version: int,
        filename: str,
        log_dir: str = 'logs',
        model: str = 'Unnamed',
        datamodule: str = 'Unknown',
        **kwargs):
    save_path = Path(log_dir) / datamodule / model / f'version_{version}'
    if isinstance(obj, dtypes.Figure):
        obj.savefig(save_path/filename, format='eps')
    if isinstance(obj, dict):
        with open(str(save_path/filename), 'w') as f:
            yaml.dump(obj, f)


def calc_out_size(in_size, padding, kernel_size, dilation, stride):
    return ((in_size + 2*padding - dilation*(kernel_size-1) - 1) / stride) + 1


class PathGlobber:
    def __init__(self, path: str):
        self.path = Path(path)

    def glob(self, pattern: str):
        return list(self.path.glob(pattern))

    def multiglob(self, patterns: Iterable[str]):
        list_of_paths = []
        for pat in patterns:
            list_of_paths.extend(self.path.glob(pat))
        return list_of_paths


def chw2hwc(x):
    if isinstance(x, torch.Tensor):
        return x.permute(1, 2, 0)
    if isinstance(x, np.array):
        return np.transpose(x, (1, 2, 0))


def unstandardize_batch(batch_in: torch.Tensor, tol: float = 0.001):
    '''
    This function is purposed for converting images pixels
    from a unit Gaussian into the range [0,1] for viewing
    '''
    if isinstance(batch_in, torch.Tensor):
        # Clone batch and detach from the computational graph
        batch = batch_in.detach().clone().to(device='cpu')

        for b in range(len(batch)):
            minimum = torch.min(batch[b])
            maximum = torch.max(batch[b])
            batch[b] = (batch[b] - minimum) / (maximum - minimum)

        # Some basic assertions to ensure correct range manipulation
        assert torch.max(batch) < (1.0 + tol), f'The maximum pixel intensity ({torch.max(batch)}) is out of range'
        assert torch.min(batch) > (0.0 - tol), f'The minimum pixel intensity ({torch.min(batch)}) is out of range'

    elif isinstance(batch_in, np.ndarray):
        # Clone batch and detach from the computational graph
        batch = batch_in.copy()

        # Convert pixel range for each image in the batch
        for b in range(len(batch)):
            minimum = np.amin(batch[b])
            maximum = np.amax(batch[b])
            batch[b] = (batch[b] - minimum) / (maximum - minimum)

        # Some basic assertions to ensure correct range manipulation
        assert np.amax(batch) < (1.0 + tol), f'The maximum pixel intensity ({torch.max(batch)}) is out of range'
        assert np.amin(batch) > (0.0 - tol), f'The minimum pixel intensity ({torch.min(batch)}) is out of range'

    return batch


def get_error_map(x_input, x_output, tol: float = 0.001):
    if isinstance(x_output, torch.Tensor):
        # Note that these operations are for batches
        x_in = x_input.detach().clone().to(device='cpu')
        x_out = x_output.detach().clone().to(device='cpu')
        x_err = F.mse_loss(x_in, x_out, reduction='none')

        if len(x_in.shape) == 4:
            # Convert each image in the batch to range [0,1]
            for e in range(len(x_err)):
                x_err[e] = x_err[e] / torch.max(x_err[e])
        elif len(x_in.shape) == 3:
            # Convert the single image to range [0,1]
            x_err = x_err / torch.max(x_err)
        else:
            raise ValueError('Input to error_map must be of shape 3 or 4')

        # Some basic assertions to ensure correct range manipulation
        assert torch.max(x_err) < (1.0 + tol), f'The maximum pixel intensity {torch.max(x_err)} is out of range'
        assert torch.min(x_err) > (0.0 - tol), f'The minimum pixel intensity {torch.min(x_err)} is out of range'
        return x_err
    if isinstance(x_output, np.ndarray):
        x_err = x_input - x_output
        return x_err / np.max(np.abs(x_err))


if __name__ == '__main__':
    p = PathGlobber('datasets/filename_list.json')
