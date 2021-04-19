"""
A mix and match of useful tools for the repo
"""
import numpy as np
import torch.nn.functional as F
import sys
import yaml

from pathlib import Path
from pprint import pprint

from utils.dtypes import *
from utils import dtypes


def load_config(default_config: str, silent=False):
    """
    Multi-use function. Exposes command-line to provide alternative configurations to training scripts.
    Also works as a stand-alone configuration importer in notebooks and scripts.
    # TODO: Create reference in README for this function.
    """
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
        if not silent:
            print(f'Experimental parameters\n------')
            pprint(y)
        return y


def save_object_to_version(obj, version: int, filename: str, log_dir='logs', model='Unnamed', datamodule='Unknown',
                           **kwargs):
    save_path = Path(log_dir) / datamodule / model / f'version_{version}'
    if isinstance(obj, str):
        f = open(save_path / filename, 'wt')
        f.write(obj)
        f.close()
    elif isinstance(obj, dtypes.Figure):
        obj.savefig(save_path / filename, format='eps')
    elif isinstance(obj, dict):
        with open(str(save_path / filename), 'w') as f:
            yaml.dump(obj, f)
    else:
        raise Exception('No correct types were found, not saving...')


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


def prepare_log_path(log_dir, datamodule, model, **kwargs):
    log_path = Path.cwd() / log_dir / datamodule / model
    version, v = next_version(log_path)
    if not (log_path / version).is_dir():
        (log_path / version).mkdir(exist_ok=True, parents=True)
    return log_path, version, v


def next_version(path: str):
    v = 0
    while (Path(path) / f'version_{v}').is_dir():
        v += 1
    return f'version_{v}', v


if __name__ == '__main__':
    p = PathGlobber('datasets/filename_list.json')
