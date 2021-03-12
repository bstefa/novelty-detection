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

from utils import dtypes

def config_from_command_line(default_config: str):
    if len(sys.argv) == 1:
        # Then the file being called is the only argument
        # so return the default configuration
        config_file = default_config
    elif len(sys.argv) == 2:
        # Then a specific configuration file has been used
        # so load it
        config_file = sys.argv[0]
    elif all([len(sys.argv) == 3, sys.argv[1] == '-f']):
        config_file = default_config
    else:
        print(sys.argv)
        raise ValueError('CLI only accepts 0 args (default) or 1 arg (path/to/config).')

    with open(str(config_file)) as f:
        y = yaml.full_load(f)
        print(f'Experimental parameters\n------')
        pprint(y)
        return y


def config_from_file(config_file: str):
    with open(str(config_file)) as f:
        y = yaml.full_load(f)


def save_dictionary_to_current_version(log_dir: str, name: str, filename: str, dictionary: dict):
    save_path = Path(log_dir)/name
    save_path = sorted(save_path.glob('v*'))[-1]

    with open(str(save_path/filename), 'w') as f:
        yaml.dump(dictionary, f)


def save_object_to_version(obj, version: int, filename: str, log_dir: str = 'logs', name: str = 'Unnamed'):
    save_path = Path(log_dir)/name/f'version_{version}'
    if isinstance(obj, dtypes.Figure):
        obj.savefig(save_path/filename, format='eps')
    if isinstance(obj, dict):
        with open(str(save_path/filename), 'w') as f:
            yaml.dump(obj, f)



def chw2hwc(x):
    if isinstance(x, torch.Tensor):
        return x.permute(1, 2, 0)
    if isinstance(x, np.array):
        return np.transpose(x, (1, 2, 0))


def unstandardize_batch(batch_in: torch.Tensor, tol: float = 0.01):
    '''
    This function is purposed for converting images pixels
    from a unit Gaussian into the range [0,1] for viewing
    '''
    if isinstance(batch_in, torch.Tensor):
        # Clone batch and detach from the computational graph
        batch = batch_in.detach().clone().to(device='cpu')

        # Convert pixel range for each image in the batch
        for b in range(len(batch)):
            extremum = torch.max(torch.abs(batch[b]))
            batch[b] = (batch[b] / (2 * extremum)) + 0.5

        # Some basic assertions to ensure correct range manipulation
        assert torch.max(batch) < (1.0 + tol), f'The maximum pixel intensity ({torch.max(batch)}) is out of range'
        assert torch.min(batch) > (0.0 - tol), f'The minimum pixel intensity ({torch.min(batch)}) is out of range'

    elif isinstance(batch_in, np.ndarray):
        # Clone batch and detach from the computational graph
        batch = batch_in.copy()

        # Convert pixel range for each image in the batch
        for b in range(len(batch)):
            extremum = np.amax(np.abs(batch[b]))
            batch[b] = (batch[b] / (2 * extremum)) + 0.5

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
        assert torch.max(x_err) < (1.0 + tol), 'The maximum pixel intensity is out of range'
        assert torch.min(x_err) > (0.0 - tol), 'The minimum pixel intensity is out of range'
        return x_err
    if isinstance(x_output, np.ndarray):
        x_err = x_input - x_output
        return x_err / np.max(np.abs(x_err))


def gaussian_window(mean, std):
    x = np.linspace(mean - 3.5 * std, mean + 3.5 * std, 100)
    coefficient = (1 / std * (2 * np.pi) ** (-1 / 2))
    argument = (-1 / 2) * (((x - mean) / std) ** 2)
    return np.exp(argument)


# INCOMPLETE: DO NOT USE
class BatchStatistics(object):
    '''
    Evaluates statistics on input batch, accessible as member functions
    for easy, on the fly access.
    '''

    def __init__(self, batch_in):
        if isinstance(batch_in, torch.Tensor):
            # Detach tensor from comp graph, clone to cpu, convert to numpy
            batch_in = batch_in.detach().clone().to(device='cpu').numpy()
        assert isinstance(batch_in, np.ndarray), 'Batch wasn\'t successfully converted to numpy'
        self.batch_in = batch_in

    # The main purpose of any decorator is to change your class methods or
    # attributes in such a way so that the user of your class no need to
    # make any change in their code.
    @property
    def shape(self):
        return self.batch_in.shape

    @property
    def mean(self):
        return np.mean(self.batch_in)

    @property
    def min(self):
        return np.amin(self.batch_in)

    @property
    def max(self):
        return np.amax(self.batch_in)

    @property
    def extremum(self):
        return np.amax(np.amax(self.batch_in), np.abs(np.amin(self.batch_in)))

    @property
    def std(self):
        return np.std(self.batch_in)


class PreprocessingPipeline:
    """
    Standard image preprocessing pipeline for Lunar Analogue data.
    Cascades processing steps:
        1) resize
        2) histogram equalization
        3) Gaussian blurring
        4) channelwise standardization
    See the slides from: https://cedar.buffalo.edu/~srihari/CSE676/12.2%20Computer%20Vision.pdf
    for more information on the steps used here.
    """

    def __init__(self):
        return

    def __call__(self, image: np.ndarray) -> np.ndarray:
        n_channels = image.shape[-1]

        image = cv.resize(image, (256, 256), interpolation=cv.INTER_AREA)

        # To conduct histogram equalization you have to operate on the intensity
        # values of the image, so a different color space is required
        image = cv.cvtColor(image, cv.COLOR_RGB2YCrCb)
        image[..., 0] = cv.equalizeHist(image[..., 0])
        image = cv.cvtColor(image, cv.COLOR_YCrCb2RGB)

        # Minor Gaussian blurring
        image = cv.GaussianBlur(image, (3, 3), 0.5)

        # Convert image dtype to float
        image = np.float32(image)

        # Standardize image
        for c in range(n_channels):
            image[..., c] = (image[..., c] - image[..., c].mean()) / image[..., c].std()

        return image


if __name__ == '__main__':
    from datasets.lunar_analogue import LunarAnalogueDataGenerator

    config = config_from_command_line('configs/incremental_pca.yaml')
    data_obj = LunarAnalogueDataGenerator(config)
    gen = data_obj.create_generator('train')

    batch = next(gen)

    bstat_obj = BatchStatistics(batch)
    print(bstat_obj.mean)

    #
    # fig = plt.figure()
    # ax1 = fig.add_subplot(211)
    # ax1.imshow(image)
    # ax2 = fig.add_subplot(212)
    # ax2.imshow(im_out);
    # plt.show()
