import torchvision.transforms as tf
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import sys
import yaml


def config_from_command_line(default_config: str):
    if len(sys.argv) == 1:
        with open(default_config) as f:
            return yaml.full_load(f)
    elif len(sys.argv) == 2:
        with open(str(sys.argv[0])) as f:
            return yaml.full_load(f)
    else:
        raise ValueError('More than one configuration file was provided.')


def chw2hwc(x: torch.Tensor):
    return x.permute(1, 2, 0)


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


def get_error_map(x_input, x_output, use_batch: bool = True, tol: float = 0.001):
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


class PreprocessingPipeline:
    '''
    Sets the contrast (e.g. standard deviation) of pixels
    in an image equal to a specified value

    See the slides from: https://cedar.buffalo.edu/~srihari/CSE676/12.2%20Computer%20Vision.pdf
    '''

    def __init__(self):
        return

    def __call__(self, image: np.ndarray) -> np.ndarray:
        n_channels = image.shape[-1]

        # Cascade processing steps: 
        # 1) resize
        # 2) histogram equalization
        # 3) channelwise standardization
        image = cv.resize(image, (512, 512), interpolation=cv.INTER_AREA)

        # To conduct histogram equalization you have to operate on the intesity
        # values of the image, so a different color space is required
        image = cv.cvtColor(image, cv.COLOR_RGB2YCrCb)
        image[..., 0] = cv.equalizeHist(image[..., 0])
        image = cv.cvtColor(image, cv.COLOR_YCrCb2RGB)
        # ! Consider experimenting with blurring, may improve system performance
        ### image = cv.GaussianBlur(image, (3, 3), 1)

        # Convert image dtype to float
        image = np.float32(image)

        # Standardize image
        for c in range(n_channels):
            image[..., c] = (image[..., c] - image[..., c].mean()) / image[..., c].std()

        ### print('Resulting image statistics: ', image.max(), image.min(), image.mean(), image.std())

        return image


if __name__ == '__main__':
    from lunar_dataset import LunarAnalogueDataset

    # !Temporary constants
    ROOT_DATA_PATH = '/home/brahste/Datasets/LunarAnalogue/images-screened'

    dataset = LunarAnalogueDataset(ROOT_DATA_PATH)
    image = dataset[552]  # arbitrary

    im_out = PreprocessingPipeline()(image)

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.imshow(image)
    ax2 = fig.add_subplot(212)
    ax2.imshow(im_out);
    plt.show()
