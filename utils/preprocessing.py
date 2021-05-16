import cv2 as cv
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from sklearn.cluster import KMeans

# import logging
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class RegionProposalBING:
    """
    This class uses Binarized normed gradients (BING) to compute unsupervised region proposals.
    Original literature can be found at: https://mmcheng.net/bing/
    """
    def __init__(
            self,
            n_regions: int = 16,
            region_shape: tuple = (64, 64, 3),
            return_tensor: bool = True,
            view_region_proposals: bool = False):
        super().__init__()

        self.n_regions = n_regions
        self.region_shape = region_shape
        self._return_tensor = return_tensor
        self._view_region_proposals = view_region_proposals

        self._saliency_obj = cv.saliency.ObjectnessBING_create()
        self._saliency_obj.setTrainingPath('models/_bing_trained_models')
        self._kmeans = KMeans(n_clusters=n_regions)

    def __call__(self, image: np.ndarray):
        # TODO: Set up asserts to ensure data is general correct upon importing
        
        # Compute saliency boxes
        success, raw_rects = self._saliency_obj.computeSaliency(image)
        rects = np.empty((len(raw_rects), 4))
        areas = np.empty(len(raw_rects))
        whaspects = np.empty(len(raw_rects))
        
        for i, (x1, y1, x2, y2) in enumerate(raw_rects.reshape(-1, 4)):
            w, h = x2 - x1, y2 - y1
            rects[i] = [x1, y1, w, h]
            areas[i] = w * h
            whaspects[i] = w / h

        area_low, area_high = np.quantile(areas, [0.9, 1.00])
        whaspect_low, whaspect_high = np.quantile(whaspects, [0., 0.7])
        hwaspect_low, hwaspect_high = np.quantile(1/whaspects, [0., 0.7])
        
        keep_rects = rects[
            (areas > area_low) & (areas < area_high) &
            (whaspects > whaspect_low) & (whaspects < whaspect_high) &
            ((1/whaspects) > hwaspect_low) & ((1/whaspects) < hwaspect_high)
            ]
                
        self._kmeans.fit(keep_rects)
        warped_crops = np.empty((self.n_regions, *self.region_shape))
        crop_bboxes = np.empty((self.n_regions, 4))

        for i, (x, y, w, h) in enumerate(self._kmeans.cluster_centers_):
            x, y, w, h = int(x), int(y), int(w), int(h)
            crop = image[y:y+h, x:x+w]
            warped_crop = cv.resize(crop, (self.region_shape[0], self.region_shape[1]), interpolation=cv.INTER_CUBIC)

            crop_bboxes[i] = np.array([x, y, w, h])
            warped_crops[i] = warped_crop

        if self._view_region_proposals:
            fig, ax = plt.subplots()
            ax.imshow(image)
            for crop in crop_bboxes:
                rect = patches.Rectangle((crop[0], crop[1]), crop[2], crop[3], facecolor='none', edgecolor='r')
                ax.add_patch(rect)
            plt.show()

        if self._return_tensor:
            warped_crops = torch.tensor(warped_crops).permute(0, 3, 1, 2).to(dtype=torch.float32)
        return warped_crops, crop_bboxes


class RegionProposalSS:
    """
    Conceptually this class would be called after another preprocessing
    pipeline is applied first, such as LunarAnaloguePreprocessingPipeline
    """
    def __init__(
            self,
            n_regions: int = 16,
            region_shape: tuple = (64, 64, 3),
            return_tensor: bool = True,
            view_region_proposals: bool = False):
        super().__init__()

        self.n_regions = n_regions
        self.region_shape = region_shape
        self._return_tensor = return_tensor
        self._view_region_proposals = view_region_proposals

        self._ss = cv.ximgproc.segmentation.createSelectiveSearchSegmentation()
        self._kmeans = KMeans(n_clusters=n_regions)

    def __call__(self, image: np.ndarray):
        # Currently, images are assumed to be in RGB form with intensities in the range [0, 1]

        # Initialize selective search algorithm with fast search strategy
        self._ss.setBaseImage(image)
        self._ss.switchToSelectiveSearchFast()

        # Get the predicted rectangles in (x, y, w, h) form
        rects = self._ss.process()

        # Filter rectangles according to total area between defined quantiles
        areas = np.empty(len(rects))
        whaspects = np.empty(len(rects))
        for i, (_, _, w, h) in enumerate(rects):
            areas[i] = w * h
            whaspects[i] = w / h

        area_low, area_high = np.quantile(areas, [0.2, 1.])
        whaspect_low, whaspect_high = np.quantile(whaspects, [0., 0.9])
        hwaspect_low, hwaspect_high = np.quantile(1/whaspects, [0., 0.9])

        keep_rects = rects[
            (areas > area_low) & (areas < area_high) &
            (whaspects > whaspect_low) & (whaspects < whaspect_high)
            & ((1/whaspects) > hwaspect_low) & ((1/whaspects) < hwaspect_high)
        ]

        self._kmeans.fit(keep_rects)
        warped_crops = np.empty((self.n_regions, *self.region_shape))
        crop_bboxes = np.empty((self.n_regions, 4))

        for i, (x, y, w, h) in enumerate(self._kmeans.cluster_centers_):
            x, y, w, h = int(x), int(y), int(w), int(h)
            crop = image[y:y+h, x:x+w]
            warped_crop = cv.resize(crop, (self.region_shape[0], self.region_shape[1]), interpolation=cv.INTER_CUBIC)

            crop_bboxes[i] = np.array([x, y, w, h])
            warped_crops[i] = warped_crop

        if self._view_region_proposals:
            fig, ax = plt.subplots()
            ax.imshow(image)
            for crop in crop_bboxes:
                rect = patches.Rectangle((crop[0], crop[1]), crop[2], crop[3], facecolor='none', edgecolor='r')
                ax.add_patch(rect)
            plt.show()

        if self._return_tensor:
            warped_crops = torch.tensor(warped_crops).permute(0, 3, 1, 2).to(dtype=torch.float32)
        return warped_crops, crop_bboxes


class LunarAnaloguePreprocessingPipeline:
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

    def __init__(self, normalize: str = 'standard', scale: float = 1/5):
        assert any( (normalize == rout for rout in ('standard', 'zero_to_one')) ), \
            'Unsupported normalization routine, must be one of: standard, zero_to_one'
        assert (0. <= scale <= 1.), \
            'Scaling factor must be between 0 and 1'
        self._normalize = normalize
        self._scale = scale

    def __call__(self, image: np.ndarray) -> np.ndarray:
        assert (len(image.shape) == 3), \
            'Lunar Analogue images must have 3 dimensions'
        assert (image.shape[-1] == 3), \
            'Lunar Analogue images must be 3-channel RGB/YCrCb'
        n_channels = image.shape[-1]

        # Here we want to preserve the aspect ratio
        h = int(image.shape[0] * self._scale)
        w = int(image.shape[1] * self._scale)
        image = cv.resize(image, (w, h), interpolation=cv.INTER_AREA)

        # To conduct histogram equalization you have to operate on the intensity
        # values of the image, so a different color space is required
        # TODO: Evaluate the effects of training your model on images in YCrCb colour space
        image = cv.cvtColor(image, cv.COLOR_RGB2YCrCb)
        image[..., 0] = cv.equalizeHist(image[..., 0])
        image = cv.cvtColor(image, cv.COLOR_YCrCb2RGB)

        # Minor Gaussian blurring
        image = cv.medianBlur(image, 3)

        # Convert image dtype to float
        image = np.float32(image)

        # Normalize image
        if self._normalize == 'standard':
            for c in range(n_channels):
                image[..., c] = (image[..., c] - image[..., c].mean()) / image[..., c].std()
        elif self._normalize == 'zero_to_one':
            image = (image - image.min()) / (image.max() - image.min())

        return image


class CuriosityPreprocessingPipeline:
    """
    Standard image preprocessing pipeline for Curiosity data.
    1) Convert data to numpy.float32
    2) Channelwise standardization
    """

    def __init__(self):
        return

    def __call__(self, image: np.ndarray) -> np.ndarray:
        assert image.shape == (64, 64, 6), \
            'Dataset not in correct format for pre-processing'
        n_channels = image.shape[-1]

        # Convert image dtype to float
        image = np.float32(image)

        # Standardize image
        for c in range(n_channels):
            image[..., c] = (image[..., c] - image[..., c].mean()) / image[..., c].std()

        return image


class NoveltyMNISTPreprocessingPipeline:
    """
    Standard image preprocessing pipeline for Novelty MNIST data.
    1) Convert data to torch.float32
    2) Standardization
    """

    def __init__(self):
        return

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        assert image.shape == (1, 28, 28), 'Dataset not in correct format for pre-processing'

        # Convert image dtype to float
        image = image.to(dtype=torch.float32)

        # Standardize image
        image = (image - image.mean()) / image.std()

        return image


if __name__ == '__main__':
    from utils import tools
    import cv2 as cv

    moon = cv.imread('utils/scripts/moon.jpeg')
    moon = cv.cvtColor(moon, cv.COLOR_BGR2RGB)

    moon = tools.LunarAnaloguePreprocessingPipeline()(moon)
    moon = tools.unstandardize_batch(moon)

    NRE = NovelRegionExtractorPipeline()
    out = NRE(moon)

    print(out.shape)




