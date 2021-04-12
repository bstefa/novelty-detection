import cv2 as cv
import numpy as np

from sklearn.cluster import KMeans


class NovelRegionExtractorPipeline:
    '''
    Conceptually this class would be called after another preprocessing
    pipeline is applied first, such as LunarAnaloguePreprocessingPipeline'''
    def __init__(self, n_regions: int = 16, region_shape: tuple = (64, 64, 3)):
        super().__init__()

        self.n_regions = n_regions
        self.region_shape = region_shape

        self._ss = cv.ximgproc.segmentation.createSelectiveSearchSegmentation()
        self._kmeans = KMeans(n_clusters=n_regions)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        # Currently, images are assumed to be in RGB form with intensities in the range [0, 1]

        # Initialize selective search algorithm with fast search strategy
        self._ss.setBaseImage(image)
        self._ss.switchToSelectiveSearchFast()

        # Get the predicted rectangles in (x, y, w, h) form
        rects = self._ss.process()

        # Filter rectangles according to total area between defined quantiles
        area_low, area_high = np.quantiles([w * h for (_, _, w, h) in rects], [0.50, 0.95])
        keep_rects = filter(lambda rect: self._filter_by_area(rect, area_low, area_high), rects)

        self._kmeans.fit(keep_rects)
        warped_crops = np.empty((self.n_regions, *self.region_shape))

        for i, (x, y, w, h) in enumerate(self._kmeans.cluster_centers_):
            crop = image[y:y+h, x:x+w]
            warped_crop = cv.resize(crop, (self.region_shape[0], self.region_shape[1]), interpolation=cv.INTER_CUBIC)
            warped_crops[i] = warped_crop

        return warped_crops

    @staticmethod
    def _filter_by_area(self, rect, area_low: float, area_high: float):
        assert len(rect) == 4, 'Only accepts bounding rectangles with elements (x, y, w, h)'
        area = rect[2] * rect[3]
        return (area > area_low) and (area < area_high)




