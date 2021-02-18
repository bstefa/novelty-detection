# Hacky way to help find packages in sibling folder
import sys
sys.path.append('.')

import unittest
import numpy as np

from datasets.lunar_analogue import LunarAnalogueDataset

class TestLunarAnalogueDataset(unittest.TestCase):

    def setUp(self):
        ROOT_DATA_PATH = '/home/brahste/Datasets/LunarAnalogue/images-screened'
        self.dataset = LunarAnalogueDataset(ROOT_DATA_PATH)

        self.MIN_DATASET_LENGTH = 1000
        self.IMAGE_SHAPE = (1242, 2208, 3)
        self.IMAGE_TYPE = np.ndarray

    def test_dataset_length(self):
        '''
        Ensures that the glob pattern and root data path
        are compatible for the dataset
        May have to be altered if labels are built into the
        dataset later on.
        '''
        self.assertGreater(len(self.dataset), self.MIN_DATASET_LENGTH)

    def test_output_image_shape(self):
        '''
        Makes sure that the images being imported are 2K
        This test will have to be altered if transforms are
        included in the dataset later on.
        '''
        image = self.dataset[np.random.randint(0, len(self.dataset))]
        self.assertEqual(image.shape, self.IMAGE_SHAPE)

    def test_output_image_type(self):
        '''
        Ensures the type of the output image is a numpy array
        May have to change if I actually want to output pytorch tensor
        '''
        image = self.dataset[np.random.randint(0, len(self.dataset))]
        self.assertTrue(isinstance(image, self.IMAGE_TYPE))

if __name__ == '__main__':
    unittest.main()