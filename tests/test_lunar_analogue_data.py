import unittest
import torch
import matplotlib.pyplot as plt
import numpy as np


from utils import tools, supported_preprocessing_transforms
from datasets import supported_datamodules


class TestLunarAnalogueData(unittest.TestCase):

    def test_whole_image_preprocessing(self):
        self.cae_config = tools.load_config('configs/cae/cae_baseline_lunar_analogue_whole.yaml', silent=True)
        preprocessing_transforms = supported_preprocessing_transforms['LunarAnalogueWholeImage']

        self.dm = supported_datamodules['LunarAnalogueDataModule'](
            data_transforms=preprocessing_transforms,
            **self.cae_config['data-parameters'])
        self.dm.prepare_data()
        self.dm.setup('train')

        batch = next(iter(self.dm.train_dataloader()))
        image, label = batch

        self.assertTrue(-1e-3 < image.mean() < 1e-3)
        print(type(image))
        print(type(label))

    def test_region_extraction_preprocessing(self):

        self.cae_config = tools.load_config('configs/cae/cae_baseline_lunar_analogue_region.yaml', silent=True)
        preprocessing_transforms = supported_preprocessing_transforms['LunarAnalogueRegionExtractor']

        self.dm = supported_datamodules['LunarAnalogueDataModule'](
            data_transforms=preprocessing_transforms,
            **self.cae_config['data-parameters'])
        self.dm.prepare_data()
        self.dm.setup('test')

        batch = next(iter(self.dm.test_dataloader()))
        image, label = batch

        for k, v in label.items():
            self.assertIn(k, ['filepaths', 'gt_bboxes', 'cr_bboxes'])


if __name__ == '__main__':
    unittest.main()
