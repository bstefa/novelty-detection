import unittest
import torch
import numpy as np

from os import path
from utils import tools, supported_preprocessing_transforms
from datasets import supported_datamodules
from datasets.lunar_analogue import LunarAnalogueDataset


class TestLunarAnalogueData(unittest.TestCase):

    def test_whole_image_preprocessing(self):
        cae_config = tools.load_config('configs/cae/cae_baseline_lunar_analogue_whole.yaml', silent=True)
        preprocessing_transforms = supported_preprocessing_transforms['LunarAnalogueWholeImage']

        dm = supported_datamodules['LunarAnalogueDataModule'](
            data_transforms=preprocessing_transforms,
            **cae_config['data-parameters'])
        dm.prepare_data()
        dm.setup('train')

        batch = next(iter(dm.train_dataloader()))
        image, label = batch

        self.assertTrue(-1e-3 < image.mean() < 1e-3)

    # def test_region_extraction_preprocessing(self):
    #
    #     preprocessing_transforms = supported_preprocessing_transforms['LunarAnalogueRegionExtractor']
    #
    #     ds = LunarAnalogueDataset(
    #         root_data_path='/home/brahste/Datasets/NoveltyLunarAnalogue',
    #         train=False,
    #         data_transforms=preprocessing_transforms,
    #         glob_pattern='test/**/*.jpeg')
    #     image, label = ds[int(torch.randint(len(ds), (1,)))]
    #
    #     self.assertTrue(isinstance(image, torch.Tensor))
    #     self.assertEqual(image.shape[0], 16)  # 16 is the expected number of region proposals
    #     self.assertEqual(image.shape[1], 3)  # Ensure 3-channel image
    #
    #     for k, v in label.items():
    #         self.assertIn(k, ['filepath', 'gt_bbox', 'cr_bboxes'])
    #
    #     self.assertTrue(path.exists(label['filepath'][torch.randint(len(image), (1,))][0]))
    #
    #     self.assertTrue(isinstance(label['gt_bbox'], np.ndarray))
    #     self.assertEqual(label['gt_bbox'].shape, (4,))
    #
    #     self.assertTrue(isinstance(label['cr_bboxes'], np.ndarray))
    #     self.assertEqual(label['cr_bboxes'].shape, (16, 4))

    def test_region_extraction_dataloading(self):

        config = tools.load_config('configs/cae/cae_baseline_lunar_analogue_region.yaml', silent=True)
        batch_size = config['data-parameters']['batch_size']

        preprocessing_transforms = supported_preprocessing_transforms['LunarAnalogueRegionExtractor']

        dm = supported_datamodules['LunarAnalogueDataModule'](
            data_transforms=preprocessing_transforms,
            **config['data-parameters'])
        dm.prepare_data()
        dm.setup('test')

        images, labels = next(iter(dm.test_dataloader()))

        self.assertTrue(isinstance(images, torch.Tensor))
        self.assertEqual(images.shape[0], batch_size)
        self.assertEqual(images.shape[1], 16)  # 16 is the expected number of region proposals
        self.assertEqual(images.shape[2], 3)  # Ensure 3-channel image

        for k, v in labels.items():
            self.assertIn(k, ['filepaths', 'gt_bboxes', 'cr_bboxes'])

        self.assertTrue(isinstance(labels['filepaths'], np.ndarray))
        self.assertEqual(labels['filepaths'].shape, (batch_size, 16))
        self.assertTrue(path.exists(labels['filepaths'][torch.randint(len(images), (1,))][0]))

        self.assertTrue(isinstance(labels['gt_bboxes'], np.ndarray))
        self.assertEqual(labels['gt_bboxes'].shape, (batch_size, 16, 4))

        self.assertTrue(isinstance(labels['cr_bboxes'], np.ndarray))
        self.assertEqual(labels['cr_bboxes'].shape, (batch_size, 16, 4))


if __name__ == '__main__':
    unittest.main()
