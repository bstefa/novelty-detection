from .preprocessing import *

from torchvision import transforms


NoveltyMNISTPreproccesing = transforms.Compose([
    NoveltyMNISTPreprocessingPipeline()
])

CuriosityPreprocessing = transforms.Compose([
    CuriosityPreprocessingPipeline(),
    transforms.ToTensor()
])

LunarAnalogueWholeImage = transforms.Compose([
    LunarAnaloguePreprocessingPipeline(),
    transforms.ToTensor()
])

# Because of the fancy labelling and collation ToTensor cannot be applied to this
# transform. It needs to return images as correctly formatted tensors
LunarAnalogueRegionExtractor = transforms.Compose([
    LunarAnaloguePreprocessingPipeline(normalize='zero_to_one'),
    NovelRegionExtractorPipeline(view_region_proposals=False),
])

supported_preprocessing_transforms = {
    'NoveltyMNISTPreprocessing': NoveltyMNISTPreproccesing,
    'CuriosityPreprocessing': CuriosityPreprocessing,
    'LunarAnalogueWholeImage': LunarAnalogueWholeImage,
    'LunarAnalogueRegionExtractor': LunarAnalogueRegionExtractor
}
