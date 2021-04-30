from .preprocessing import *
from .tools import unstandardize_batch

from torchvision import transforms

LunarAnalogueWholeImage = transforms.Compose([
    LunarAnaloguePreprocessingPipeline()
])

LunarAnalogueRegionExtractor = transforms.Compose([
    LunarAnaloguePreprocessingPipeline(normalize='zero_to_one'),
    NovelRegionExtractorPipeline(view_region_proposals=True),
    # transforms.Lambda(lambda regions: torch.stack([transforms.ToTensor()(region) for region in regions])),
    # transforms.Lambda(lambda x: x.to(dtype=torch.float32))
])

supported_preprocessing_transforms = {
    'LunarAnalogueWholeImage': LunarAnalogueWholeImage,
    'LunarAnalogueRegionExtractor': LunarAnalogueRegionExtractor
}