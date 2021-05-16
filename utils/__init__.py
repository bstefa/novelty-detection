from .preprocessing import *

from torchvision import transforms


CuriosityPreprocessing = transforms.Compose([
    CuriosityPreprocessingPipeline(),
    transforms.ToTensor()
])

LunarAnalogueWholeImage = transforms.Compose([
    LunarAnaloguePreprocessingPipeline(normalize='standard'),
    transforms.ToTensor()
])

# Because of the fancy labelling and collation, ToTensor cannot be applied to this
# transform. Formatting and type casting of tensors is handled in the NovelRegion
LunarAnalogueRegionExtractor = transforms.Compose([
    LunarAnaloguePreprocessingPipeline(normalize='zero_to_one'),
    RegionProposalSS(return_tensor=True),
])

# Novelty MNIST doesn't need to be transformed to a Tensor because the data
# is already in Tensor format upon importing
NoveltyMNISTPreproccesing = transforms.Compose([
    NoveltyMNISTPreprocessingPipeline()
])

supported_preprocessing_transforms = {
    'NoveltyMNISTPreprocessing': NoveltyMNISTPreproccesing,
    'CuriosityPreprocessing': CuriosityPreprocessing,
    'LunarAnalogueWholeImage': LunarAnalogueWholeImage,
    'LunarAnalogueRegionExtractor': LunarAnalogueRegionExtractor
}
