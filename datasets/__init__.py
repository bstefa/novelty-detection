from .novelty_mnist import NoveltyMNISTDataModule
from .lunar_analogue import LunarAnalogueDataModule, LunarAnalogueDataGenerator
from .curiosity import CuriosityDataModule

supported_datamodules = {
    'LunarAnalogueDataModule': LunarAnalogueDataModule,
    'LunarAnalogueDataGenerator': LunarAnalogueDataGenerator,
    'NoveltyMNISTDataModule': NoveltyMNISTDataModule,
    'CuriosityDataModule': CuriosityDataModule
}