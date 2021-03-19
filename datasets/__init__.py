from .emnist import EMNISTDataModule
from .mnist import MNISTDataModule
from .lunar_analogue import LunarAnalogueDataModule, LunarAnalogueDataGenerator
from .curiosity import CuriosityDataModule

supported_datamodules = {
    'LunarAnalogueDataModule': LunarAnalogueDataModule,
    'LunarAnalogueDataGenerator': LunarAnalogueDataGenerator,
    'MNISTDataModule': MNISTDataModule,
    'EMNISTDataModule': EMNISTDataModule,
    'CuriosityDataModule': CuriosityDataModule
}