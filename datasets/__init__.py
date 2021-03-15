from .emnist import EMNISTDataModule
from .mnist import MNISTDataModule
from .lunar_analogue import LunarAnalogueDataModule, LunarAnalogueDataGenerator

import datasets

supported_datamodules = {
    'lunar-analogue-datamodule': LunarAnalogueDataModule,
    'lunar-analogue-datagenerator': LunarAnalogueDataGenerator,
    'mnist-datamodule': MNISTDataModule,
    'emnist-datamodule': datasets.emnist.EMNISTDataModule
}