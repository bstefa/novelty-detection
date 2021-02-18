# Hacky way to find packages in sibling folder
import sys
sys.path.append('.')

from datasets.lunar_analogue import *

def load_configuration_file():
    return 1

def test_dataset_initialization():
        ROOT_DATA_PATH = '/home/brahste/Datasets/LunarAnalogue/batches'
        dataset = LunarAnalogueDataset(ROOT_DATA_PATH)

        print(len(dataset))
        assert 1 == 2