#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 15:21:21 2021

@author: brahste
"""
import json
import shutil
import skimage.io as io
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split


def multi_glob(path: Path, *args):
    files_grabbed = []
    for sub_glob in args:
        files_grabbed.extend(list(path.glob(str(sub_glob))))
    return files_grabbed

def list_files(d: dict, title: str=''):
    print(f'{title}\n-----')
    print('\n'.join(f'{k}: {len(v)}' for k, v in d.items()))


def main():
    filelist_path = Path('/home/brahste/Projects/novelty-detection/datasets/filename_list.json')
    data_path = Path('/home/brahste/Datasets/LunarAnalogue')
    src_path = data_path/'all'
    
    with open(filelist_path, 'r') as f:
        filelist_json = json.load(f)
        
    list_files(filelist_json, 'Label distribution')

    try:
        (data_path / 'nov-labelled' / 'trainval').mkdir(parents=True)
    except:
        pass

    for key in filelist_json:
        try:
            (data_path / 'nov-labelled' / 'test' / str(key)).mkdir(parents=True)
        except:
            pass

    # Remove typical files from filename dictionary and create list
    typical_files = filelist_json.pop('typical')
    # Split files into training/validation and typical test sets
    trainval_files, typical_test_files = train_test_split(typical_files, test_size=420)

    # Copy train, val files
    for f in trainval_files:
        shutil.copy2(data_path/'all'/f, data_path/'nov-labelled'/'trainval')
    # Copy test files
    for f in typical_test_files:
        shutil.copy2(data_path/'all'/f, data_path/'nov-labelled'/'test'/'typical')
    for key in filelist_json:
        for f in filelist_json[key]:
            shutil.copy2(data_path/'all'/f, data_path/'nov-labelled'/'test'/key)
    
    
    
    
    
    
    


if __name__ == '__main__':
    main()
    
    

