# clear_custom_dataset.py - utility so save some time managing files 
# Tristan Hill - Feb. 27, 2025 

""" deletes file to prepare for new dataset

Usage example: python ./clear_custom_data.py
"""
import os
import shutil
import argparse

parser=argparse.ArgumentParser(
  prog='clear_custom_dataset',
  description='deletes dataset from custom_features/generate_dataset.py for votenet')
parser.add_argument('--overwrite', type=bool, required=False)

# should add a common directory 'CustomFeatures', this is a little lazy
# folders to be deleted
DATA_FOLDER = './CustomFeatures/data'
LABELS_FOLDER = './CustomFeatures/labels'
PCDS_FOLDER = './CustomFeatures/pcds'
POINTS_FOLDER = './CustomFeatures/points'

# file to be deleted
SPLIT_FILES = ['./CustomFeatures/custom_train.txt', 
               './CustomFeatures/custom_test.txt', 
               './CustomFeatures/custom_val.txt']

def clear_dataset():

    if os.path.exists(DATA_FOLDER):
        print('Deleting data folder: {}'.format(DATA_FOLDER))
        shutil.rmtree(DATA_FOLDER)

    if os.path.exists(LABELS_FOLDER):
        print('Deleting labels folder: {}'.format(LABELS_FOLDER))
        shutil.rmtree(LABELS_FOLDER)
    
    if os.path.exists(PCDS_FOLDER):
        print('Deleting pcds folder: {}'.format(PCDS_FOLDER))
        shutil.rmtree(PCDS_FOLDER)

    if os.path.exists(POINTS_FOLDER):
        print('Deleting points folder: {}'.format(POINTS_FOLDER))
        shutil.rmtree(POINTS_FOLDER)
    
    for sf in SPLIT_FILES:
      if os.path.isfile(sf):
        print('Deleting split file: {}'.format(sf))
        os.remove(sf)

if __name__=='__main__':    
    config=parser.parse_args()
    clear_dataset()
