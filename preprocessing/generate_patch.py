# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 22:07:59 2021

@author: mrinal

This code generates patches from the original image.

It reads train, test, and validation names from a json file. Then reads images from the
given directory and creats patches.

The variable 'patch_size' defines each patch size. In addition, the variable 'overlap'
indicates overlap between two adjacent patches. 
"""
import json
import numpy as np
import os
from PIL import Image
from patch import Patch
from my_utils import create_patch

patch_size = [512, 512]
overlap = [10, 10]

# Open JSON file
f = open('names.json')
names = json.load(f)
f.close()

# Read names
names_train = names['train']
names_val = names['val']
names_test = names['test']

# Load directory
base_loc = './a_panoramic_teeth_segmentation/tooth_dataset'

'''
Directory format - 

tooth_dataset
          | - images
                | - cate1
                | - cate2
                | - cate3
                | .
                | .
                | - cate10

Following is an exmaple of directory of cate1 - 
  './a_panoramic_teeth_segmentation/tooth_dataset/images/cate1'
'''

# Store directory
base_patch_dir = './a_panoramic_teeth_segmentation/patch_' + str(patch_size[0])

# Create train patch 
create_patch(base_loc, base_patch_dir, patch_size, overlap, mode='train', names=names_train, csv_output=False)

# Create validation patch 
create_patch(base_loc, base_patch_dir, patch_size, overlap, mode='val', names=names_val, csv_output=False)

# Uncomment to create test patch. In this project, test patches were not created beforehand. Instead, they were
# created during the test phase to save memory. 

# create_patch(base_loc, base_patch_dir, patch_size, overlap, mode='test', names=names_test, csv_output=True)
