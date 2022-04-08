# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 22:21:51 2021

@author: mrinal
"""
import numpy as np
import os
from PIL import Image
from patch import Patch

def create_patch(base_loc, base_patch_dir, patch_size, overlap, mode, names,  csv_output):
    
    img_patch_dir = os.path.join(base_patch_dir, mode, 'images')
    mask_patch_dir = os.path.join(base_patch_dir, mode, 'mask')
    
    if not os.path.exists(img_patch_dir): os.makedirs(img_patch_dir)
    if not os.path.exists(mask_patch_dir): os.makedirs(mask_patch_dir)
    
    for name in names:
        # Read image and mask
        img = Image.open(os.path.join(base_loc, 'images', name[:-6], name + '.jpg'))
        mask = Image.open(os.path.join(base_loc, 'tooth-semantic-masks', name[:-6], name + '.png'))
        # Create patches
        patch = Patch(patch_size, overlap, patch_name=name, csv_output=csv_output)
        patches_im, _, _ = patch.patch2d(np.array(img))    
        patches_mask, _, _ = patch.patch2d(np.array(mask))    
        # Save patches
        patch.save2d(patches_im, save_dir=img_patch_dir, ext = '.png')
        patch.save2d(patches_mask, save_dir=mask_patch_dir, ext = '.png')
