import numpy as np
import pandas as pd
import os
from PIL import Image
import scipy.io as sio

#%% This function creates patches
class Patch:
    def __init__(self, patch_shape, overlap, patch_name='patch', csv_output=False):
        """
        
        patch_shape: Shape of each patch. 
            2D Example: patch_shape = [32, 32]
            3D Example: patch_shape = [32, 32, 16, 1]
        overlap: Overlap between two adjacent patches. 
            2D Example: overlap = [10, 10]
            3D Example: overlap = [8, 8, 8, 0]
        patch_name: It will be used to name the patches. For instance, if patch_name = 'patch',
            then patches will be named as patch_0000, patch_0001, patch_0002, ...
        csv_output: Boolean. If true, then position of each patch in the original image will be
            stored in a .csv file. 
        """
        self.patch_shape = patch_shape
        self.overlap = overlap
        self.patch_name = patch_name
        self.csv_output = csv_output        
        
    def patch2d(self, image):
        # image.shape = height x width x channel         
        """
        image: A 2D or 3D image. For 3D, it should be 4 dimensional (height x width x depth x channel)
        It returns:
            patches: A list that contains all 2D patches
            df: A pandas dataframe that stores location of each patches in the original image
            org_shape: Shape of the original image
        """
        
        df = pd.DataFrame(index=[], columns = [
            'patch_name', 
            'sIdx_ax1', 'eIdx_ax1',
            'sIdx_ax2', 'eIdx_ax2'], dtype=object)
       
        # Get starting index    
        def start(val, step, size):
            if (val+step)>size: s = size - step
            else: s = val
            return s
        
        patches = []
        org_shape = image.shape
        patch_no = 0
        
        for ax1 in np.arange(0, org_shape[0], self.patch_shape[0] - self.overlap[0]):
            for ax2 in np.arange(0, org_shape[1], self.patch_shape[1] - self.overlap[1]):                    
                        ax1_start = start(ax1, self.patch_shape[0], org_shape[0])  # axis 1 starts
                        ax1_end = ax1_start + self.patch_shape[0]  # axis 1 ends
                        ax2_start = start(ax2, self.patch_shape[1], org_shape[1])
                        ax2_end = ax2_start + self.patch_shape[1]
                        
                        cropped = image[ax1_start:ax1_end, ax2_start:ax2_end]          
                        patches.append(cropped)
                        store_name = self.patch_name + '_' + str(patch_no)
                        temp = pd.Series([store_name, ax1_start, ax1_end, ax2_start, ax2_end],
                                         index = [                                  
                                                 'patch_name', 
                                                 'sIdx_ax1', 'eIdx_ax1',
                                                 'sIdx_ax2', 'eIdx_ax2'])
                    
                        df = df.append(temp, ignore_index=True)                
                        patch_no += 1
        if self.csv_output:                
            df.to_csv(self.patch_name + '.csv', index=False)
            
        return patches, df, org_shape
    
    
    def save2d(self, patches, save_dir, ext = '.png'):
        """
        It stores patches to a location specified by the save_dir.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for i, patch in enumerate(patches):
            name = self.patch_name + '_' + str(i).zfill(4) + ext
            patch = Image.fromarray(patch)
            patch.save(os.path.join(save_dir, name))
        
    
    def patch3d(self, image):
        # image.shape = height x width x depth x channel 
        """
        It returns:
            image: A 2D or 3D image. For 3D, it should be 4 dimensional (height x width x depth x channel)
            patches: A list that contains all 2D patches
            df: A pandas dataframe that stores location of each patches in the original image
            org_shape: Shape of the original image
        """
        assert len(image.shape) == 4, 'Data should be 4 dimensional'
        assert len(self.patch_shape) == 4 and len(self.overlap) == 4, 'Length of patch shape and overlap should be 4'
        assert self.overlap[0] < self.patch_shape[0] and self.overlap[1] < self.patch_shape[1] and \
        self.overlap[2] < self.patch_shape[2] and self.overlap[3] < self.patch_shape[3], \
        'Overlap should be smaller than patch shape'
        
        df = pd.DataFrame(index=[], columns = [
            'patch_name', 
            'sIdx_ax1', 'eIdx_ax1',
            'sIdx_ax2', 'eIdx_ax2',
            'sIdx_ax3', 'eIdx_ax3',
            'sIdx_ax4', 'eIdx_ax4'], dtype=object)
    
        def start(val, step, size):
            if (val+step)>size: s = size - step
            else: s = val
            return s
        
        patches = []
        org_shape = image.shape
        patch_no = 0
        
        for ax1 in np.arange(0, org_shape[0], self.patch_shape[0] - self.overlap[0]):
            for ax2 in np.arange(0, org_shape[1], self.patch_shape[1] - self.overlap[1]):
                for ax3 in np.arange(0, org_shape[2], self.patch_shape[2] - self.overlap[2]):
                    for ax4 in np.arange(0, org_shape[3], self.patch_shape[3] - self.overlap[3]):                    
                        ax1_start = start(ax1, self.patch_shape[0], org_shape[0])  # ax1 starts
                        ax1_end = ax1_start + self.patch_shape[0]  # ax1 ends
                        ax2_start = start(ax2, self.patch_shape[1], org_shape[1])
                        ax2_end = ax2_start + self.patch_shape[1]
                        ax3_start = start(ax3, self.patch_shape[2], org_shape[2])
                        ax3_end = ax3_start + self.patch_shape[2]
                        ax4_start = start(ax4, self.patch_shape[3], org_shape[3])
                        ax4_end = ax4_start + self.patch_shape[3]
                        
                        cropped = image[ax1_start:ax1_end,ax2_start:ax2_end,ax3_start:ax3_end,ax4_start:ax4_end]          
                        patches.append(cropped)
                        store_name = self.patch_name + '_' + str(patch_no)
                        temp = pd.Series([store_name, ax1_start, ax1_end, ax2_start, ax2_end, ax3_start, ax3_end, ax4_start, ax4_end],
                                         index = [                                  
                                                 'patch_name', 
                                                 'sIdx_ax1', 'eIdx_ax1',
                                                 'sIdx_ax2', 'eIdx_ax2',
                                                 'sIdx_ax3', 'eIdx_ax3',
                                                 'sIdx_ax4', 'eIdx_ax4'])
                    
                        df = df.append(temp, ignore_index=True)                
                        patch_no += 1
                        
        df.to_csv(self.patch_name + '.csv', index=False)
        
        return patches, df, org_shape

    def save3d(self, patches, save_dir, ext = '.mat'):
        """
        It stores patches to a location specified by the save_dir. Patches are stored in .mat format.
        """        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for i, patch in enumerate(patches):
            name = self.patch_name + '_' + str(i).zfill(4) + ext
            sio.savemat(os.path.join(save_dir, name), {'p': patch}, do_compression=True)

#%% Merge patches
class Merge:
    def __init__(self, info, org_shape, dtype='uint8'):
        """
        info: It's a dataframe or array that contains patches' location in the original image
        org_shape: Shape of the original image
        dtype: data type
        """
        self.info = info
        self.org_shape = org_shape
        self.dtype = dtype
        
    def merge2d(self, patches):
        """
        A list that contains all patches, is provided. Then it merges all patches together. 
        """
        info = np.array(self.info)
        merged = np.zeros(self.org_shape).astype(self.dtype)
        
        cnt = 0
        for row in range(len(patches)):
            # Collect info from this row for this patch
            this_row = info[int(row)]
            sIdx_ax1, eIdx_ax1 = this_row[1], this_row[2] # start and end idx of this patch along ax1
            sIdx_ax2, eIdx_ax2 = this_row[3], this_row[4]
            # Read this patch
            this_patch = patches[cnt]
        
            merged[sIdx_ax1:eIdx_ax1, sIdx_ax2:eIdx_ax2] = this_patch
            cnt += 1
        
        return merged
    

    def merge_from_dir2d(self, patch_dir):
        """
        It reads patches from a directory, then merges them together. 
        """
        names = os.listdir(patch_dir)
        
        rows = []
        for name in names:
            name_split = os.path.splitext(name)[0]
            idx = name_split[-4:] # idx is last 4 characters
            rows.append(idx) 

        info = np.array(self.info)
        merged = np.zeros(self.org_shape).astype(self.dtype)
        
        cnt = 0
        for row in rows:
            # Collect info from this row for this patch
            this_row = info[int(row)]
            sIdx_ax1, eIdx_ax1 = this_row[1], this_row[2] # start and end idx of this patch along ax1
            sIdx_ax2, eIdx_ax2 = this_row[3], this_row[4]            
            this_patch = Image.open(os.path.join(patch_dir, names[cnt])) # Read this patch
        
            merged[sIdx_ax1:eIdx_ax1, sIdx_ax2:eIdx_ax2] = this_patch
            cnt += 1
        
        return merged
    
    
    def merge3d(self, patches):
        """
        A list that contains all patches, is provided. Then it merges all patches together. 
        """
        info = np.array(self.info)
        merged = np.zeros(self.org_shape).astype(self.dtype)
        
        cnt = 0
        for row in range(len(patches)):
            # Collect info from this row for this patch
            this_row = info[row]
            sIdx_ax1, eIdx_ax1 = this_row[1], this_row[2] # start and end idx of this patch along ax1
            sIdx_ax2, eIdx_ax2 = this_row[3], this_row[4]
            sIdx_ax3, eIdx_ax3 = this_row[5], this_row[6]
            sIdx_ax4, eIdx_ax4 = this_row[7], this_row[8]            
            this_patch = patches[cnt] # Read this patch
        
            merged[sIdx_ax1:eIdx_ax1, sIdx_ax2:eIdx_ax2, sIdx_ax3:eIdx_ax3, sIdx_ax4:eIdx_ax4] = this_patch
            cnt += 1
        
        return merged
    
    
    def merge_from_dir3d(self, patch_dir):
        """
        It reads patches from a directory, then merges them together. 
        """
        names = os.listdir(patch_dir)
        
        rows = []
        for name in names:
            name_split = os.path.splitext(name)[0]
            idx = name_split[-4:] # idx is last 4 characters
            rows.append(idx) 

        info = np.array(self.info)
        merged = np.zeros(self.org_shape).astype(self.dtype)
        
        cnt = 0
        for row in rows:
            # Collect info from this row for this patch
            this_row = info[int(row)]
            sIdx_ax1, eIdx_ax1 = this_row[1], this_row[2] # start and end idx of this patch along ax1
            sIdx_ax2, eIdx_ax2 = this_row[3], this_row[4]
            sIdx_ax3, eIdx_ax3 = this_row[5], this_row[6]
            sIdx_ax4, eIdx_ax4 = this_row[7], this_row[8]            
            this_patch = sio.loadmat(os.path.join(patch_dir, names[cnt])) # Read this patch
            this_patch = this_patch['p']
        
            merged[sIdx_ax1:eIdx_ax1, sIdx_ax2:eIdx_ax2, sIdx_ax3:eIdx_ax3, sIdx_ax4:eIdx_ax4] = this_patch
            cnt += 1
        
        return merged