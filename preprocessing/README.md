# Generate patches
This code generates patches from the original image. <br>

Run the code `generate_patch.py`. <br>

Set the dataset directory. The format of the dataset directory is described in the code. <br>

Here is the format -
```
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
```

It reads train, test, and validation names from a json file. Then reads images from the given directory and creats patches. <br>

In this project, test patches were not created beforehand. Instead, they were created during the test phase to save memory. 

The variable `patch_size` defines size of each patch. In addition, the variable `overlap`
indicates overlap between two adjacent patches. 
