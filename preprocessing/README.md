# Generate patches
This code generates patches from the original image. <br>

Run the code `generate_patch.py`. <br>

Set the dataset directory. The format of the dataset directory is described in the code. <br>

It reads train, test, and validation names from a json file. Then reads images from the given directory and creats patches.

The variable `patch_size` defines each patch size. In addition, the variable `overlap`
indicates overlap between two adjacent patches. 
