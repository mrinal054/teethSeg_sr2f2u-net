# S-R2F2U-Net: A single-stage model for teeth segmentation
This implementation is leveraged from [Yingkai Sha’s](https://github.com/yingkaisha/keras-unet-collection) repository. Base models for recurrent, residual and attention are taken from the above mentioned repository. <br><br>

The original paper can be found [here](https://arxiv.org/abs/2204.02939). <br><br>

Install the following packages - 
```
pip install keras-unet-collection
pip install jenti
```
To install in Colab - 
```
!pip install --target='/content/drive/MyDrive/library' keras-unet-collection
!pip install --target='/content/drive/MyDrive/library' jenti
```
You can set your own target location. <br><br>
During the test phase, the package jenti is used to create patches. More details of it can be found [here](https://github.com/mrinal054/patch_and_merge).

## Implementation details
* How to create patches is described in the folder `preprocessing`.
* 2D models are described in `Colab_hybrid_unet_2d.ipynb`. In the paper, this code is used for implementation and evaluation. 
* Both 2D and 3D models are described in `Colab_hybrid_unet_2d_3d.ipynb`.

Class `DataGenerator`
----------------------------
This class is used in both `Colab_hybrid_unet_2d.ipynb` and `Colab_hybrid_unet_2d_3d.ipynb`. It loads data batch-wise from a given directory. <br>

Loading the entire training and validation dataset is memory expensive for Colab. Instead, a data loader class called `DataGenerator` is implemented that loads images on-the-fly. In other words, it takes a list of image names and the directory where the images are situated. Then it loads images batch-wise while training the model. <br>

Reference: [Click here](https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly) <br>

Here is an example of how to use it - 
```
list_IDs_train = os.listdir(img_dir_train) # list of training image names 
list_IDs_val = os.listdir(img_dir_val) # list of validation image names

# Call DataGenerator
train_gen = DataGenerator(list_IDs=list_IDs_train,
                          dir_image=img_dir_train, # directory of images
                          dir_mask=mask_dir_train, # directory of masks or ground truth
                          n_channels_image=3,
                          n_channels_mask=1,
                          dim=(512,512),
                          batch_size=batch_size,
                          shuffle=True)

val_gen = DataGenerator(list_IDs=list_IDs_val,
                          dir_image=img_dir_val,
                          dir_mask=mask_dir_val,
                          n_channels_image=3,
                          n_channels_mask=1,
                          dim=(512,512),
                          batch_size=batch_size,
                          shuffle=True)
```


`Colab_hybrid_unet_2d.ipynb`
----------------------------
It can create models from any combination of the following four parameters - <br>
* Residual
* Recurrent
* Attention
* Filter doubling <br>
Here are some network models shown in the paper - 
|Models|Residual|Recurrent1|Recurrent2|Filter doubling|Attention|
|:----:|:------:|:--------:|:--------:|:-------------:|:-------:|



## Description not complete yet

### Reference:
Sha, Y. Keras-unet-collection. GitHub repository (2021) doi:10.5281/zenodo.5449801.
