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
**[Note: An augmentation-enabled DataGenerator is also available upon request. Currently, it supports rotation, flip and shift. To get it, email at mdhar@uwm.edu with the subject line '*Requesting for augmentation-enabled DataGenerator*'.]**

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
It can create models from any combination of following four parameters - <br>
* Residual
* Recurrent
* Attention
* Filter doubling <br><br>
Here are some network models shown in the paper - <br>


| Models | Residual | Recurrent1 | Recurrent2 | Filter doubliing | Attention |
| :---: | :---: |  :---: |  :---: |  :---: |  :---: |
| Attention U-Net | &cross; |  &cross; | &cross; | &cross; | &check; |
| R2U-Net | &check; |  &check; | &check; | &cross; | &cross; |
| S-R2U-Net | &check; |  &cross; | &check; | &cross; | &cross; |
| S-R2F2U-Net | &check; |  &cross; | &check; | &check; | &check; |
| S-R2F2-Attn-U-Net | &check; |  &cross; | &check; | &check; | &check; |

Following is a example of how to use `Colab_hybrid_unet_2d.ipynb`.

```
# Hyper-parameters
IMG_HEIGHT = 512 
IMG_WIDTH  = 512 
IMG_CHANNELS = 3 
NUM_LABELS = 2  #Binary
input_shape = (IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS)
batch_size = 2
FILTER_NUM = [32, 64, 128, 256, 512]
STACK_NUM_DOWN = 2
STACK_NUM_UP = 2
DILATION_RATE = 1
FILTER_DOUBLE = True
RECUR_STATUS = (False, True)
RECUR_NUM = 2
IS_RESIDUAL = True
IS_ATTENTION = True
ATTENTION_ACTIVATION = 'ReLU'
ATTENTION = 'add'
ACTIVATION = 'ReLU'
OUTPUT_ACTIVATION = 'Softmax'
BATCH_NORM = True
POOL = False
UNPOOL = False
RETRAIN = False


# Current version works for "stack_num_down = stack_num_up" only
model = hybrid_unet_2d(input_shape, filter_num=FILTER_NUM, 
                       n_labels=NUM_LABELS, 
                       stack_num_down=STACK_NUM_DOWN, stack_num_up=STACK_NUM_UP, 
                       dilation_rate=DILATION_RATE,
                       filter_double=FILTER_DOUBLE,
                       recur_status=RECUR_STATUS, recur_num=RECUR_NUM,
                       is_residual=IS_RESIDUAL,
                       is_attention=IS_ATTENTION,
                       atten_activation=ATTENTION_ACTIVATION, attention=ATTENTION,
                       activation=ACTIVATION, output_activation=OUTPUT_ACTIVATION, 
                       batch_norm=BATCH_NORM, pool=POOL, unpool=UNPOOL, name='hybrid_unet')
```

`Colab_hybrid_unet_2d_3d.ipynb`
-------------------------------
It can work for both 2d and 3d. A new parameter called `CONV` is used to define 2d or 3d model. For 2d model, set `CONV='2d'`, and for 3d model, set `CONV='3d'`.

## Description not complete yet

### Reference:
Sha, Y. Keras-unet-collection. GitHub repository (2021) doi:10.5281/zenodo.5449801.
