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
During the test phase, the package `jenti` is used to create patches. More details of it can be found [here](https://github.com/mrinal054/patch_and_merge).

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
It can work for both 2d and 3d. A new parameter called `CONV` is used to define 2d or 3d model. 
* For 2d model, set `CONV='2d'`
* For 3d model, set `CONV='3d'`

`Colab_hybrid_layer_output.ipynb`
---------------------------------
It demonstrates layers' outputs during the decoding process. For better visualization, all the feature maps are resize to the original image size. It can generate layer outputs in two ways - 
* Create model using layer names, and then generate layer outputs
* Create model using layer index, and then generate layer outputs

Here is an example of creating model using layer names - 

```from keras.models import load_model

# Load an existing model
existing_model = load_model('/content/drive/MyDrive/panoramicDentalSegmentation/checkpoints/hybrid_unet_2d/recur_F_T_residual_filter_double/2022-02-24_06-24-34/cp-0025.ckpt', compile=False)```

# Get the desired layer names from the model summary
existing_model.summary()

# Layer names
input_layer_name = 'hybrid_unet_input'
output_layer_name1 = 'hybrid_unet_up1_add1'
output_layer_name2 = 'hybrid_unet_up2_add1'
output_layer_name3 = 'hybrid_unet_up3_add1'
output_layer_name4 = 'hybrid_unet_up4_add1'
output_layer_name5 = 'hybrid_unet_output_activation'

# Method 1: Previous-model input is taken as the input
model1 = Model(inputs=existing_model.input, outputs=existing_model.get_layer(output_layer_name1).output)
model2 = Model(inputs=existing_model.input, outputs=existing_model.get_layer(output_layer_name2).output)
model3 = Model(inputs=existing_model.input, outputs=existing_model.get_layer(output_layer_name3).output)
model4 = Model(inputs=existing_model.input, outputs=existing_model.get_layer(output_layer_name4).output)
model5 = Model(inputs=existing_model.input, outputs=existing_model.get_layer(output_layer_name5).output)

models = [model1, model2, model3, model4, model5]
```
Here is an example of how to create a model from layer indices - 
```
layers = {i: v for i, v in enumerate(existing_model.layers)}

# Method 1
model = Model(inputs=existing_model.input, outputs=existing_model.get_layer(index=100).output)
```
## Performance analysis table

| Model | Acc | Spec. | Pre. | Recall | Dice | Param (M)|
| :---: | :---: |  :---: |  :---: |  :---: |  :---: |  :---: |
| Attention U-Net | 97.06 | 98.60 | 94.28 | 91.22 | 92.55 | 43.94 |
| U<sup>2</sup> NET | 96.82 | **98.84** | **95.12** | 89.01 | 91.78 | 60.11 |
| R2U-NET (up to 4 levels) | 97.27 | 98.66 | 94.61 | 92.02 | 93.11 | 108.61 |
| ResUNET-a | 97.10 | 98.50 | 93.95 | 91.77 | 92.66 | 4.71 |
| U-Net | 96.04 | 97.68 | 89.89 | 90.18 | 89.33 | 31.04 |
| BiseNet | 95.05 | 95.98 | 85.53 | 92.48 | 87.8 | 12.2 |
| DenseASPP | 95.5 | 97.76 | 90.09 | 86.88 | 88.13 | 46.16 |
| SegNet | 96.38 | 98.32 | 92.26 | 89.05 | 90.15 | 29.44 |
|BASNet | 96.77 |98.64 | 94.56 | 90.11 | 92.12 | 87.06 |
|TSASNet | 96.94 | 97.81 | 94.77 | **93.77** | 92.72 | 78.27 |
|MSLPNet | 97.30 | 98.45 | 93.35 | 92.97 | 93.01 | - |
| LTPEDN | 94.32 | - | - | - | 92.42 | - |
| S-R2U-Net | 97.22 | 98.52 | 94.06 | 92.33 | 93.01 | 77.17 |
| S-R2F2U-Net | **97.31** | 98.55 | 94.27 | 92.61 | **93.26** | 59.12 |
| S-R2F2-Attn-U-Net | 97.23 | 98.55 | 94.19 | 92.16 | 93.00 | 59.25 |



### Reference:
Sha, Y. Keras-unet-collection. GitHub repository (2021) doi:10.5281/zenodo.5449801.
