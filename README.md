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

`Colab_hybrid_unet_2d.ipynb`
----------------------------

## Description not complete yet

### Reference:
Sha, Y. Keras-unet-collection. GitHub repository (2021) doi:10.5281/zenodo.5449801.
