# Diffusion-Guided Cross learning with Contrastive Label Propagation for Semi-Supervised Medical Image Segmentation

We thank for the code of "Towards Generic Semi-Supervised Framework for Volumetric Medical Image Segmentation", upon which our implementation is built.

## 1. Environment
- pytorch: 2.2.2+cu118
- linux
- GPU: NVIDIA GeForce RTX 2080
- Need to install Mamba in advance, we use code of "Segmamba: Long-range sequential modeling mamba for 3d medical image segmentation".

## 2. Data Preparation

The file structure should be: 
```
.
├── Datasets
│   ├── LASeg
│   │   ├── 2018LA_Seg_Training Set
│   │   │   ├── 0RZDK210BSMWAA6467LU
│   │   │   │   ├── mri_norm2.h5
│   │   │   ├── 1D7CUD1955YZPGK8XHJX
│   │   │   └── ...
│   │   ├── test.list
│   │   └── train.list
│   ├── Pancreas
│   │   ├── Pancreas_data
│   │   │   ├── PANCREAS_0001.h5
│   │   │   └── ...
│   │   ├── test.list
│   │   └── train.list
```

Run preprocess_la.py

Run preprocess_pancreas.py

Run preprocess_brats.py


## 3. Training & Testing & Evaluating

Training: train_diff_cnn_cl.py

Parameters:
```
LA_8: -task 'la' -exp 'diff_cnn_cl_la' -mu 2.0 -sl labeled_0.1 -su unlabeled_0.1 -se test -ep 300 --base_lr 0.1 
```

Testing & Evaluating: test.py & evaluate.py

Parameters:
```
LA_8: -task 'la' -exp 'diff_cnn_cl_la'
```
