## Overview

This is the PyTorch implementation of the paper [JPTS:Solving Performance Degradation of Massive
MIMO CSI Feedback]().
If you feel this repo helpful, please cite our paper:

```
@article{}
}

```


## Requirements

To use this project, you need to ensure the following requirements are installed.

- Python >= 3.7
- [PyTorch >= 1.2 also compatible with PyTorch >= 1.7 with fft fix](https://pytorch.org/get-started/locally/)
- [thop](https://github.com/Lyken17/pytorch-OpCounter)

## Project Preparation

#### A. Data Preparation

The channel state information (CSI) matrix is generated from [COST2100](https://ieeexplore.ieee.org/document/6393523) model. Chao-Kai Wen and Shi Jin group provides a pre-processed version of COST2100 dataset in [Google Drive](https://drive.google.com/drive/folders/1_lAMLk_5k1Z8zJQlTr5NRnSD6ACaNRtj?usp=sharing), which is easier to use for the CSI feedback task; You can also download it from [Baidu Netdisk](https://pan.baidu.com/s/1Ggr6gnsXNwzD4ULbwqCmjA).

You can generate your own dataset according to the [open source library of COST2100](https://github.com/cost2100/cost2100) as well. The details of data pre-processing can be found in our paper.

#### B. Project Tree Arrangement

We recommend you to arrange the project tree as follows.

```
home
├── JPTS  # The cloned CLNet repository
│   ├── dataset
│   ├── models
│   ├── utils
│   ├── main.py
├── COST2100  # The data folder
│   ├── DATA_Htestin.mat
│   ├── ...
├── Experiments
│   ├── checkpoints  # The checkpoints folder
│   │     ├── in_04.pth
│   │     ├── ...
│   ├── run.sh  # The bash script
...
```

## Adpoting JPTS to SOTA methods

This repo provides three SOTA methods: CSINet, CRNet and CLNet. You can simply configure [crnet.py](https://github.com/SIJIEJI/JPTS/blob/main/models/crnet.py#L312) to change the models and train it from the scratch. 
An example of run.sh is listed below. Simply use it with `sh run.sh`.  Change scenario using `--scenario` and change compression ratio with `--cr`.

``` bash
python /home/JPTS/main.py \
  --data-dir '/home/COST2100' \
  --scenario 'in' \
  --epochs 1000 \
  --batch-size 200 \
  --workers 8 \
  --cr 4 \
  --scheduler cosine \
  --gpu 0 \
  2>&1 | tee log.out
```

## Results and Reproduction


The NMSE result reported in the paper as follow:

## CSINet:
|Scenario | Compression Ratio | NMSE | Checkpoints
|:--: | :--: | :--: | :--: | 
|indoor | 1/4 | -26.55 |  CSINET_IN_4.pth |
|indoor | 1/8 |  -24.16|  CSINET_IN_8.pth|
|indoor | 1/16 | -22.30 |  CSINET_IN_16.pth|
|indoor | 1/32 | -21.29 |  CSINET_IN_32.pth|
|indoor | 1/64 | -19.19 |  CSINET_IN_64.pth|
|indoor | 1/128 | -18.25 |  CSINET_IN_128.pth|
|indoor | 1/256 | -16.90 |   CSINET_IN_256.pth|
|indoor | 1/512 | -16.38 |  CSINET_IN_512.pth|
|indoor | 1/1024 | -15.95 | CSINET_IN_1024.pth|
|outdoor | 1/4 | -20.77 | CSINET_OUT_4.pth|
|outdoor | 1/8 | -18.78 |  CSINET_OUT_8.pth|
|outdoor | 1/16 | -17.41 |  CSINET_OUT_16.pth|
|outdoor | 1/32 | -15.80 |  CSINET_OUT_32.pth|
|outdoor | 1/64 | -14.38 |  CSINET_OUT_64.pth|
|outdoor | 1/128 | 14.16 | CSINET_OUT_128.pth|
|outdoor | 1/256 | -13.62 | CSINET_OUT_256.pthh|
|outdoor | 1/512 | -13.39 | CSINET_OUT_512.pthh|
|outdoor | 1/1024 | -13.32 | CSINET_OUT_1024.pth |

## CRNet:
|Scenario | Compression Ratio | NMSE | Checkpoints
|:--: | :--: | :--: | :--: | 
|indoor | 1/4 | -23.79 |  CRNET_IN_4.pth |
|indoor | 1/8 |  -22.98|  CRNET_IN_8.pth|
|indoor | 1/16 | -21.57 |  CRNET_IN_16.pth|
|indoor | 1/32 | -21.74 |  CRNET_IN_32.pth|
|indoor | 1/64 | -20.42 |  CRNET_IN_64.pth|
|indoor | 1/128 | -18.43 |  CRNET_IN_128.pth|
|indoor | 1/256 | -16.59 |  CRNET_IN_256.pth|
|indoor | 1/512 | -16.38 |  CRNET_IN_512.pth|
|indoor | 1/1024 | -16.01 |  CRNET_IN_1024.pth|
|outdoor | 1/4 | -20.12 | CRNET_OUT_4.pth|
|outdoor | 1/8 | -18.92 |  CRNET_OUT_8.pth|
|outdoor | 1/16 | -16.77 |  CRNET_OUT_16.pth|
|outdoor | 1/32 | -15.58 |  CRNET_OUT_32.pth|
|outdoor | 1/64 | -14.95 |  CRNET_OUT_64.pth|
|outdoor | 1/128 | -13.91 |  CRNET_OUT_128.pth|
|outdoor | 1/256 | -13.50 |  CRNET_OUT_256.pth|
|outdoor | 1/512 | -13.41 |  CRNET_OUT_512.pth|
|outdoor | 1/1024 | -13.33 |  CRNET_OUT_1024.pth|

## CLNet:
|Scenario | Compression Ratio | NMSE | Checkpoints
|:--: | :--: | :--: | :--: | 
|indoor | 1/4 | -24.87 |  CLNET_IN_4.pth |
|indoor | 1/8 |  -23.73|  CLNET_IN_8.pth|
|indoor | 1/16 | -22.85 | CLNET_IN_16.pth|
|indoor | 1/32 | -21.77 |  CLNET_IN_32.pth|
|indoor | 1/64 | -20.26 |  CLNET_IN_64.pth|
|indoor | 1/128 | -18.51 |  CLNET_IN_128.pth|
|indoor | 1/256 | -17.43 | CLNET_IN_256.pth|
|indoor | 1/512 | -16.83 |  CLNET_IN_512.pth|
|indoor | 1/1024 | -16.06 | CLNET_IN_1024.pth|
|outdoor | 1/4 | -20.77 | CLNET_OUT_4.pth|
|outdoor | 1/8 | -18.78 |  CLNET_OUT_8.pth|
|outdoor | 1/16 | -17.41 |  CLNET_OUT_16.pth|
|outdoor | 1/32 | -15.80 | CLNET_OUT_32.pth|
|outdoor | 1/64 | -14.38 | CLNET_OUT_64.pth|
|outdoor | 1/128 | -14.26 |  CLNET_OUT_128.pth|
|outdoor | 1/256 | -13.59 | CLNET_OUT_256.pth|
|outdoor | 1/512 | -13.36 | CLNET_OUT_512.pth|
|outdoor | 1/1024 | -13.32 | CLNET_OUT_1024.pth|

If you want to reproduce our result, you can directly download the corresponding checkpoints from [OneDrive](https://entuedu-my.sharepoint.com/:f:/g/personal/sijie001_e_ntu_edu_sg/ErxmhooY67FAj_z_MNNgaHgBiqvcWv_QQXr0lOTRbYyI1A?e=Zxxay5)


**To reproduce all these results, simple add `--evaluate` to `run.sh` and pick the corresponding pre-trained model with `--pretrained`.** An example is shown as follows.

``` bash
python /home/JPTS/main.py \
  --data-dir '/home/COST2100' \
  --scenario 'in' \
  --pretrained './checkpoints/in4.pth' \
  --evaluate \
  --batch-size 200 \
  --workers 0 \
  --cr 4 \
  --cpu \
  2>&1 | tee test_log.out

```

## Acknowledgment

This repository is modified from the [CLNet open source code](https://github.com/SIJIEJI/CLNet) & [CRNet open source code](https://github.com/Kylin9511/CRNet). 
Thanks Chao-Kai Wen and Shi Jin group for providing the pre-processed COST2100 dataset, you can find their related work named CsiNet in [Github-Python_CsiNet](https://github.com/sydney222/Python_CsiNet) 

