## Overview

This is the PyTorch implementation of the ICC'23 paper [JPTS:Enhancing Deep Learning Performance of Massive
MIMO CSI Feedback](https://arxiv.org/pdf/2208.11333.pdf).
If you feel this repo helpful, please cite our paper:

```
@article{ji2022enhancing,
  title={Enhancing Deep Learning Performance of Massive MIMO CSI Feedback},
  author={Ji, Sijie and Li, Mo},
  journal={arXiv preprint arXiv:2208.11333},
  year={2022}
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

## JPTS-CSINet:
|Scenario | Compression Ratio | NMSE | Checkpoints
|:--: | :--: | :--: | :--: | 
|indoor | 1/4 | -24.19 |  CSINET_IN_4.pth |
|indoor | 1/8 |  -15.20|  CSINET_IN_8.pth|
|indoor | 1/16 | -10.65 |  CSINET_IN_16.pth|
|indoor | 1/32 | -8.59 |  CSINET_IN_32.pth|
|indoor | 1/64 | -6.26 |  CSINET_IN_64.pth|
|outdoor | 1/4 | -12.20 | CSINET_OUT_4.pth|
|outdoor | 1/8 | -7.97 |  CSINET_OUT_8.pth|
|outdoor | 1/16 | -5.22 |  CSINET_OUT_16.pth|
|outdoor | 1/32 | -3.12 |  CSINET_OUT_32.pth|
|outdoor | 1/64 | -2.17 |  CSINET_OUT_64.pth|


## JPTS-CRNet:
|Scenario | Compression Ratio | NMSE | Checkpoints
|:--: | :--: | :--: | :--: | 
|indoor | 1/4 | -26.84 |  CRNET_IN_4.pth |
|indoor | 1/8 |  -16.32|  CRNET_IN_8.pth|
|indoor | 1/16 | -11.55 |  CRNET_IN_16.pth|
|indoor | 1/32 | -8.98 |  CRNET_IN_32.pth|
|indoor | 1/64 | -6.50 |  CRNET_IN_64.pth|
|outdoor | 1/4 | -12.72 | CRNET_OUT_4.pth|
|outdoor | 1/8 | -8.01 |  CRNET_OUT_8.pth|
|outdoor | 1/16 | -5.41 |  CRNET_OUT_16.pth|
|outdoor | 1/32 | -3.38 |  CRNET_OUT_32.pth|
|outdoor | 1/64 | -2.21 |  CRNET_OUT_64.pth|

## JPTS-CLNet:
|Scenario | Compression Ratio | NMSE | Checkpoints
|:--: | :--: | :--: | :--: | 
|indoor | 1/4 | -28.38 |  CLNET_IN_4.pth |
|indoor | 1/8 |  -16.03|  CLNET_IN_8.pth|
|indoor | 1/16 | -12.16 | CLNET_IN_16.pth|
|indoor | 1/32 | -9.00 |  CLNET_IN_32.pth|
|indoor | 1/64 | -6.86 |  CLNET_IN_64.pth|
|outdoor | 1/4 | -12.90 | CLNET_OUT_4.pth|
|outdoor | 1/8 | -8.40 |  CLNET_OUT_8.pth|
|outdoor | 1/16 | -5.61 |  CLNET_OUT_16.pth|
|outdoor | 1/32 | -3.61 | CLNET_OUT_32.pth|
|outdoor | 1/64 | -2.30 | CLNET_OUT_64.pth|


If you want to reproduce our result, you can directly download the corresponding checkpoints from [OneDrive](https://entuedu-my.sharepoint.com/:f:/g/personal/sijie001_e_ntu_edu_sg/Ej0oi6n3H1RNs84AnGfK2ukBk_9gynl4KBDsvLmZnnT5Zg?e=HYWsff)


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

