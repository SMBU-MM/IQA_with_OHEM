# Deep Blind Image Quality Assessment Powered by Online Hard Example Mining

## Introduction
This repository contains the official pytorch implementation of the paper ["Deep Blind Image Quality Assessment Powered by Online Hard Example Mining"]([https://ieeexplore.ieee.org/document/10070789]) by Zhihua Wang, Qiuping Jiang, Shanshan Zhao, Wensen Feng and Weisi Lin,  IEEE Transactions on Multimedia, 2023.

Recently, blind image quality assessment (BIQA) models based on deep neural networks (DNNs) have achieved impressive correlation numbers on existing datasets. However, due to the intrinsic imbalance property of the training set, not all distortions or images are handled equally well. Online hard example mining (OHEM) is a promising way to alleviate this issue. Inspired by the recent finding that model compression hampers the memorization of a tractable subset, we propose an effective "plug-and-play" OHEM pipeline, especially for generalizable deep BIQA. Specifically, We train two different parallel weight-sharing branches simultaneously, where one is the full model while the other is "self-competitor" generated from the full model online by network pruning. Then, we leverage the prediction disagreement between full model and its pruned variant (i.e., the self-competitor) to expose easily "forgettable" samples which are therefore regarded as the hard ones. We then enforce the prediction consistency between the full model and its pruned variant to implicitly put more focus on these hard samples, which benefits the full model to recover the forgettable information introduced by pruning. Since the pruned variant is online generated and updated from the latest full model, the two branches co-evolve during optimization. Extensive experiments across multiple datasets and BIQA models demonstrate that the proposed OHEM that can further improve the model performance and generalizability measured by correlation numbers and group maximum differentiation competition (gMAD).

## Prerequisites
* python 3.10

* pytorch 1.12.0

* ``pip install -r requirements.txt``

## Training
To train the CD-Flow from scratch, execute the following command:
```bash
python Main.py 
```
For the training and testing split, please check the .
## Evaluation
To evaluate of your checkpoints on test set, execute:
```bash
python test.py
```
## Citation
```
@article{wang2023deep,
  author={Wang, Zhihua and Jiang, Qiuping and Zhao, Shanshan and Feng, Wensen and Lin, Weisi},
  journal={IEEE Transactions on Multimedia}, 
  title={Deep Blind Image Quality Assessment Powered by Online Hard Example Mining}, 
  year={2023},
  pages={1-11}}
```
