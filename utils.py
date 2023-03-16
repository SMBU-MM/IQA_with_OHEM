import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from BaseCNN import BaseCNN
from Transformers import AdaptiveResize
import scipy.stats
import numpy as np
from itertools import combinations
import os, random, copy, math
import prettytable as pt
from shutil import copyfile
from torch.utils.data import DataLoader

random.seed(10)
np.random.seed(10)
torch.manual_seed(10)

def init_train(train_path, config):
    imgs = np.loadtxt(train_path, dtype=str, delimiter='\t', usecols=(0))
    moss = np.loadtxt(train_path, dtype=float, delimiter='\t', usecols=(1))
    mos_stds = np.loadtxt(train_path, dtype=float, delimiter='\t', usecols=(2))
    idxs = [i for i in range(len(imgs.tolist()))]
    random.shuffle(idxs)
    idxs = idxs[:config.num_per_round]
    img_sampled, mos_sampled, mos_std_sampled, img_unlabeled, mos_unlabeled , mos_std_unlabeled= [], [], [], [], [], []
    for step, (img, mos, mos_std) in enumerate(zip(imgs.tolist(), moss.tolist(), mos_stds.tolist())):
        if step in idxs:
            img_sampled.append(img)
            mos_sampled.append(mos)
            mos_std_sampled.append(mos_std)
        else:
            img_unlabeled.append(img)
            mos_unlabeled.append(mos)
            mos_std_unlabeled.append(mos_std)
    return np.array(img_sampled), np.array(mos_sampled), np.array(mos_std_sampled), \
           np.array(img_unlabeled), np.array(mos_unlabeled), np.array(mos_std_unlabeled)

def pair_wise(i, img_sampled, mos_sampled, std_sampled, img_unlabeled, path='spaq', num_pairs=5000, train_txt=None):
    random.seed(i*100)
    np.random.seed(i*100)
    torch.manual_seed(i*100)
    img_sampled = img_sampled.tolist()
    mos_sampled = mos_sampled.tolist()
    std_sampled = std_sampled.tolist()
    img_unlabeled = img_unlabeled.tolist()
    n = len(img_sampled)
    combs = combinations([i for i in range(n)], 2)
    comb_lists = []
    for item in combs:
        comb_lists.append(item)
    random.shuffle(comb_lists)
    comb_lists = comb_lists[:num_pairs] if len(comb_lists)>num_pairs else comb_lists
    unlabel_1 = copy.deepcopy(img_unlabeled)
    unlabel_2 = copy.deepcopy(img_unlabeled)
    random.shuffle(unlabel_2)
    print('[*] The number of labeled and unlabeled images are {} and {}, respectively.'.format(len(img_sampled), len(unlabel_2)))
    with open(train_txt, 'w') as wfile:    
        # pairewise training data
        for step, (i, j) in enumerate(comb_lists):
            img1 = img_sampled[i]
            img2 = img_sampled[j]
            diff = float(mos_sampled[i]) - float(mos_sampled[j])
            sq = np.sqrt(float(std_sampled[i])*float(std_sampled[i]) \
                           + float(std_sampled[j])*float(std_sampled[j])) + 1e-8
            prob_label = 0.5 * (1 + math.erf(diff / sq))
            binary_label = 1 if mos_sampled[i]>mos_sampled[j] else 0
            un_img1 = unlabel_1[step%len(unlabel_1)]
            un_img2 = unlabel_2[step%len(unlabel_2)]
            wstr = "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(img1, img2, un_img1, un_img2,
                                                             binary_label,0,prob_label,binary_label)
            wfile.write(wstr)
    return 0

def print_tb(srcc, plcc):
    # evaluate after every epoch
    tb = pt.PrettyTable()
    tb.field_names = ["Model1", "VALID", "LIVE", "KADID10k"]
    tb.add_row(['SRCC', srcc["valid"], srcc["test"], srcc["test1"]])
    tb.add_row(['PLCC', plcc["valid"], plcc["test"], plcc["test1"]])
    return tb


