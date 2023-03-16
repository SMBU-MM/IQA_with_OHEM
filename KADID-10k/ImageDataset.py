import os
import torch
import functools
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def image_loader(image_name):
    I = Image.open(image_name).convert('RGB') 
    if I.size[0] < 384 or I.size[1] < 384:
        if I.size[0] < I.size[1]:
            I = I.resize((384, int(384*I.size[1]/I.size[0])), Image.BICUBIC)
        else:
            I = I.resize((int(384*I.size[0]/I.size[1]), 384), Image.BICUBIC)
    return I


def get_default_img_loader():
    return functools.partial(image_loader)


class ImageDataset(Dataset):
    def __init__(self, csv_file,
                 img_dir,
                 transform=None,
                 test=False,
                 get_loader=get_default_img_loader):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory of the images.
            transform (callable, optional): transform to be applied on a sample.
        """
        print('start loading csv data...')
        self.data = pd.read_csv(csv_file, sep='\t', header=None)
        print('%d csv data successfully loaded!' % self.__len__())
        self.img_dir = img_dir
        self.test = test
        self.transform = transform
        self.loader = get_loader()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            samples: a Tensor that represents a video segment.
        """
        if self.test:
            image_name = os.path.join(self.img_dir, self.data.iloc[index, 0])
            I = self.loader(image_name)
            if self.transform is not None:
                I = self.transform(I)

            mos = self.data.iloc[index, 1]
            sample = {'I': I, 'mos': mos}
        else:
            img_name1 = os.path.join(self.img_dir, self.data.iloc[index, 0])
            img_name2 = os.path.join(self.img_dir, self.data.iloc[index, 1])

            img1 = self.loader(img_name1)
            img2 = self.loader(img_name2)
            if self.transform is not None:
                I1 = self.transform(img1)
                I2 = self.transform(img2)
                I3 = self.transform(img1)
                I4 = self.transform(img2)
            y = torch.FloatTensor(self.data.iloc[index, 2:].tolist())
            sample = {'I1': I1, 'I2': I2, 'I3': I3, 'I4': I4, 'y': y[3], 'std1':y[1], 'std2':y[2], 'yb': y[3]}

        return sample

    def __len__(self):
        return len(self.data.index)
