from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
from PIL import Image
import numpy as np
import os.path as osp
import io

import torch
from torch.utils.data import Dataset

from .transforms import RandomHorizontalFlip_custom

import pdb
from collections import defaultdict
import copy

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""
    def __init__(self, dataset, transform=None, return_path = False):
        self.dataset = dataset
        self.transform = transform
        self.return_path = return_path

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)
        flipped = False
        if self.transform is not None:
            for t in self.transform.transforms:
                #pdb.set_trace()
                if isinstance(t, RandomHorizontalFlip_custom):
                    img, flipped = t(img)
                else:
                    img = t(img)
        #pdb.set_trace()

        if self.return_path:
            return img, pid, camid, img_path
        return img, pid, camid


class ImageDataset_customSampling(Dataset):
    """Image Person ReID Dataset"""
    def __init__(self, dataset, transform=None, return_path = False):
        self.dataset = dataset
        self.transform = transform
        self.return_path = return_path

        self.index_dic = defaultdict(list)
        for index, (_, pid, camid) in enumerate(self.dataset):
            self.index_dic[pid].append((index,camid))
        self.pids = list(self.index_dic.keys())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)
        flipped = False
        if self.transform is not None:
            for t in self.transform.transforms:
                #pdb.set_trace()
                if isinstance(t, RandomHorizontalFlip_rot):
                    img, flipped = t(img)
                else:
                    img = t(img)
        #pdb.set_trace()

        idxs = copy.deepcopy(self.index_dic[pid])
        idxs = [k for k,cam in idxs if cam !=camid]
        idxs = np.random.choice(idxs, size=1, replace=True)

        img_path_second, pid_second, camid_second = self.dataset[idxs[0]]
        img_second = read_image(img_path_second)
        if self.transform is not None:
            for t in self.transform.transforms:
                if isinstance(t, RandomHorizontalFlip_custom):
                    img_second, flipped = t(img_second)
                else:
                    img_second = t(img_second)
        if self.return_path:
            return img, pid, camid, img_path
        return (img, pid, camid), (img_second, pid_second, camid_second)

class VideoDataset(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset, seq_len=15, sample='evenly', transform=None):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        num = len(img_paths)

        if self.sample == 'random':
            """
            Randomly sample seq_len items from num items,
            if num is smaller than seq_len, then replicate items
            """
            indices = np.arange(num)
            replace = False if num >= self.seq_len else True
            indices = np.random.choice(indices, size=self.seq_len, replace=replace)
            # sort indices to keep temporal order (comment it to be order-agnostic)
            indices = np.sort(indices)
        elif self.sample == 'evenly':
            """
            Evenly sample seq_len items from num items.
            """
            if num >= self.seq_len:
                num -= num % self.seq_len
                indices = np.arange(0, num, num/self.seq_len)
            else:
                # if num is smaller than seq_len, simply replicate the last image
                # until the seq_len requirement is satisfied
                indices = np.arange(0, num)
                num_pads = self.seq_len - num
                indices = np.concatenate([indices, np.ones(num_pads).astype(np.int32)*(num-1)])
            assert len(indices) == self.seq_len
        elif self.sample == 'all':
            """
            Sample all items, seq_len is useless now and batch_size needs
            to be set to 1.
            """
            indices = np.arange(num)
        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))

        imgs = []
        for index in indices:
            img_path = img_paths[int(index)]
            img = read_image(img_path)
            if self.transform is not None:
                img = self.transform(img)
            img = img.unsqueeze(0)
            imgs.append(img)
        imgs = torch.cat(imgs, dim=0)

        return imgs, pid, camid
