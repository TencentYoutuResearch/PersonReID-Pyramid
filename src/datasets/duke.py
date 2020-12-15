#!/usr/bin/env python
# -*- coding: utf-8 -*-
import collections
import os
import random
import re

from torch.utils.data import dataset, sampler
from torchvision.datasets.folder import default_loader
from datasets.base_dataset import BaseDataset


class Duke(BaseDataset):
    """
    Attributes:
        imgs (list of str): dataset image file paths
        _id2label (dict): mapping from person id to softmax continuous label
    """

    @staticmethod
    def id(file_path):
        """
        :param file_path: unix style file path
        :return: person id
        """
        return int(file_path.split('/')[-1].split('_')[0])

    @staticmethod
    def camera(file_path):
        """
        :param file_path: unix style file path
        :return: camera id
        """
        return int(file_path.split('/')[-1].split('_')[1][1])
