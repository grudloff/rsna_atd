import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import rsna_atd.config as config
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import Logger

def split_group(group, test_size=0.2):
    """Split a group into train and test sets. The case where the group has only one element is handled separately
       by sampling the element to be in the train or test set with a probability of test_size and 1-test_size
       respectively.
    """
    if test_size == 0:
        return (group, pd.DataFrame())
    if len(group) == 1:
        return (group, pd.DataFrame()) if np.random.rand() < test_size else (pd.DataFrame(), group)
    else:
        return train_test_split(group, test_size=test_size, random_state=42)

class SimpleLogger(Logger):
    """Simple logger for pytorch lightning. It logs metrics and hyperparameters to a dictionary. This logger has
    a similar use as tensorflow history objects."""

    def __init__(self):
        super().__init__()
        self.logged_metrics = {}    

    @property
    def name(self):
        return "CustomLogger"

    @property
    def version(self):
        return "0"

    @rank_zero_only
    def log_hyperparams(self, params):
        # params is an argparse.Namespace
        # your code to record hyperparameters goes here
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        pass

    def log_metrics(self, metrics, step):
        super().log_metrics(metrics, step)
        for key, value in metrics.items():
            if key not in self.logged_metrics:
                self.logged_metrics[key] = []
            self.logged_metrics[key].append(value)