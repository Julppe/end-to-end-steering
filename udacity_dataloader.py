import json
import torch
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
from torchvision import transforms
import numpy as np
import os
import random
import pandas as pd


# Future support for the udacity dataset, current models do not yield good results with it
class UdacityDataset(Dataset):

    def __init__(self, steer_file, img_dir, transform=None):
        """
        Args:
            bus_file (string): Path to the bus data file with steering angles.
            img_dir (string): Directory with all the images and image specific json files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.steer_file = steer_file
        self.img_dir = img_dir
        self.transform = transform
        self.samples = pd.DataFrame(columns=['tstamp', 'steering_angle', 'image'])

        steer_angles = pd.read_csv(steer_file)

        for img_file in os.listdir(img_dir):
            tstamp = img_file[:11]
            angle = float(steer_angles[steer_angles['timestamp'] < int(tstamp + '99999999')][steer_angles['timestamp'] > int(tstamp + '00000000')].iloc[0]['angle'])
            self.samples.loc[len(self.samples)] = {'tstamp':tstamp, 'steering_angle':angle, 'image':img_file}

    def __len__(self):
        length = int(len([name for name in os.listdir(self.img_dir) if os.path.isfile(os.path.join(self.img_dir, name))]))
        return length


    def __getitem__(self, idx):
        row = self.samples.iloc[idx]
        image = io.imread(os.path.join(self.img_dir, row['image']))
        label = row['steering_angle']
        sample = {'image':image, 'steering_angle':label}

        if self.transform:
            sample = self.transform(sample)

        return sample

