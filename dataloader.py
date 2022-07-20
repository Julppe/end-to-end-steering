import json
import torch
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
from torchvision import transforms
import numpy as np
import os
import tarfile
import random


# Dataset of single RGB images with steering angles as labels from the a2d2 dataset
class ImgSteeringDataset(Dataset):

    def __init__(self, bus_file, img_dir, transform=None):
        """
        Args:
            bus_file (string): Path to the bus data file with steering angles.
            img_dir (string): Directory with all the images and image specific json files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.bus_file = bus_file
        self.img_dir = img_dir
        self.transform = transform
        self.angles = {}
        self.signs = {}
        self.labels = {}
        self.image_name_start = os.listdir(img_dir)[0][:34]
        self.first_img_num = int(sorted(os.listdir(img_dir))[0][34:43].lstrip("0"))

        bus_file = json.load(open(self.bus_file))
        for stamp, sign in bus_file['steering_angle_calculated_sign']['values']:
            self.signs[int(str(stamp)[:-5])] = int(sign)
        for stamp, angle in bus_file['steering_angle_calculated']['values']:
            if self.signs[int(str(stamp)[:-5])] == 1:
                self.angles[int(str(stamp)[:-5])] = np.pi * angle/180
            else:
                self.angles[int(str(stamp)[:-5])] = np.pi * -angle/180

        for idx in range(int(len(os.listdir(self.img_dir))/2)-1):
            file_number = str(idx + self.first_img_num).rjust(9, '0')

            #tarinfo = tf.getmember(os.path.join(self.img_dir, '20180810150607_camera_frontcenter_'+str(file_number)+'.png'))
            #img_name = tf.extractfile(tarinfo)
            #image = io.imread(img_name)
            #tarinfo_j = tf.getmember(os.path.join(self.img_dir, '20180810150607_camera_frontcenter_'+str(file_number)+'.json'))
            #j_name = tf.extractfile(tarinfo_j)
            #j_data = json.load(j_name)
            #tstamp = j_data['cam_tstamp']
            j_file = open(os.path.join(self.img_dir,self.image_name_start+str(file_number)+'.json'))
            j_data = json.load(j_file)
            tstamp = j_data['cam_tstamp']
        
            label = self.angles[int(str(tstamp)[:-5])]
            self.labels[file_number] = label


    def __len__(self):
        length = int(len(os.listdir(self.img_dir))/2)-1
        return length


    def __getitem__(self, idx):
        file_number = str(idx + self.first_img_num).rjust(9, '0')
        label = self.labels[file_number]
        #with tarfile.open('data/data.tar') as tf:
        #    tarinfo = tf.getmember(os.path.join(self.img_dir, '20180810150607_camera_frontcenter_'+file_number+'.png'))
        #    img_name = tf.extractfile(tarinfo)
        #    image = io.imread(img_name)

        #sample = {'image':image, 'steering_angle':label}

        #if self.transform:
        #    sample = self.transform(sample)

        img_name = os.path.join(self.img_dir, self.image_name_start+str(file_number)+'.png')
        image = io.imread(img_name)

        sample = {'image':image, 'steering_angle':label}
        if self.transform:
            sample = self.transform(sample)

        return sample


# Rescale
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, steer_angle = sample['image'], sample['steering_angle']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))
        return {'image':img, 'steering_angle':steer_angle}


# Crop
class Crop(object):
    """Crop the image in a sample.

    Args:
        output_size (tuple): Desired output size.
        pixels_from_bottom (int): Pixels from the bottom of the image
            to leave out of the cropped image.
            default = 0
    """
    def __init__(self, output_size, pixels_from_bottom=0):
        assert isinstance(output_size, tuple)
        assert isinstance(pixels_from_bottom, int)
        self.output_size = output_size
        self.bottom = pixels_from_bottom

    def __call__(self, sample):
        image, steer_angle = sample['image'], sample['steering_angle']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        img = image[int((h-new_h-self.bottom)):h-self.bottom, int((w-new_w)/2):int(new_w+(w-new_w)/2)]
        return {'image':img, 'steering_angle':steer_angle}


# To tensor
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image, steer_angle = sample['image'], sample['steering_angle']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        return {'image':torch.from_numpy(image), 'steering_angle':torch.tensor([steer_angle])}


# Random horizontal flip
class HorizontalFlip(object):
    """Flip image horizontally and multiply label by -1"""
    def __init__(self):
        self.flip = transforms.RandomHorizontalFlip(p=1)
    def __call__(self, sample):
        image, steer_angle = sample['image'], sample['steering_angle']
        # Randomness performed here instead of on the flip method for label modification
        if random.random() > 0.5:
            img = self.flip(image)
            steer = (-1) * steer_angle
        else:
            img = image
            steer = steer_angle
        return {'image':img, 'steering_angle':steer}


# Random rotation TODO
class RandomRotation(object):
    """Add small random rotation to the image for improved training"""
    def __init__(self):
        # Maximum of 10 degrees rotation
        self.rotate = transforms.RandomRotation(10)
    def __call__(self, sample):
        image, steer_angle = sample['image'], sample['steering_angle']
        # TODO rotate
        img = self.rotate(image)
        return {'image':img, 'steering_angle':steer_angle}


# Random lighting change
class RandomLighting(object):
    """Simulate a lighting change by changing the brightness of the 
    image a random amount"""
    def __init__(self):
        # Random brightness by max 25%
        self.lighting = transforms.ColorJitter(0.25)
    def __call__(self, sample):
        image, steer_angle = sample['image'], sample['steering_angle']
        # Add value in the range of -0.1 and 0.1 if the value does not go negative
        #rand = random.uniform(-0.1, 0.1)
        #if rand < 0:
        #    image[image > rand] += rand
        #else:
        #    image[image < 1-rand] += rand
        img = self.lighting(image)
        return {'image':img, 'steering_angle':steer_angle}


# Class for applying transformations to validation and training sets separately
class MapDataset(Dataset):
    """
    Given a dataset, creates a dataset which applies a mapping function
    to its items (lazily, only when an item is called).

    Note that data is not cloned/copied from the initial dataset.
    """

    def __init__(self, dataset, map_fn):
        self.dataset = dataset
        self.map = map_fn

    def __getitem__(self, index):
        return self.map(self.dataset[index])

    def __len__(self):
        return len(self.dataset)