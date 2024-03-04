import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import albumentations as A
import scipy

class CorrectionImageDataset(Dataset):
    def __init__(self, file_list, train=True,transform=None):
        self.transform = transform
        self.file_list = file_list
        self.train = train

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)  

        # Get mask
        transform = A.RandomBrightnessContrast(p=1, brightness_limit=(0.2,0.2), contrast_limit=(0.1,0.1)) # Contrast limit has to be set otherwise it fluctuates
        bright_image = torch.from_numpy(transform(image=image.numpy())['image'])
        mask = torch.mean(bright_image,0) >= 255 / 255
        #print(mask.shape)
        # Use dilation operation
        # expanded_mask = scipy.ndimage.binary_dilation(mask, iterations=3)
        # expanded_mask = transforms.ToTensor()(expanded_mask).squeeze(0)
        #print(expanded_mask.shape)

        if self.train:
        # return
            return image, mask
        else:
            return self.hist_fix(bright_image), mask
        
    def hist_fix(self, im):
        new_im = im.clone()
        
        new_im[0] = (im[0] - torch.min(im[0])) / (torch.max(im[0])-torch.min(im[0]))
        new_im[1] = (im[1] - torch.min(im[1])) / (torch.max(im[1])-torch.min(im[1]))
        new_im[2] = (im[2] - torch.min(im[2])) / (torch.max(im[2])-torch.min(im[2]))

        return new_im
    
def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
    return img
