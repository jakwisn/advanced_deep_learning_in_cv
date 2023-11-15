import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2

class CorrectionImageDataset(Dataset):
    def __init__(self, file_list, transform=None, return_og=False):
        self.transform = transform
        self.file_list = file_list
        self.return_og = return_og

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        image = increase_brightness(Image.open(img_name))
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)  

        if self.return_og:
            return self.hist_fix(image), image
        else:
            return self.hist_fix(image)
        
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
