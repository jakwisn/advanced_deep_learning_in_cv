import torchvision.transforms.v2.functional as F
import torchvision.transforms.v2 as transforms
import cv2

IMAGE_SIZE = 128

def get_train_transforms():
    # v2.RandomCrop(size=(128, 128)
    return transforms.Compose([transforms.ToTensor(), 
                               transforms.RandomCrop(size=(IMAGE_SIZE, IMAGE_SIZE))
                              ])

def get_test_transforms():
    return transforms.Compose([transforms.ToTensor(), 
                               transforms.Resize(size=(IMAGE_SIZE, IMAGE_SIZE))
                              ])
