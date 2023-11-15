from dataset import CorrectionImageDataset
import torch
from torch.utils.data import Dataset, DataLoader
from ddpm.ddpm import *
from ddpm.ddpm_train import *
import glob
from torchvision import transforms
from torchvision.transforms import v2

def main():
    BATCH_SIZE=2
    RESIZE_SIZE = 256
    IM_SIZE=64
    
    transform = transforms.Compose([
        transforms.ToTensor(),                # from [0,255] to range [0.0,1.0]
        v2.Resize(size=(RESIZE_SIZE, RESIZE_SIZE)),
        v2.RandomCrop(size=(IM_SIZE, IM_SIZE))
    ])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    print(f"Model will run on {device}")

    input_images = glob.glob('data/training/INPUT_IMAGES/*P1.5.JPG')
    train_dataset = CorrectionImageDataset(input_images, train=True, transform=transform)
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    train(dataloader, device=device, T=250, img_size=IM_SIZE, batch_size=BATCH_SIZE)

if __name__ == '__main__':
    main()