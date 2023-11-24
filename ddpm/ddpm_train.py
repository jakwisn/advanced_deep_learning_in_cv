import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import random
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision
from tqdm import tqdm
from torch import optim
import logging

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

#from ddpm import Diffusion
from ddpm.ddpm import Diffusion
from ddpm.model import UNet
from loss import MaskedMSELoss

SEED = 1
DATASET_SIZE = 40000

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def save_images(images, path, show=True, title=None, nrow=10):
    grid = torchvision.utils.make_grid(images, nrow=nrow)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    if title is not None:
        plt.title(title)
    plt.imshow(ndarr)
    plt.axis('off')
    if path is not None:
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
    if show:
        plt.show()
    plt.close()

def prepare_dataloader(batch_size):
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    from ddpm.dataset.sprites_dataset import SpritesDataset
    transform = transforms.Compose([
    transforms.ToTensor(),                # from [0,255] to range [0.0,1.0]
    transforms.Normalize((0.5,), (0.5,))  # range [-1,1]

    ])
    dataset = SpritesDataset(transform, num_samples=DATASET_SIZE, seed=SEED)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def create_result_folders(experiment_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", experiment_name), exist_ok=True)
    os.makedirs(os.path.join("results", experiment_name), exist_ok=True)

def train(dataloader, valdataloader, device='cpu', T=500, img_size=16, input_channels=3, channels=32, time_dim=256,
          batch_size=100, lr=1e-3, num_epochs=30, experiment_name="ddpm", show=False):
    """Implements algrorithm 1 (Training) from the ddpm paper at page 4"""
    create_result_folders(experiment_name)
    
    if not dataloader:
        dataloader = prepare_dataloader(batch_size)

    model = UNet(img_size=img_size, c_in=input_channels, c_out=input_channels, 
                 time_dim=time_dim,channels=channels, device=device).to(device)
    diffusion = Diffusion(img_size=img_size, T=T, beta_start=1e-4, beta_end=0.02, device=device)
    
    ckpt = torch.load("checkpoints/unconditional_ckpt.pt")
    model.load_state_dict(ckpt)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    #loss_fn = torch.nn.MSELoss() # use MSE loss 
    loss_fn = MaskedMSELoss()
    
    logger = SummaryWriter(os.path.join("runs", experiment_name))
    l = len(dataloader)
    MAX_STEPS = 150
    for epoch in range(1, num_epochs + 1):
        logging.info(f"Starting epoch {epoch}:")
        
        pbar = tqdm(dataloader, total=MAX_STEPS)

        with torch.enable_grad():
            for i, (images, masks) in enumerate(pbar):
                images = images.to(device)
                masks = masks.to(device).unsqueeze(1)

                t = diffusion.sample_timesteps(images.shape[0]).to(device) # line 3 from the Training algorithm

                # q sample does is done just like before
                x_t, noise = diffusion.q_sample(images, t)

                known_regions = x_t * ~masks
                x_p = diffusion.p_sample(model, x_t, t)

                unknown_regions = x_p * masks

                x_t = known_regions + unknown_regions

                predicted_noise = model(x_t, t) # predict noise of x_t using the UNet
                unknown_regions = predicted_noise * masks

                loss = loss_fn(noise, unknown_regions, masks) # loss between masked noise and masked predicted noise

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_postfix(Loss=loss.item())
                logger.add_scalar("Loss", loss.item(), global_step=epoch * l + i)
                
                if i >= MAX_STEPS:
                    break
             
        with torch.no_grad():
            pbar = tqdm(valdataloader)
            for i, (images, masks) in enumerate(pbar):
                images = images.to(device)
                masks = masks.to(device)
            sampled_images = diffusion.p_sample_loop(images, masks, model,  batch_size=images.shape[0])
            save_images(images=sampled_images, path=os.path.join("results", experiment_name, f"{epoch}.jpg"),
                        show=show, title=f'Epoch {epoch}')

        torch.save(model.state_dict(), os.path.join("models", experiment_name, f"weights-{epoch}.pt"))


def train_profiler(dataloader, valdataloader, device='cpu', T=500, img_size=16, input_channels=3, channels=32, time_dim=256,
          batch_size=100, lr=1e-3, num_epochs=30, experiment_name="ddpm", show=False):
    """Implements algrorithm 1 (Training) from the ddpm paper at page 4"""
    create_result_folders(experiment_name)

    if not dataloader:
        dataloader = prepare_dataloader(batch_size)

    model = UNet(img_size=img_size, c_in=input_channels, c_out=input_channels,
                 time_dim=time_dim, channels=channels, device=device).to(device)
    diffusion = Diffusion(img_size=img_size, T=T, beta_start=1e-4, beta_end=0.02, device=device)

    ckpt = torch.load("checkpoints/unconditional_ckpt.pt")
    model.load_state_dict(ckpt)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    mse = torch.nn.MSELoss()  # use MSE loss

    logger = SummaryWriter(os.path.join("runs", experiment_name))
    l = len(dataloader)

    # PROFILER
    wait = 1
    warmup = 1
    active = 3
    repeat = 5
    prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./prof/train'),
        record_shapes=True,
        with_stack=False)
    prof.start()
    prof_step = 0

    for epoch in range(1, num_epochs + 1):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)

        with torch.enable_grad():
            for i, (images, masks) in enumerate(pbar):

                # Torch profiler
                torch.cuda.synchronize()

                images = images.to(device)
                masks = masks.to(device)

                # Torch profiler
                torch.cuda.synchronize()

                t = diffusion.sample_timesteps(images.shape[0]).to(device)  # line 3 from the Training algorithm

                # q sample does is done just like before
                x_t, noise = diffusion.q_sample(images, t)
                # Torch profiler
                torch.cuda.synchronize()

                known_regions = x_t * ~masks.unsqueeze(1)
                x_p = diffusion.p_sample(model, x_t, t)

                # Torch profiler
                torch.cuda.synchronize()

                unknown_regions = x_p * masks.unsqueeze(1)

                x_t = known_regions + unknown_regions

                # Torch profiler
                torch.cuda.synchronize()

                predicted_noise = model(x_t, t)  # predict noise of x_t using the UNet

                # Torch profiler
                torch.cuda.synchronize()

                unknown_regions = predicted_noise * masks.unsqueeze(1)

                loss = mse(noise * masks.unsqueeze(1),
                           unknown_regions)  # loss between masked noise and masked predicted noise

                # Torch profiler
                torch.cuda.synchronize()

                optimizer.zero_grad()
                loss.backward()
                # Torch profiler
                torch.cuda.synchronize()

                optimizer.step()

                pbar.set_postfix(MSE=loss.item())
                logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)
                prof_step += 1
                prof.step()

                if prof_step >= (wait + warmup + active) * repeat:
                    break
        print('Stopped profiling')
        prof.stop()
        break

        with torch.no_grad():
            pbar = tqdm(valdataloader)
            for i, (images, masks) in enumerate(pbar):
                images = images.to(device)
                masks = masks.to(device)
            sampled_images = diffusion.p_sample_loop(images, masks, model, batch_size=images.shape[0])
            save_images(images=sampled_images, path=os.path.join("results", experiment_name, f"{epoch}.jpg"),
                        show=show, title=f'Epoch {epoch}')

        torch.save(model.state_dict(), os.path.join("models", experiment_name, f"weights-{epoch}.pt"))


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    print(f"Model will run on {device}")
    set_seed(seed=SEED)
    input_images = glob.glob('data/training/INPUT_IMAGES/*P1.5.JPG')
    train_dataset = CorrectionImageDataset(input_images, train=True, transform=transform)
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    train(dataloader, device=device, T=250, img_size=IM_SIZE, batch_size=BATCH_SIZE)

if __name__ == '__main__':
    main()
    

        