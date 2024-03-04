import torch
from tqdm import tqdm
import logging
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, T=500, beta_start=1e-4, beta_end=0.02, img_size=16, device="cuda"):
        """
        T : total diffusion steps (X_T is pure noise N(0,1))
        beta_start: value of beta for t=0
        b_end: value of beta for t=T
        """

        self.T = T
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        
        # TASK 1: Implement beta, alpha, and alpha_bar
        self.betas = self.get_betas('cosine').to(device)
        self.alphas = 1. - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0) # cumulative products of alpha 


    def get_betas(self, schedule='linear'):
        if schedule == 'linear':
            # HINT: use torch.linspace to create a linear schedule from beta_start to beta_end
            return torch.linspace(self.beta_start, self.beta_end, self.T)
        # add your own (e.g. cosine)
        elif schedule == 'cosine':
            def f(t):
                s=0.008
                return torch.cos((t / self.T + s) / (1 + s) * 0.5 * torch.pi) ** 2
            x = torch.linspace(0, self.T, self.T + 1)
            alphas_cumprod = f(x) / f(torch.tensor([0]))
            betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
            betas = torch.clip(betas, 0.0001, 0.999)
            return betas
        else :
            raise NotImplementedError('Not implemented!')
    

    def q_sample(self, x, t, mask=None):
        """
        x: input image (x0)
        t: timestep: should be torch.tensor

        Forward diffusion process
        q(x_t | x_0) = sqrt(alpha_hat_t) * x0 + sqrt(1-alpha_hat_t) * N(0,1)

        Should return q(x_t | x_0), noise
        """
        # TASK 2: Implement the forward process
        sqrt_alpha_bar =  torch.sqrt(self.alphas_bar).to(self.device)
        sqrt_alpha_bar = sqrt_alpha_bar[:, None, None, None] # match image dimensions
        
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alphas_bar).to(self.device)
        sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar[:, None, None, None]
        
        noise = torch.normal(0, 1, size=x.size()).to(self.device)
        assert noise.shape == x.shape, 'Invalid shape of noise'
        
        x_noised = sqrt_alpha_bar[t] * x + sqrt_one_minus_alpha_bar[t]*noise
        return x_noised, noise
    

    def p_mean_std(self, model, x_t, t):
        """
        Calculate mean and std of p(x_{t-1} | x_t) using the reverse process and model
        """
        alpha = self.alphas[t][:, None, None, None] # match image dimensions
        alpha_bar = self.alphas_bar[t][:, None, None, None] # match image dimensions 
        beta = self.betas[t][:, None, None, None] # match image dimensions

        # TASK 3 : Implement the revese process
        predicted_noise = model(x_t,t) # HINT: use model to predict noise
        mean = 1/torch.sqrt(alpha).to(self.device) * (x_t - (beta / torch.sqrt(1-alpha_bar).to(self.device)) * predicted_noise) # HINT: calculate the mean of the distribution p(x_{t-1} | x_t). See Eq. 11 in the ddpm paper at page 4
        std = torch.sqrt(beta).to(self.device)

        return mean, std

    def p_sample(self, model, x_t, t):
        """
        Sample from p(x{t-1} | x_t) using the reverse process and model
        """
        # TASK 3: implement the reverse process
        mean, std = self.p_mean_std(model, x_t, t)

        noise = torch.normal(0, 1, size=x_t.shape).to(self.device) if t[0] > 1 else torch.zeros(size=x_t.shape).to(self.device)
        # Calculate x_{t-1}, see line 4 of the Algorithm 2 (Sampling) at page 4 of the ddpm paper.
        # betas here? # TODO

        x_t_prev = 1/torch.sqrt(self.alphas[t]).view(x_t.shape[0], 1, 1, 1) * (x_t -((1- self.alphas[t]) / torch.sqrt(1-self.alphas_bar[t])).view(x_t.shape[0], 1, 1, 1) * model(x_t, t)) +  std*noise
        return x_t_prev


    def p_sample_loop(self, images, masks, model, batch_size , timesteps_to_save=None):
        """
        Implements algrorithm 2 (Sampling) from the ddpm paper at page 4

        # xq is an reverse masked image from q sample
        """
        logging.info(f"Sampling {batch_size} new images....")
        model.eval()
        if timesteps_to_save is not None:
            intermediates = []
        with torch.no_grad():
            x = torch.randn((batch_size, 3, self.img_size, self.img_size)).to(self.device) # Xt ~ N(0,1)

            for i in tqdm(reversed(range(1, self.T)), position=0, total=self.T-1):
                t = (torch.ones(batch_size) * i).long().to(self.device)

                known, _ = self.q_sample(images * ~ masks, t)
                knonw_regions = known * ~ masks


                x = self.p_sample(model, x, t) # sample from p(x_{t-1} | x_t) - here the previous iteration goes
                unknown_regions = x * masks  # mask the image and add the known regions
                x = knonw_regions + unknown_regions # add the known regions to the unknown regions


                if timesteps_to_save is not None and i in timesteps_to_save:
                    x_itermediate = (x.clamp(-1, 1) + 1) / 2
                    x_itermediate = (x_itermediate * 255).type(torch.uint8)
                    intermediates.append(x_itermediate)

        model.train()
        # x = (x.clamp(-1, 1) + 1) / 2
        # x = (x * 255).type(torch.uint8)

        if timesteps_to_save is not None:
            intermediates.append(x)
            return x, intermediates
        else :
            return x


    def sample_timesteps(self, batch_size):
        """
        Sample timesteps uniformly for training
        """
        return torch.randint(low=1, high=self.T, size=(batch_size,), device=self.device)