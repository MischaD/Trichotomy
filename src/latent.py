import torch
import torchvision.transforms as transforms
import torchvision.io as io
from diffusers import AutoencoderKL
from einops import repeat


def compute_latent_representation(input_tensor, model, batch_size):
    # L x C x H x W
    latent_representations = []
    L = input_tensor.shape[0]

    for i in range(0, L, batch_size):
        end = min(i + batch_size, L)
        batch = input_tensor[i:end]
        
        with torch.no_grad():
            latent = model.encode(batch).latent_dist.sample()
        
        latent_representations.append(latent)
    
    latent_representations = torch.cat(latent_representations, dim=0)
    return latent_representations

def decode_latent_representation(latent_representations, model):
    latent_representations = latent_representations
    
    with torch.no_grad():
        decoded_video = model.decode(latent_representations).sample
    
    return decoded_video

def get_latent_model(path=None):
    # Load the VQ-VAE model
    if path is None: 
        path = "stabilityai/stable-diffusion-2"
    model = AutoencoderKL.from_pretrained(path, subfolder="vae")
    model = model.to("cuda")
    model.eval()
    return model