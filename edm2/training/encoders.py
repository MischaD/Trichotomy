# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Converting between pixel and latent representations of image data."""

import os
import warnings
import numpy as np
import torch
from torch_utils import persistence
from torch_utils import misc

warnings.filterwarnings('ignore', 'torch.utils._pytree._register_pytree_node is deprecated.')
warnings.filterwarnings('ignore', '`resume_download` is deprecated')

#----------------------------------------------------------------------------
# Abstract base class for encoders/decoders that convert back and forth
# between pixel and latent representations of image data.
#
# Logically, "raw pixels" are first encoded into "raw latents" that are
# then further encoded into "final latents". Decoding, on the other hand,
# goes directly from the final latents to raw pixels. The final latents are
# used as inputs and outputs of the model, whereas the raw latents are
# stored in the dataset. This separation provides added flexibility in terms
# of performing just-in-time adjustments, such as data whitening, without
# having to construct a new dataset.
#
# All image data is represented as PyTorch tensors in NCHW order.
# Raw pixels are represented as 3-channel uint8.

@persistence.persistent_class
class Encoder:
    def __init__(self):
        pass

    def init(self, device): # force lazy init to happen now
        pass

    def __getstate__(self):
        return self.__dict__

    def encode(self, x): # raw pixels => final latents
        return self.encode_latents(self.encode_pixels(x))

    def encode_pixels(self, x): # raw pixels => raw latents
        raise NotImplementedError # to be overridden by subclass

    def encode_latents(self, x): # raw latents => final latents
        raise NotImplementedError # to be overridden by subclass

    def decode(self, x): # final latents => raw pixels
        raise NotImplementedError # to be overridden by subclass

#----------------------------------------------------------------------------
# Standard RGB encoder that scales the pixel data into [-1, +1].

@persistence.persistent_class
class StandardRGBEncoder(Encoder):
    def __init__(self):
        super().__init__()

    def encode_pixels(self, x): # raw pixels => raw latents
        return x

    def encode_latents(self, x): # raw latents => final latents
        return x.to(torch.float32) / 127.5 - 1

    def decode(self, x): # final latents => raw pixels
        return (x.to(torch.float32) * 127.5 + 128).clip(0, 255).to(torch.uint8)

#----------------------------------------------------------------------------
# Pre-trained VAE encoder from Stability AI.
STATS = {
    "imgnet":  {'raw_means':[5.81, 3.25, 0.12, -2.14],   'raw_std': [4.17, 4.62, 3.71, 3.28]},
    "cxr14":   {'raw_means':[6.98, 2.89, 0.94, -2.73],   'raw_std': [3.12, 4.18, 2.9, 2.61]},# same as cxr8 but all images and different split
    "celebvqh":{'raw_means':[6.85, 0.24, 0.98, -1.02],   'raw_std': [3.36, 4.48, 3.88, 2.6]},
    "echonet": {'raw_means':[5.77, -1.66, 2.39, -0.13],  'raw_std': [3.3, 4.26, 3.46, 2.09]},
    "ffhq":    {'raw_means':[5.87, 3.02, -0.88, -2.45],  'raw_std': [3.61, 4.3, 3.79, 2.93]},
    "cxr8":    {'raw_means':[ 7.00,  2.91,  0.94, -2.75],'raw_std': [3.13, 4.18, 2.89, 2.61]}, # same as cxr14 but only single disease images/different split
    "chexpert":{'raw_means':[ 5.58, 3.63, 0.30, -3.32],   'raw_std': [3.50, 4.76, 2.96, 3.16]},
    "mimic":   {'raw_means':[ 6.00,  3.30,  0.56, -3.11], 'raw_std': [3.26, 5.24, 3.06, 3.32]}
}


@persistence.persistent_class
class StabilityVAEEncoder(Encoder):
    def __init__(self,
        vae_name    = 'stabilityai/sd-vae-ft-mse',  # Name of the VAE to use.
        raw_mean    = [5.81, 3.25, 0.12, -2.15],    # Assumed mean of the raw latents.
        raw_std     = [4.17, 4.62, 3.71, 3.28],     # Assumed standard deviation of the raw latents.
        final_mean  = 0,                            # Desired mean of the final latents.
        final_std   = 0.5,                          # Desired standard deviation of the final latents.
        batch_size  = 8,                            # Batch size to use when running the VAE.
        encoder_norm_mode = None, # use-predefined rawmean and raw stds
    ):
        super().__init__()
        self.vae_name = vae_name
        if encoder_norm_mode is not None: 
            raw_mean = STATS[encoder_norm_mode]["raw_means"]
            raw_std = STATS[encoder_norm_mode]["raw_std"]
        self.scale = np.float32(final_std) / np.float32(raw_std)
        self.bias = np.float32(final_mean) - np.float32(raw_mean) * self.scale
        self.batch_size = int(batch_size)
        self._vae = None

    def init(self, device): # force lazy init to happen now
        super().init(device)
        if self._vae is None:
            self._vae = load_stability_vae(self.vae_name, device=device)
        else:
            self._vae.to(device)

    def __getstate__(self):
        return dict(super().__getstate__(), _vae=None) # do not pickle the vae

    def _run_vae_encoder(self, x):
        d = self._vae.encode(x).latent_dist.sample()
        return d 

    def _run_vae_decoder(self, x):
        return self._vae.decode(x)['sample']

    def encode_pixels(self, x): # raw pixels => raw latents
        self.init(x.device)
        x = x.to(torch.float32) / 255
        x = torch.cat([self._run_vae_encoder(batch) for batch in x.split(self.batch_size)])
        return x

    def encode_latents(self, x): # raw latents => final latents
        x = x * misc.const_like(x, self.scale).reshape(1, -1, 1, 1)
        x = x + misc.const_like(x, self.bias).reshape(1, -1, 1, 1)
        return x

    def decode(self, x): # final latents => raw pixels
        self.init(x.device)
        x = x.to(torch.float32)
        x = x - misc.const_like(x, self.bias).reshape(1, -1, 1, 1)
        x = x / misc.const_like(x, self.scale).reshape(1, -1, 1, 1)
        x = torch.cat([self._run_vae_decoder(batch) for batch in x.split(self.batch_size)])
        x = x.clamp(0, 1).mul(255).to(torch.uint8)
        return x

#----------------------------------------------------------------------------

def load_stability_vae(vae_name='stabilityai/sd-vae-ft-mse', device=torch.device('cpu')):
    from diffusers import AutoencoderKL

    model = AutoencoderKL.from_pretrained(vae_name, subfolder="vae")

    return model.eval().requires_grad_(False).to(device)
#----------------------------------------------------------------------------