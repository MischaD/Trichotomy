from pathlib import Path
from edm2.generate_images import edm_sampler, StackedRandomGenerator

import torch.distributed as dist
import pickle
import dnnlib
import torch
import os
import tqdm
import numpy as np
from ..utils import export_image
import sys
from torchvision.transforms import functional as F
import torch


class ImageIterableDSE:
    def __init__(self, 
                 train_ds, 
                 device, 
                 net, 
                 sampler_fn, 
                 gnet, 
                 encoder, 
                 outdir=None, 
                 verbose=False, 
                 sampler_kwargs={},
                 indices=[],
                 max_batch_size=32, 
                 add_seed_to_path=True, 
                 dse=None):
        self.train_ds = train_ds
        self.device = device
        self.net = net
        self.sampler_fn = sampler_fn
        self.gnet = gnet
        self.encoder = encoder
        self.outdir = outdir
        self.verbose = verbose
        self.max_batch_size = max_batch_size
        self.sampler_kwargs = sampler_kwargs
        self.guidance_strength = self.sampler_kwargs["guidance"]

        # Prepare seeds and batches
        self.num_batches = max((len(indices) - 1) // max_batch_size + 1, 1)
        self.rank_batches = np.array_split( np.arange(len(indices)), self.num_batches)
        self.indices = np.array_split(np.array(indices), self.num_batches)
        self.add_seed_to_path = add_seed_to_path

        self.dse = dse

        if verbose:
            print(f'Generating {len(self.seeds)} images...')

    def __len__(self):
        return len(self.rank_batches)

    def __iter__(self):

        for batch_idx in range(len(self.rank_batches)):
            # one batch only consists of one single image!
            image_generated = False
            self.sampler_kwargs["guidance"] = self.guidance_strength
            indices = self.indices[batch_idx]
            r = dnnlib.EasyDict(images=None, labels=None, noise=None, 
                                batch_idx=batch_idx, 
                                num_batches=len(self.rank_batches), 
                                indices=indices, 
                                paths=None, 
                                memorization_prediction=[], 
                                consistency_score=[],
                                real_image=None,
                                )
            r.seeds =  self.rank_batches[batch_idx] 

            while not image_generated: 
                if len(r.seeds) > 0:
                    while not image_generated:
                        # Generate noise and labels
                        rnd = StackedRandomGenerator(self.device, r.seeds)
                        r.noise = rnd.randn([len(r.seeds), self.net.img_channels, self.net.img_resolution, self.net.img_resolution], device=self.device)
                        r.labels = torch.stack([self.train_ds.get_label(x) for x in r.indices]).to(self.device)
                        r.paths = [self.train_ds.file_list[x] for x in r.indices]

                        # Generate images
                        latents = dnnlib.util.call_func_by_name(func_name=self.sampler_fn, net=self.net, noise=r.noise,
                                                                labels=r.labels, gnet=self.gnet, randn_like=rnd.randn_like, **self.sampler_kwargs)

                        assert r.indices[0] == r.indices[-1], "Wrong indices in sampler"
                        real_image_latent = self.encoder.encode_latents(self.train_ds[r.indices[0]][0]).to(self.device)

                        mixed_latents = self.encoder.decode(torch.cat([real_image_latent, latents]))
                        mixed_images = mixed_latents.float() / 255.
                        r.real_image = mixed_images[0]
                        r.images = mixed_images[1:]

                        clf_pred_scores, priv_pred = self.dse.lazy_predict(mixed_images)
                        r.memorization_prediction.extend([x.item() for x in priv_pred])
                        r.consistency_score.extend([x.item() for x in clf_pred_scores])

                        print(f"Memorization Rate {r.paths[0]}: {priv_pred.float().mean()}")
                        if priv_pred.min() < 1: 
                            clf_pred_scores = clf_pred_scores + priv_pred
                            image_generated = True
                            idx = clf_pred_scores.argmin()
                            path_real = r.paths[idx]
                            image_pth = os.path.join(self.outdir, path_real)
                            os.makedirs(os.path.dirname(image_pth), exist_ok=True)

                            export_image(r.images[idx], image_pth)
                            if (1 - priv_pred).sum() > 1:  # also export second best as augmentation technique
                                clf_pred_scores[idx] = 1 # 1 is max 
                                idx = clf_pred_scores.argmin()
                                path_real = r.paths[idx]
                                os.makedirs(os.path.dirname(image_pth), exist_ok=True)

                                name, ending = ".".join(path_real.split(".")[:-1]), path_real.split(".")[-1]
                                image_pth = os.path.join(self.outdir, name + "_2nd." + ending)
                                export_image(r.images[idx], image_pth)


                        if not image_generated:
                            r.seeds = r.seeds + self.max_batch_size
                            print(f"only memorized for guidance: {self.sampler_kwargs['guidance']} and path: {r.paths[0]}")
                            self.sampler_kwargs["guidance"] = self.sampler_kwargs["guidance"] - 0.1



            # Yield results
            yield r


class ImageIterable:
    def __init__(self, 
                 train_ds, 
                 device, 
                 net, 
                 sampler_fn, 
                 gnet, 
                 encoder, 
                 outdir=None, 
                 verbose=False, 
                 sampler_kwargs={},
                 indices=[],
                 max_batch_size=32, 
                 add_seed_to_path=True):
        self.train_ds = train_ds
        self.device = device
        self.net = net
        self.sampler_fn = sampler_fn
        self.gnet = gnet
        self.encoder = encoder
        self.outdir = outdir
        self.verbose = verbose
        self.max_batch_size = max_batch_size
        self.sampler_kwargs = sampler_kwargs

        # Prepare seeds and batches
        self.num_batches = max((len(indices) - 1) // max_batch_size + 1, 1)
        self.rank_batches = np.array_split( np.arange(len(indices)), self.num_batches)
        self.indices = np.array_split(np.array(indices), self.num_batches)
        self.add_seed_to_path = add_seed_to_path

        if verbose:
            print(f'Generating {len(self.seeds)} images...')

    def __len__(self):
        return len(self.rank_batches)

    def __iter__(self):
        for batch_idx in range(len(self.rank_batches)):
            indices = self.indices[batch_idx]
            r = dnnlib.EasyDict(images=None, labels=None, noise=None, 
                                batch_idx=batch_idx, num_batches=len(self.rank_batches), 
                                indices=indices, paths=None)
            r.seeds =  self.rank_batches[batch_idx] 
            if len(r.seeds) > 0:
                # Generate noise and labels
                rnd = StackedRandomGenerator(self.device, r.seeds)
                r.noise = rnd.randn([len(r.seeds), self.net.img_channels, self.net.img_resolution, self.net.img_resolution], device=self.device)
                r.labels = torch.stack([self.train_ds.get_label(x) for x in r.indices]).to(self.device)
                r.paths = [self.train_ds.file_list[x] for x in r.indices]

                # Generate images
                latents = dnnlib.util.call_func_by_name(func_name=self.sampler_fn, net=self.net, noise=r.noise,
                                                        labels=r.labels, gnet=self.gnet, randn_like=rnd.randn_like, **self.sampler_kwargs)
                r.images = self.encoder.decode(latents)

                # Save images
                if self.outdir is not None:
                    for path, image, seed in zip(r.paths, r.images.permute(0, 2, 3, 1).cpu().numpy(), r.seeds):
                        file_name = "".join(path.split(".")[:-1]) 
                        if self.add_seed_to_path: 
                            file_name += f"_seed_{seed}.png"
                        else: 
                            file_name += ".png"
                        image_pth = os.path.join(self.outdir, file_name)

                        os.makedirs(os.path.dirname(image_pth), exist_ok=True)
                        PIL.Image.fromarray(image, 'RGB').save(image_pth)

            # Yield results
            yield r


def get_image_generation_model(path_net, path_gnet, model_weights, gmodel_weights, name, device=None): 
    if device is None: 
        device = "cuda"

    encoder_batch_size = 4
        # Rank 0 goes first.
    net = path_net
    gnet = path_gnet

    # Load main network.
    if isinstance(net, str):
        print(f'Loading network from {net} ...')
        with dnnlib.util.open_url(net, verbose=True) as f:
            data = pickle.load(f)
        net = data['ema'].to(device)
        net.load_state_dict(torch.load(model_weights)["net"])
        
        encoder = data.get('encoder', None)
        encoder_mode = encoder.init_kwargs.encoder_norm_mode
        encoder = dnnlib.util.construct_class_by_name(class_name='edm2.training.encoders.StabilityVAEEncoder', vae_name=encoder.init_kwargs.vae_name, encoder_norm_mode=encoder_mode)
        print(f"Encoder was initilized with {encoder._init_kwargs}")

    assert net is not None

    # Load guidance network.
    if isinstance(gnet, str):
        print(f'Loading guidance network from {gnet} ...')
        with dnnlib.util.open_url(gnet, verbose=True) as f:
            data = pickle.load(f)
        gnet = data['ema'].to(device)
        gnet.load_state_dict(torch.load(gmodel_weights)["net"])

    assert gnet is not None

    # Initialize encoder.
    assert encoder is not None
    print(f'Setting up {type(encoder).__name__}...')
    encoder.init(device)
    if encoder_batch_size is not None and hasattr(encoder, 'batch_size'):
        encoder.batch_size = encoder_batch_size

    return net, gnet, encoder
