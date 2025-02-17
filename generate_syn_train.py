import sys
import os
import argparse 
from pprint import pprint
import torch 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn.functional as F
from einops import repeat
from tqdm import tqdm
import pickle
import seaborn as sns
from PIL import Image
from torchvision import transforms
from edm2.training.dataset import LatentDataset
import edm2.dnnlib as dnnlib
from edm2.generate_images import StackedRandomGenerator, edm_sampler


def load_latents(basepath, model_name, n_splits=1):
    """load .pt latent for <model_name>. Assumes only n_splits .pt for each model"""
    path = os.path.join(basepath, model_name)
    files = os.listdir(path)
    pts = [x for x in files if x.endswith(".pt") ]
    csv = [x for x in files if x.endswith(".csv")]
    assert len(pts) == n_splits and len(csv) == n_splits, f"Unexpected number of csv/pts found in {path}"

    feature_path = os.path.join(path, pts[0])
    print(f"Loading features in {feature_path}")
    features = torch.load(feature_path)
    paths = pd.read_csv(os.path.join(path, csv[0]))
    return features, paths


def update_matplotlib_font(fontsize=11, fontsize_ticks=8, tex=True, scale=1):
    import matplotlib.pyplot as plt
    fontsize = scale * fontsize
    fontsize_ticks = scale * fontsize_ticks
    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": tex,
        "font.family": "serif",
        # Use 11pt font in plots, to match 11pt font in document
        "axes.labelsize": fontsize,
        "font.size": fontsize,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": fontsize_ticks,
        
        "xtick.labelsize": fontsize_ticks,
        "ytick.labelsize": fontsize_ticks
    }
    plt.rcParams.update(tex_fonts)


class ToTensorIfNotTensor:
    def __call__(self, input):
        if isinstance(input, torch.Tensor):
            return input
        return F.to_tensor(input)


def get_classification_model(model_path): 
    global class_labels
    import os
    import numpy as np

    import torch
    import torch.nn as nn
    import torch.backends.cudnn as cudnn

    import torchvision.transforms as T 
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from sklearn.metrics import roc_auc_score

    import torchvision

    class DenseNet121(nn.Module):

        def __init__(self, classCount, isTrained):
        
            super(DenseNet121, self).__init__()
            
            self.densenet121 = torchvision.models.densenet121(pretrained=isTrained)

            kernelCount = self.densenet121.classifier.in_features
            
            self.densenet121.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

        def forward(self, x):
            x = self.densenet121(x)
            return x

    cudnn.benchmark = True
    
    #-------------------- SETTINGS: NETWORK ARCHITECTURE, MODEL LOAD
    model = DenseNet121(len(class_labels), True).cuda()
    model = model.cuda() 

    modelCheckpoint = torch.load(model_path)
    state_dict = {k[7:]:v for k, v in modelCheckpoint['state_dict'].items()}
    model.load_state_dict(state_dict)


    class Classifier(nn.Module): 
        def __init__(self, model, transforms="default") -> None:
            super().__init__()
            if transforms == "default": 
                normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

                transformList = []
                #transformList.append(T.Resize(256)) -- forward pass during inference uses tencrop 
                transformList.append(T.Resize(226))
                transformList.append(T.CenterCrop(226))
                transformList.append(ToTensorIfNotTensor())
                transformList.append(normalize)
                self.transforms=T.Compose(transformList)
            else: 
                self.transforms = transforms
            self.model = model

        def forward(self, x): 
            x_in = self.transforms(x)
            return self.model(x_in)
        
        def lazy_foward(self, x): 
            # accepts tensor, 0-1, bchw 
            self.model.eval()
            self.model.to("cuda")
            
            with torch.no_grad():
                x_in = self.transforms(x)
                if x_in.dim() == 3: 
                    x_in = x_in.unsqueeze(dim=0)
                
                varInput = x_in.cuda()

                features = self.model.densenet121.features(varInput)
                out = F.relu(features, inplace=True)
                out = F.adaptive_avg_pool2d(out, (1, 1))
                hidden_features = torch.flatten(out, 1)
                out = self.model.densenet121.classifier(hidden_features)
                #outMean = out.view(bs, ).mean(1)
            return out.data, hidden_features.data

    return Classifier(model)


def get_privacy_model(path="/vol/ideadata/ed52egek/pycharm/trichotomy/privacy/archive/Siamese_ResNet50_allcxr/Siamese_ResNet50_allcxr_checkpoint.pth"): 
    import torch
    import sys
    from pathlib import Path

    # Add the directory containing edm2/generate.py to the Python path
    script_dir = Path("/vol/ideadata/ed52egek/pycharm/trichotomy/privacy").resolve()  # Replace with the actual path
    sys.path.append(str(script_dir))

    from networks.SiameseNetwork import SiameseNetwork

    net = SiameseNetwork()
    net.load_state_dict(torch.load(path)["state_dict"])

    return net


def get_image_generation_model(path_net, path_gnet, model_weights, gmodel_weights, name): 
    import dnnlib

    encoder_batch_size = 4
    max_batch_size = 32
        # Rank 0 goes first.
    net = path_net
    gnet = path_gnet

    # Load main network.
    if isinstance(net, str):
        print(f'Loading network from {net} ...')
        with dnnlib.util.open_url(net, verbose=True) as f:
            data = pickle.load(f)
        net = data['ema'].to("cuda")
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
        gnet = data['ema'].to("cuda")
        gnet.load_state_dict(torch.load(gmodel_weights)["net"])

    assert gnet is not None

    # Initialize encoder.
    assert encoder is not None
    print(f'Setting up {type(encoder).__name__}...')
    encoder.init("cuda")
    if encoder_batch_size is not None and hasattr(encoder, 'batch_size'):
        encoder.batch_size = encoder_batch_size

    return net, gnet, encoder


def get_ds_and_indices(filelist="", basedir="", cond_mode="", class_idx=None, N=100, n_per_index=1, **kwargs): 
    # given a class index, basedir and potential moultiple n
    if class_idx is None: 
        raise ValueError("No longer supported")

    train_ds = LatentDataset(filelist_txt=filelist, basedir=basedir, cond_mode=cond_mode, load_to_memory=False)

    #if n_per_index != 1 and cond_mode=="pseudo_cond": 

    #    print("Generating multiple images with the same class label using n_per_index is the same as just generating more images for the same class")

    #    indices = torch.cat([torch.tensor([n,]*n_per_index) for n in range(N)]) 
    #else: 
    indices = []
    i = 0
    last_subject_id = -1
    while len(indices) * n_per_index < N:
        if train_ds.label_list[i] == class_idx: 
            subject_id = int(train_ds.file_list[i].split("/")[-1].split("_")[0])
            if subject_id == last_subject_id: 
                i+=1
                continue
            else: 
                last_subject_id = subject_id
                indices.extend([i,]*n_per_index)
        i+=1
    #indices = torch.cat(indices)

    return train_ds, indices 


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
                        Image.fromarray(image, 'RGB').save(image_pth)

            # Yield results
            yield r


def main(filelist, mode, outdir, model_weights, gmodel_weights, net, gnet, cond_mode, guidance, basedir):
    default_kwargs = {
        "EDM-2-AG":{
            "autoguidance":True,
            "guidance":1.4,
            "model_kwargs":{
                "model_weights":"/vol/ideadata/ed52egek/pycharm/trichotomy/importantmodels/cxr8_diffusionmodels/baseline-runs/cxr8_cond/training-state-0050331.pt",
                "gmodel_weights":"/vol/ideadata/ed52egek/pycharm/trichotomy/importantmodels/cxr8_diffusionmodels/baseline-runs/cxr8_cond/training-state-0008388.pt",
                "path_net":"/vol/ideadata/ed52egek/pycharm/trichotomy/importantmodels/cxr8_diffusionmodels/baseline-runs/cxr8_cond/network-snapshot-0050331-0.100.pkl",
                "path_gnet":"/vol/ideadata/ed52egek/pycharm/trichotomy/importantmodels/cxr8_diffusionmodels/baseline-runs/cxr8_cond/network-snapshot-0008388-0.050.pkl",
            },
            "ds_kwargs":{
                "cond_mode":"cond", # pseudocond, cond
                "basedir":"/vol/idea_ramses/ed52egek/data/trichotomy",
                "basedir_images":"/vol/ideadata/ed52egek/data/chestxray14"
            }
        },
        "EDM-2":{
            "autoguidance":False,
            "guidance":1.4,
            "model_kwargs":{
                "model_weights":"/vol/ideadata/ed52egek/pycharm/trichotomy/importantmodels/cxr8_diffusionmodels/baseline-runs/cxr8_cond/training-state-0050331.pt",
                "gmodel_weights":"/vol/ideadata/ed52egek/pycharm/trichotomy/importantmodels/cxr8_diffusionmodels/baseline-runs/cxr8_uncond/training-state-0008388.pt",
                "path_net":"/vol/ideadata/ed52egek/pycharm/trichotomy/importantmodels/cxr8_diffusionmodels/baseline-runs/cxr8_cond/network-snapshot-0050331-0.100.pkl",
                "path_gnet":"/vol/ideadata/ed52egek/pycharm/trichotomy/importantmodels/cxr8_diffusionmodels/baseline-runs/cxr8_uncond/network-snapshot-0008388-0.050.pkl",
            },
            "ds_kwargs":{
                "cond_mode":"cond", # pseudocond, cond
                "basedir":"/vol/idea_ramses/ed52egek/data/trichotomy",
                "basedir_images":"/vol/ideadata/ed52egek/data/chestxray14"
            }
        },
        "DiADM":{
            "autoguidance":False,
            "guidance":1.4,
            "model_kwargs":{
                "model_weights":"/vol/ideadata/ed52egek/pycharm/trichotomy/importantmodels/cxr8_diffusionmodels/baseline-runs/cxr8_pseudocond/training-state-0050331.pt",
                "gmodel_weights":"/vol/ideadata/ed52egek/pycharm/trichotomy/importantmodels/cxr8_diffusionmodels/baseline-runs/cxr8_uncond/training-state-0008388.pt",
                "path_net":"/vol/ideadata/ed52egek/pycharm/trichotomy/importantmodels/cxr8_diffusionmodels/baseline-runs/cxr8_pseudocond/network-snapshot-0050331-0.100.pkl",
                "path_gnet":"/vol/ideadata/ed52egek/pycharm/trichotomy/importantmodels/cxr8_diffusionmodels/baseline-runs/cxr8_uncond/network-snapshot-0008388-0.050.pkl",
            },
            "ds_kwargs":{
                "cond_mode":"pseudocond", # pseudocond, cond
                "basedir":"/vol/idea_ramses/ed52egek/data/trichotomy",
                "basedir_images":"/vol/ideadata/ed52egek/data/chestxray14"
            }
        },
        "DiADM-AG":{
            "autoguidance":True,
            "guidance":1.4,
            "model_kwargs":{
                "model_weights":"/vol/ideadata/ed52egek/pycharm/trichotomy/importantmodels/cxr8_diffusionmodels/baseline-runs/cxr8_pseudocond/training-state-0050331.pt",
                "gmodel_weights":"/vol/ideadata/ed52egek/pycharm/trichotomy/importantmodels/cxr8_diffusionmodels/baseline-runs/cxr8_pseudocond/training-state-0008388.pt",
                "path_net":"/vol/ideadata/ed52egek/pycharm/trichotomy/importantmodels/cxr8_diffusionmodels/baseline-runs/cxr8_pseudocond/network-snapshot-0050331-0.100.pkl",
                "path_gnet":"/vol/ideadata/ed52egek/pycharm/trichotomy/importantmodels/cxr8_diffusionmodels/baseline-runs/cxr8_pseudocond/network-snapshot-0008388-0.050.pkl",
            },
            "ds_kwargs":{
                "cond_mode":"pseudocond", # pseudocond, cond
                "basedir":"/vol/idea_ramses/ed52egek/data/trichotomy",
                "basedir_images":"/vol/ideadata/ed52egek/data/chestxray14"
            }
        }
    }

    print(f"Generating images for {mode}")

    default_kwargs[mode]["guidance"] = guidance
    model_kwargs = default_kwargs[mode]["model_kwargs"]
    ds_kwargs = default_kwargs[mode]["ds_kwargs"]
    
    model_kwargs["name"] = mode
    model_kwargs["model_weights"] = model_weights
    model_kwargs["gmodel_weights"] = gmodel_weights
    ds_kwargs["cond_mode"] = cond_mode
    ds_kwargs["basedir"] = basedir
    
    model_kwargs["path_net"] = net 
    model_kwargs["path_gnet"] = gnet 
    

    print("="*80)
    print("Model kwargs:")
    pprint(model_kwargs)
    print("Dataset kwargs:")
    pprint(ds_kwargs)


    train_ds = LatentDataset(filelist_txt=filelist, load_to_memory=False, basedir=ds_kwargs["basedir"], cond_mode=ds_kwargs["cond_mode"])
    print("="*80)
    net, gnet, encoder =  get_image_generation_model(**model_kwargs)
    print(f"Saving images to {outdir}")

    # indices are the indices of the dataset with certain calss 
    sampler_kwargs = {"autoguidance":default_kwargs[mode]["autoguidance"], 
                    "guidance":default_kwargs[mode]["guidance"], }

    print("Sampler kwargs")
    pprint(sampler_kwargs)


    image_iter = ImageIterable(train_ds=train_ds, indices=np.arange(len(train_ds)), device=torch.device("cuda"), net=net, sampler_fn=edm_sampler, gnet=gnet, encoder=encoder,outdir=outdir, add_seed_to_path=False, sampler_kwargs=sampler_kwargs)

    for r in tqdm(image_iter, unit='batch', total=len(image_iter), desc=f"Generating images"):
        for i in range(len(r.images)): 
            break



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images from a latent dataset.")
    
    parser.add_argument("--filelist", type=str, required=True, help="Path to the filelist.")
    parser.add_argument("--mode", type=str, required=True, choices=["EDM-2", "EDM-2-AG", "DiADM", "DiADM-AG"], help="Mode for image generation.")
    parser.add_argument("--outdir", type=str, default="./snth_train_images/", help="Output directory for generated images.")
    parser.add_argument("--model_weights", type=str, required=True, help="Path to model weights.")
    parser.add_argument("--gmodel_weights", type=str, required=True, help="Path to generative model weights.")
    parser.add_argument("--net", type=str, required=True, help="Network configuration.")
    parser.add_argument("--gnet", type=str, required=True, help="Generative network configuration.")
    parser.add_argument("--cond_mode", type=str, required=True, help="Conditioning mode.")
    parser.add_argument("--guidance", type=float, required=True, help="Guidance parameter.")
    parser.add_argument("--basedir", type=str, required=True, help="Base directory for Feature Dataset.")
    
    args = parser.parse_args()
    
    main(
        args.filelist, args.mode, args.outdir, 
        args.model_weights, args.gmodel_weights, 
        args.net, args.gnet, args.cond_mode, 
        args.guidance, args.basedir
    )