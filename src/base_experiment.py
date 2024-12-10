import ml_collections
import os
import math

config = ml_collections.ConfigDict()
config.seed = 42
config.base_dir = "/vol/ideadata/ed52egek/data/" if os.path.abspath(".").startswith("/vol") else "/home/atuin/b180dc/b180dc10/data/"

config.data = data = ml_collections.ConfigDict()
data.mode = "image" # video

# compute latent
config.compute_latent = compute_latent =  ml_collections.ConfigDict()
config.compute_latent.batch_size = 16
config.compute_latent.input_size = 512
config.compute_latent.model_path = "stabilityai/stable-diffusion-2"

# compute features
config.compute_feature = compute_feature =  ml_collections.ConfigDict()
config.compute_feature.batch_size = 512
