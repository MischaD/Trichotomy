import numpy as np
import torch
import os
import torch
from torch.utils.data import Dataset, DataLoader
from utils import main_setup
import argparse
import tqdm


class LatentDataset(Dataset):
    """A dataset that uses a FileList.csv to load a bunch of tensors. Does not care whether the tensors are images or videos. If they are videos, randomly returns one index."""
    def __init__(self, filelist_txt, basedir):
        self.basedir = basedir
        self.file_list = []

        line = True
        with open(filelist_txt, "r") as fp:
            while line:
                line = fp.readline()
                if line:
                    image_path = line.split()[0]
                    self.file_list.append(image_path  + ".pt")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = self.file_list[idx]
        tensor = torch.load(os.path.join(self.basedir, path))
        return tensor, idx, path  # Return index to maintain order


def incremental_channelwise_mean_std(arrays):
    # Initialize total counts and running mean for channel-wise computations
    n_total = 0
    mean_total = None
    var_total = None

    # First pass to calculate the channel-wise mean
    for array in tqdm.tqdm(arrays):
        # array shape: (batch_size, channels, height, width)
        n = array[0].shape[0]  # Number of samples in the batch
        batch_mean = array[0].mean(dim=(0, 2, 3))  # Channel-wise mean

        if mean_total is None:
            mean_total = batch_mean
        else:
            # Update running mean
            mean_total = (n_total * mean_total + n * batch_mean) / (n_total + n)

        batch_var = ((array[0] - mean_total[None, :, None, None]) ** 2).mean(dim=(0, 2, 3))  # Channel-wise variance

        if var_total is None:
            var_total = n * batch_var
        else:
            var_total += n * batch_var

        n_total += n

    # Final variance and standard deviation calculation
    variance = var_total / n_total
    std_dev = torch.sqrt(variance)

    return mean_total, std_dev


def save_tensor(tensor, filelist, ending):
    """Save the channel-wise mean to a file."""
    # Replace "/" in the filelist name with "_"
    sanitized_name = filelist.replace("/", "_")
    output_file = f"{sanitized_name}_channel_{ending}.pt"
    torch.save(tensor, output_file)
    print(f"Channel-wise mean saved to: {output_file}")


def main(config):
    latent = LatentDataset(filelist_txt=config.filelist, basedir=config.basedir)
    dataloader = DataLoader(
        latent,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers
    )

    mean, std = incremental_channelwise_mean_std(dataloader)
    print(f"Channel-wise Mean: {mean}")
    print(f"Channel-wise Std: {std}")

    save_tensor(mean, config.filelist, "mean")
    save_tensor(std, config.filelist, "std")


def get_args():
    parser = argparse.ArgumentParser(description="Compute channel-wise mean and std for latent tensors.")
    parser.add_argument("--filelist", type=str, required=True, help="Path to the filelist CSV.")
    parser.add_argument("--basedir", type=str, required=True, help="Base directory for tensor files.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for DataLoader.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle data during DataLoader iteration.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader.")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)


