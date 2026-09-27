"""
Compute the mean/std of the image radius r = ||x|| (or rho = log r) over a
dataset, for the --radius_stats flag of train.py / sample.py.

  python scripts/imagenet64/radius_stats.py --data /path/to/imagenet64.npy --out configs/imagenet64_radius.json

--data is a .npy of uint8 images [N, H, W, 3] or a folder of images.
"""
import argparse

import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from polar_diffusion.improved_diffusion.image_datasets import ImageDataset, _list_image_files_recursively
from polar_diffusion.polar import compute_radius_stats, save_radius_stats


class NpyImages(Dataset):
    def __init__(self, path):
        self.images = np.load(path, mmap_mode="r")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        arr = np.array(self.images[idx]).astype(np.float32) / 127.5 - 1.0
        return np.transpose(arr, [2, 0, 1])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--field", choices=["r", "rho"], default="r")
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    if args.data.endswith(".npy"):
        dataset = NpyImages(args.data)
    else:
        dataset = ImageDataset(args.image_size, _list_image_files_recursively(args.data))
    loader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)
    stats = compute_radius_stats(tqdm(loader), field=args.field)
    save_radius_stats(args.out, stats)
    print(f"{len(dataset)} images: {stats} -> {args.out}")


if __name__ == "__main__":
    main()
