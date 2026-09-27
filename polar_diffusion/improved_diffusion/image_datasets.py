import os
import os.path as op

import blobfile as bf
import numpy as np
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def load_data(
    *, data_dir, batch_size, image_size, class_cond=False, deterministic=False, num_workers=4
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory, or a .npy file of uint8 images [N, H, W, 3]
                     (class labels, if any, in labels.npy next to it)
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    if op.isdir(data_dir):
        all_files = _list_image_files_recursively(data_dir)
    else:
        all_files = data_dir
    classes = None
    if class_cond:
        if op.isdir(data_dir):
            # Assume classes are the first part of the filename,
            # before an underscore.
            class_names = [bf.basename(path).split("_")[0] for path in all_files]
        else:
            class_names = np.load(op.join(op.dirname(data_dir), 'labels.npy')).tolist()
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]

    if dist.is_initialized():
        rank, world_size = dist.get_rank(), dist.get_world_size()
    else:
        rank, world_size = 0, 1

    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=rank,
        num_shards=world_size,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(self, resolution, image_paths, classes=None, shard=0, num_shards=1):
        super().__init__()
        self.resolution = resolution
        if isinstance(image_paths, list):
            self.local_images = image_paths[shard:][::num_shards]
            self.use_numpy = False
        else:
            assert image_paths.endswith(".npy")
            self.images = np.load(image_paths, mmap_mode="r")
            self.local_indices = np.arange(len(self.images))[shard:][::num_shards]
            self.use_numpy = True
        self.local_classes = None if classes is None else classes[shard:][::num_shards]


    def __len__(self):
        if self.use_numpy:
            return len(self.local_indices)
        else:
            return len(self.local_images)

    def _image_process_pipeline(self, idx):
        # Load and reshape image and return [H,W,3] unit8 np array
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()

        # We are not on a new enough PIL to support the `reducing_gap`
        # argument, which uses BOX downsampling at powers of two first.
        # Thus, we do it by hand to improve downsample quality.
        while min(*pil_image.size) >= 2 * self.resolution:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        scale = self.resolution / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )

        arr = np.array(pil_image.convert("RGB"))
        crop_y = (arr.shape[0] - self.resolution) // 2
        crop_x = (arr.shape[1] - self.resolution) // 2
        arr = arr[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]
        return arr


    def __getitem__(self, idx):
        if self.use_numpy:
            arr = np.array(self.images[self.local_indices[idx]])
        else:
            arr = self._image_process_pipeline(idx)
        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict



def load_data_web(
    *, data_dir, batch_size, image_size, class_cond=False, deterministic=False, cache_dir="./_cache",
    num_workers=4,
):
    """
    Like load_data(), but streams WebDataset .tar shards of 64x64 PNGs, e.g.
    data_dir="/data/imagenet64-{0001..1282}.tar". batch_size is per rank.
    """
    import webdataset as wds
    from torchvision import transforms

    os.makedirs(cache_dir, exist_ok=True)
    assert class_cond is False
    assert deterministic is False

    # This is the basic WebDataset definition: it starts with a URL and add shuffling,
    # decoding, and augmentation. Note `resampled=True`; this is essential for
    # distributed training to work correctly.
    transform = transforms.Compose(
        [
                transforms.ToTensor(),
        ]
    )
    def make_sample(sample):
        return ( 2. * transform(sample["png"]) - 1.,)

    trainset = wds.WebDataset(
        data_dir,
        resampled=True,
        shardshuffle=True,
        cache_dir=cache_dir,
        nodesplitter=wds.split_by_node,
    )
    trainset = trainset.shuffle(1000).decode("pil").map(make_sample)

    # For IterableDataset objects, the batching needs to happen in the dataset.
    trainset = trainset.batched(batch_size)

    trainloader = wds.WebLoader(trainset, batch_size=None, num_workers=num_workers)

    # We unbatch, shuffle, and rebatch to mix samples from different workers.
    trainloader = trainloader.unbatched().shuffle(1000).batched(batch_size)

    # A resampled dataset is infinite size, but we can recreate a fixed epoch length.
    trainloader = trainloader.with_epoch(1282 * 1000 // batch_size)

    while True:
        for data in trainloader:
            yield data[0], {}