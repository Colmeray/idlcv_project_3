"""Indlæsning, splits og PyTorch-dataset for PH2 og DRIVE."""
import json
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

import config
from datasets.transforms import PairedAugment, resize_image, resize_mask


def list_ids(dataset):
    """Id'er på alle billeder med target (DRIVE's officielle testsæt har intet facit)."""
    if dataset == "ph2":
        return sorted(p.name for p in config.PH2_ROOT.iterdir() if p.is_dir())
    if dataset == "drive":
        images = (config.DRIVE_ROOT / "training" / "images").glob("*_training.tif")
        return sorted(p.name.split("_")[0] for p in images)
    raise ValueError(f"Ukendt datasæt: {dataset}")


def load_raw(dataset, sample_id):
    """Returnerer (PIL RGB-billede, target som bool-array, FOV som bool-array) i original opløsning.

    PH2 har ingen FOV-maske, så hele billedet regnes som synsfelt.
    """
    if dataset == "ph2":
        folder = config.PH2_ROOT / sample_id
        image = Image.open(folder / f"{sample_id}_Dermoscopic_Image" / f"{sample_id}.bmp").convert("RGB")
        lesion = Image.open(folder / f"{sample_id}_lesion" / f"{sample_id}_lesion.bmp")
        target = np.array(lesion.convert("L")) > 127
        fov = np.ones_like(target)
    else:
        folder = config.DRIVE_ROOT / "training"
        image = Image.open(folder / "images" / f"{sample_id}_training.tif").convert("RGB")
        target = np.array(Image.open(folder / "1st_manual" / f"{sample_id}_manual1.gif").convert("L")) > 127
        fov = np.array(Image.open(folder / "mask" / f"{sample_id}_training_mask.gif").convert("L")) > 127
    return image, target, fov


def split_path(dataset):
    return config.SPLITS_DIR / f"{dataset}.json"


def make_split(dataset, seed=config.SPLIT_SEED, overwrite=False):
    path = split_path(dataset)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} findes allerede (brug --overwrite for at lave det om)")
    ids = list_ids(dataset)
    n_train, n_val, n_test = config.SPLIT_SIZES[dataset]
    assert n_train + n_val + n_test == len(ids), f"{dataset}: split-størrelser passer ikke til {len(ids)} billeder"
    random.Random(seed).shuffle(ids)
    split = {
        "seed": seed,
        "train": sorted(ids[:n_train]),
        "val": sorted(ids[n_train:n_train + n_val]),
        "test": sorted(ids[n_train + n_val:]),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=2))
    return split


def load_split(dataset):
    path = split_path(dataset)
    if not path.exists():
        raise FileNotFoundError(f"{path} findes ikke. Kør `python main.py splits` først.")
    return json.loads(path.read_text())


def sparse_target(clicks, n_clicks, orig_shape, size):
    """Sparse target i size x size: 1 = positivt klik, 0 = negativt klik, -1 = ignore.

    Klik er gemt i original opløsning og skaleres til den resizede opløsning.
    """
    target = torch.full((size, size), -1.0)
    h, w = orig_shape
    for value, key in ((1.0, "positive"), (0.0, "negative")):
        for r, c in clicks[key][:n_clicks]:
            target[int(r * size / h), int(c * size / w)] = value
    return target


class SegmentationDataset(Dataset):
    """Returnerer (billede 3xHxW, target HxW, FOV HxW), resizet til image_size x image_size.

    Ved fuld supervision er target {0, 1}. Med `clicks` (træning i del 2) er target en sparse
    maske med 1/0 på de første `n_clicks` positive/negative klik og -1 alle andre steder.
    Alle billeder resizes og holdes i hukommelsen, så det kun sker én gang.
    """

    def __init__(self, dataset, split, image_size, augment=False, crop_size=None, repeat=1,
                 clicks=None, n_clicks=None):
        self.dataset = dataset
        self.ids = load_split(dataset)[split]
        self.augment = PairedAugment(crop_size) if augment else None
        self.repeat = repeat
        self.images, self.targets, self.fovs = [], [], []
        for sample_id in self.ids:
            image, target, fov = load_raw(dataset, sample_id)
            self.images.append(resize_image(image, image_size))
            if clicks is None:
                self.targets.append(resize_mask(target, image_size).float())
            else:
                self.targets.append(sparse_target(clicks[sample_id], n_clicks, target.shape, image_size))
            self.fovs.append(resize_mask(fov, image_size))

    def __len__(self):
        return len(self.ids) * self.repeat

    def __getitem__(self, idx):
        idx %= len(self.ids)
        image, target, fov = self.images[idx], self.targets[idx], self.fovs[idx]
        if self.augment is not None:
            image, target, fov = self.augment(image, target, fov)
        return image, target, fov


def dataset_stats(dataset):
    """Tal til datasætbeskrivelsen i rapporten: antal, opløsninger og andel forgrundspixels."""
    split = load_split(dataset)
    sizes, fractions, fractions_fov = {}, {}, {}
    for name in ("train", "val", "test"):
        for sample_id in split[name]:
            image, target, fov = load_raw(dataset, sample_id)
            sizes[f"{image.width}x{image.height}"] = sizes.get(f"{image.width}x{image.height}", 0) + 1
            fractions.setdefault(name, []).append(float(target.mean()))
            fractions_fov.setdefault(name, []).append(float(target[fov].mean()))
    all_fractions = sum(fractions.values(), [])
    stats = {
        "n_labelled": len(all_fractions),
        "split_sizes": {k: len(split[k]) for k in ("train", "val", "test")},
        "resolutions": dict(sorted(sizes.items(), key=lambda kv: -kv[1])),
        "foreground_fraction_mean": float(np.mean(all_fractions)),
        "foreground_fraction_std": float(np.std(all_fractions)),
        "foreground_fraction_per_split": {k: float(np.mean(v)) for k, v in fractions.items()},
    }
    if dataset == "drive":
        official_test = len(list((config.DRIVE_ROOT / "test" / "images").glob("*.tif")))
        stats["official_test_without_labels"] = official_test
        stats["foreground_fraction_in_fov_mean"] = float(np.mean(sum(fractions_fov.values(), [])))
    return stats
