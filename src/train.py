"""Træning af én konfiguration. Kaldes via `python main.py train ...` (se README)."""
import json
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from datasets.clicks import load_clicks
from datasets.image_dataset import SegmentationDataset
from evaluate import evaluate_run
from models import build_model
from training import POINT_LOSSES, build_loss, fit
from training.losses import bce_loss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_name(args):
    name = f"{args.dataset}_{args.model}_{args.loss}"
    if args.n_clicks:
        name += f"_{args.click_strategy}_n{args.n_clicks}"
    return f"{name}_s{args.seed}"


def pos_weight_from(dataset):
    """#baggrundspixels / #forgrundspixels i træningssættets targets."""
    targets = torch.stack(dataset.targets)
    return ((targets == 0).sum() / (targets == 1).sum()).item()


def run_training(args):
    weak = bool(args.n_clicks)
    if weak != (args.loss in POINT_LOSSES):
        raise ValueError("Punkt-losses kræver --n-clicks, og --n-clicks kræver en punkt-loss")
    if weak and args.dataset != "ph2":
        raise ValueError("Klik-supervision er kun lavet for PH2")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = config.RUNS_DIR / (args.name or run_name(args))
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run: {run_dir} | device: {device}", flush=True)

    clicks = load_clicks(args.click_strategy) if weak else None
    train_set = SegmentationDataset(args.dataset, "train", args.image_size, augment=not args.no_augment,
                                    crop_size=args.crop_size or None, repeat=args.repeat,
                                    clicks=clicks, n_clicks=args.n_clicks)
    val_set = SegmentationDataset(args.dataset, "val", args.image_size)
    pos_weight = pos_weight_from(train_set) if args.loss == "weighted_bce" else None

    run_config = {**vars(args), "pos_weight": pos_weight}
    run_config.pop("func", None)
    (run_dir / "config.json").write_text(json.dumps(run_config, indent=2))

    model = build_model(args.model, args.base_channels, args.depth).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.eval_batch_size, num_workers=args.num_workers)

    # Val-loss er altid almindelig BCE mod de fulde masker, så kurverne kan sammenlignes på tværs af kørsler.
    fit(model, train_loader, val_loader, build_loss(args.loss, pos_weight), bce_loss, optimizer,
        args.epochs, device, run_dir, use_fov=not args.no_fov)
    evaluate_run(run_dir)
