"""Evaluering af en færdig kørsel og opsamling af resultater på tværs af kørsler."""
import json

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

import config
from datasets.image_dataset import SegmentationDataset
from models import build_model
from training.trainer import evaluate
from utils.metrics import METRICS
from utils.visualization import save_examples


def save_mask(array, path):
    # array: 2D numpy array med 0 og 1
    im_arr = (array * 255)
    Image.fromarray(np.uint8(im_arr)).save(path)


def load_run(run_dir, device, checkpoint="best.pt"):
    cfg = json.loads((run_dir / "config.json").read_text())
    model = build_model(cfg["model"], cfg["base_channels"], cfg["depth"])
    model.load_state_dict(torch.load(run_dir / checkpoint, map_location=device))
    return cfg, model.to(device)


def evaluate_run(run_dir, save_masks=True, n_examples=4):
    """Beregner de fem metrikker på train/val/test med best.pt og gemmer masker, figur og metrics.json.

    Evalueringen sker mod de fulde masker (også for klik-kørsler) i den resizede opløsning.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, model = load_run(run_dir, device)
    results = {}
    for split in ("train", "val", "test"):
        dataset = SegmentationDataset(cfg["dataset"], split, cfg["image_size"])
        loader = DataLoader(dataset, batch_size=cfg["eval_batch_size"])
        metrics, tracker, preds = evaluate(model, loader, device, use_fov=not cfg["no_fov"], return_preds=True)
        results[split] = metrics
        per_image = pd.DataFrame({k: v.numpy() for k, v in tracker.per_image().items()}, index=dataset.ids)
        per_image.to_csv(run_dir / f"per_image_{split}.csv", index_label="id")
        if save_masks:
            mask_dir = run_dir / "predictions" / split
            mask_dir.mkdir(parents=True, exist_ok=True)
            for sample_id, pred in zip(dataset.ids, preds):
                save_mask(pred.numpy(), mask_dir / f"{sample_id}.png")
        if split == "test":
            save_examples(dataset, preds, run_dir / "test_examples.png", n_examples)
    (run_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    print(pd.DataFrame(results).T[list(METRICS)].round(4).to_string(), flush=True)
    return results


def summarize_runs(runs_dir=config.RUNS_DIR, out_path=config.RESULTS_DIR / "results.csv"):
    """Samler config + metrics.json fra alle kørsler i én tabel (én række per kørsel og split)."""
    rows = []
    for metrics_path in sorted(runs_dir.glob("*/metrics.json")):
        cfg = json.loads((metrics_path.parent / "config.json").read_text())
        for split, metrics in json.loads(metrics_path.read_text()).items():
            rows.append({"run": metrics_path.parent.name, "dataset": cfg["dataset"], "model": cfg["model"],
                         "loss": cfg["loss"], "n_clicks": cfg["n_clicks"] or None,
                         "click_strategy": cfg["click_strategy"] if cfg["n_clicks"] else None,
                         "seed": cfg["seed"], "split": split, **metrics})
    table = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)
    print(f"{len(table)} rækker gemt i {out_path}")
    if len(table):
        print(table[table.split == "test"].drop(columns="split").round(4).to_string(index=False))
    return table
