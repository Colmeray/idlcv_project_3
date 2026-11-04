"""Kommandolinje for hele projektet. Køres fra src/: `uv run python main.py <kommando> ...`

  splits     lav og gem train/val/test-splits (én gang, committes)
  clicks     simulér og gem klik til del 2 (én gang, committes) + eksempelfigur
  stats      tal til datasætbeskrivelsen
  shapes     lag-for-lag shapes for en model (U-Net-tabellen i rapporten)
  train      træn én konfiguration og evaluér den bagefter
  evaluate   evaluér en eksisterende kørsel igen
  summarize  saml alle kørsler i results/results.csv (+ Dice vs. klik-figur)
"""
import argparse
import json
from pathlib import Path

import config


def cmd_splits(args):
    from datasets.image_dataset import make_split

    for dataset in config.DATASETS:
        split = make_split(dataset, overwrite=args.overwrite)
        print(f"{dataset}: " + ", ".join(f"{k} {len(split[k])}" for k in ("train", "val", "test")))


def cmd_clicks(args):
    from datasets.clicks import load_clicks, make_clicks
    from datasets.image_dataset import load_raw, load_split
    from utils.visualization import save_click_examples

    for strategy in config.CLICK_STRATEGIES:
        make_clicks(strategy, margin=args.margin, overwrite=args.overwrite)
        print(f"Klik gemt for strategi '{strategy}'")
    sample_ids = load_split("ph2")["train"][:3]
    raw = {sample_id: load_raw("ph2", sample_id) for sample_id in sample_ids}
    path = config.RESULTS_DIR / "click_strategies.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    save_click_examples({k: v[0] for k, v in raw.items()}, {k: v[1] for k, v in raw.items()},
                        {s: load_clicks(s) for s in config.CLICK_STRATEGIES}, 10, path)
    print(f"Eksempelfigur gemt i {path}")


def cmd_stats(args):
    from datasets.image_dataset import dataset_stats

    stats = {dataset: dataset_stats(dataset) for dataset in config.DATASETS}
    path = config.RESULTS_DIR / "dataset_stats.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


def cmd_shapes(args):
    from models import build_model, shape_table

    model = build_model(args.model, args.base_channels, args.depth)
    print(f"{'lag':<14}{'type':<18}{'output (C x H x W)':<24}{'parametre':>10}")
    for name, kind, shape, n_params in shape_table(model, args.image_size):
        print(f"{name:<14}{kind:<18}{' x '.join(map(str, shape)):<24}{n_params:>10,}")
    print(f"I alt {sum(p.numel() for p in model.parameters()):,} parametre")


def cmd_train(args):
    from train import run_training

    for key, value in config.DATASET_DEFAULTS[args.dataset].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    run_training(args)


def cmd_evaluate(args):
    from evaluate import evaluate_run

    evaluate_run(Path(args.run_dir), save_masks=not args.no_masks)


def cmd_summarize(args):
    from evaluate import summarize_runs
    from utils.visualization import plot_dice_vs_clicks

    table = summarize_runs()
    if len(table) and table.n_clicks.notna().any():
        path = config.RESULTS_DIR / "dice_vs_clicks.png"
        plot_dice_vs_clicks(table, path)
        print(f"Figur gemt i {path}")


def add_model_args(parser):
    parser.add_argument("--model", choices=("encdec", "unet"), default="unet")
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--depth", type=int, default=4, help="antal downsampling-niveauer")


def main():
    from training.losses import LOSSES

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("splits")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_splits)

    p = sub.add_parser("clicks")
    p.add_argument("--margin", type=int, default=config.CLICK_MARGIN)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_clicks)

    sub.add_parser("stats").set_defaults(func=cmd_stats)

    p = sub.add_parser("shapes")
    add_model_args(p)
    p.add_argument("--image-size", type=int, default=256)
    p.set_defaults(func=cmd_shapes)

    p = sub.add_parser("train", help="standardværdier per datasæt står i config.DATASET_DEFAULTS")
    p.add_argument("--dataset", choices=config.DATASETS, required=True)
    add_model_args(p)
    p.add_argument("--loss", choices=LOSSES, default="bce")
    p.add_argument("--image-size", type=int)
    p.add_argument("--crop-size", type=int, help="tilfældige crops under træning (0 = ingen)")
    p.add_argument("--repeat", type=int, help="gentag hvert træningsbillede n gange per epoch")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--eval-batch-size", type=int, default=4)
    p.add_argument("--epochs", type=int)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--no-fov", action="store_true", help="beregn metrikker på hele billedet, ikke kun i FOV")
    p.add_argument("--n-clicks", type=int, default=0, help="del 2: klik per klasse per billede (0 = fulde masker)")
    p.add_argument("--click-strategy", choices=config.CLICK_STRATEGIES, default="random")
    p.add_argument("--name", help="navn på kørslen (standard: genereres ud fra argumenterne)")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate")
    p.add_argument("run_dir")
    p.add_argument("--no-masks", action="store_true")
    p.set_defaults(func=cmd_evaluate)

    sub.add_parser("summarize").set_defaults(func=cmd_summarize)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
