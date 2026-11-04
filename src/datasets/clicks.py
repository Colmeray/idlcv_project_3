"""Simulerede punkt-klik på PH2's træningsbilleder til svagt superviseret træning (del 2).

Strategier (positive klik i læsionen, negative i baggrunden):
- random:   uniformt over hele klassen.
- interior: mindst CLICK_MARGIN pixels fra læsionens kant, som en annotator der klikker
            "sikkert" inde i/uden for objektet.
- boundary: højst CLICK_MARGIN pixels fra kanten (sværere, men mere informativ supervision).

Klikkene genereres én gang med fast seed og gemmes i clicks/ph2_<strategi>.json i original
opløsning som [række, kolonne].
"""
import json

import numpy as np
from scipy.ndimage import distance_transform_edt

import config
from datasets.image_dataset import load_raw, load_split


def candidate_pixels(mask, strategy, margin):
    """Koordinater (N x 2) hvor der må klikkes for klassen `mask` (True = klassen)."""
    if strategy == "random":
        return np.argwhere(mask)
    # Afstand fra hver pixel i klassen til nærmeste pixel uden for den, dvs. til læsionens kant.
    dist = distance_transform_edt(mask)
    if strategy == "interior":
        allowed = dist >= margin
        if allowed.sum() < config.MAX_CLICKS:  # meget lille læsion: brug den inderste halvdel
            allowed = dist >= dist.max() / 2
    elif strategy == "boundary":
        allowed = mask & (dist <= margin)
    else:
        raise ValueError(f"Ukendt klik-strategi: {strategy}")
    return np.argwhere(allowed)


def sample_clicks(strategy, seed=config.SPLIT_SEED, margin=config.CLICK_MARGIN):
    rng = np.random.default_rng(seed)
    clicks = {}
    for sample_id in load_split("ph2")["train"]:
        _, target, _ = load_raw("ph2", sample_id)
        clicks[sample_id] = {}
        for key, mask in (("positive", target), ("negative", ~target)):
            candidates = candidate_pixels(mask, strategy, margin)
            idx = rng.choice(len(candidates), size=config.MAX_CLICKS,
                             replace=len(candidates) < config.MAX_CLICKS)
            clicks[sample_id][key] = candidates[idx].tolist()
    return clicks


def clicks_path(strategy):
    return config.CLICKS_DIR / f"ph2_{strategy}.json"


def make_clicks(strategy, seed=config.SPLIT_SEED, margin=config.CLICK_MARGIN, overwrite=False):
    path = clicks_path(strategy)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} findes allerede (brug --overwrite for at lave det om)")
    data = {"strategy": strategy, "seed": seed, "margin": margin,
            "clicks": sample_clicks(strategy, seed, margin)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return data


def load_clicks(strategy):
    path = clicks_path(strategy)
    if not path.exists():
        raise FileNotFoundError(f"{path} findes ikke. Kør `python main.py clicks` først.")
    return json.loads(path.read_text())["clicks"]
