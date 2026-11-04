"""Stier og standardværdier, der deles af hele projektet."""
from pathlib import Path

DATA_ROOT = Path("/dtu/datasets1/02516")
PH2_ROOT = DATA_ROOT / "PH2_Dataset_images"
DRIVE_ROOT = DATA_ROOT / "DRIVE"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = PROJECT_ROOT / "splits"    # committes, så hele gruppen bruger samme split
CLICKS_DIR = PROJECT_ROOT / "clicks"    # committes, så alle kørsler i del 2 bruger samme klik
RUNS_DIR = PROJECT_ROOT / "runs"        # én mappe per træningskørsel (ikke i git)
RESULTS_DIR = PROJECT_ROOT / "results"  # samlede tabeller og figurer til rapporten

DATASETS = ("ph2", "drive")

# Splits laves kun ud fra billeder med target. DRIVE's officielle testsæt har ingen
# karmasker, så de 20 annoterede træningsbilleder deles i train/val/test.
SPLIT_SEED = 42
SPLIT_SIZES = {"ph2": (140, 30, 30), "drive": (14, 3, 3)}

# Træningsstandarder per datasæt (kan overskrives fra kommandolinjen).
# DRIVE: 512 bevarer de tynde kar bedre end fx 128. Der trænes på tilfældige 256-crops,
# og hvert billede gentages `repeat` gange per epoch, fordi der kun er 14 træningsbilleder.
DATASET_DEFAULTS = {
    "ph2": {"image_size": 256, "crop_size": 0, "repeat": 1, "batch_size": 8, "epochs": 60},
    "drive": {"image_size": 512, "crop_size": 256, "repeat": 10, "batch_size": 4, "epochs": 60},
}

# Del 2: antal klik der gemmes per klasse per billede. Kørsler med færre klik bruger de
# første n, så klik-sættene er indlejrede (5 klik er en delmængde af 10 klik osv.).
MAX_CLICKS = 20
CLICK_MARGIN = 20  # pixels i original opløsning (afstand til læsionens kant)
CLICK_STRATEGIES = ("random", "interior", "boundary")
