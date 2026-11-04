# idlcv_project_3

Projekt 3 i DTU 02516: segmentering af PH2 (hudlæsioner) og DRIVE (blodkar i nethinden),
med fuld supervision (del 1) og med punkt-klik (del 2).

## Opsætning

```sh
uv sync          # installerer afhængigheder i .venv
cd src           # alle kommandoer køres herfra
```

`splits/` og `clicks/` ligger i git, så hele gruppen bruger præcis samme split og klik.
Kør kun `splits`/`clicks` igen med `--overwrite`, hvis I bevidst vil lave dem om.

## Kommandoer

```sh
uv run python main.py stats                      # tal til datasætbeskrivelsen -> results/dataset_stats.json
uv run python main.py shapes --model unet --image-size 256   # U-Net-tabel (lag, shape, parametre)
uv run python main.py train --dataset ph2 --model unet --loss focal
uv run python main.py train --dataset ph2 --loss point --n-clicks 5 --click-strategy interior
uv run python main.py evaluate ../runs/<kørsel>  # evaluér en kørsel igen
uv run python main.py summarize                  # alle kørsler -> results/results.csv (+ Dice vs. klik)
```

Se `uv run python main.py train --help` for alle argumenter. Standardværdier per datasæt
(billedstørrelse, crops, batch size, epochs) står i `src/config.py`.

Hele del 1 og del 2 køres som HPC-jobs fra repo-roden:

```sh
bsub < jobs/part1.sh
bsub < jobs/part2.sh
```

Kør kun ét tungt job ad gangen i gruppen.

## Hvad en kørsel gemmer (`runs/<navn>/`)

| Fil | Indhold |
|---|---|
| `config.json` | alle argumenter (+ pos_weight for vægtet BCE) |
| `history.csv` | train-loss og val-metrikker per epoch |
| `best.pt`, `last.pt` | vægte for bedste val-Dice og sidste epoch |
| `metrics.json` | de fem metrikker på train/val/test (med `best.pt`) |
| `per_image_<split>.csv` | metrikker per billede |
| `predictions/<split>/<id>.png` | forudsagte masker |
| `test_examples.png` | billede, ground truth, forudsigelse og fejlkort |

## Valg der skal nævnes i rapporten

- **Splits:** DRIVE's officielle testsæt har ingen karmasker, så de 20 annoterede billeder er
  delt 14/3/3. PH2 er delt 140/30/30. Seed 42.
- **Opløsning:** PH2 resizes til 256×256, DRIVE til 512×512 (træning på tilfældige 256-crops,
  hvert billede 10 gange per epoch). Metrikkerne beregnes i den resizede opløsning.
- **FOV:** I DRIVE beregnes metrikkerne kun inden for FOV-masken (`--no-fov` slår det fra).
  Loss beregnes på hele billedet.
- **Augmentation:** crops, flips, 90-graders rotationer og color jitter. Der er ingen
  interpolation, så masker og enkelte klik-pixels bevares præcist.
- **Model:** logits ud, threshold `logits > 0` (= sigmoid > 0.5). Bedste epoch vælges på
  val-Dice. Adam, lr 1e-3.
- **Focal loss:** standardversionen, hvor begge led vægtes med (1 − p_t)^γ, γ = 2.
- **Vægtet BCE:** pos_weight = #baggrund / #forgrund i træningssættet.
- **Klik (del 2):** 20 klik per klasse per billede gemmes per strategi (`random`, `interior`
  ≥ 20 px fra kanten, `boundary` ≤ 20 px fra kanten). En kørsel med n klik bruger de første n,
  så klik-sættene er indlejrede. Punkt-loss er BCE kun på klikkede pixels, balanceret mellem
  positive og negative klik. Evaluering sker mod de fulde masker.
