#!/bin/sh
# Del 2: U-Net på PH2 med punkt-klik, ablation over antal klik og samplingstrategi.
# Kræver at `python main.py clicks` er kørt (filerne ligger i clicks/ i git).
# Submittes fra repo-roden: bsub < jobs/part2.sh

#BSUB -q c02516
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J part2
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 8:00
#BSUB -o part2_%J.out
#BSUB -e part2_%J.err

cd src

for strategy in random interior boundary; do
    for n in 1 3 5 10 20; do
        uv run python main.py train --dataset ph2 --model unet --loss point \
            --n-clicks $n --click-strategy $strategy
    done
done

uv run python main.py summarize
