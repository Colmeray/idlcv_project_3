#!/bin/sh
# Del 1: EncDec og U-Net (BCE, focal, vægtet BCE) på PH2 og DRIVE.
# Submittes fra repo-roden: bsub < jobs/part1.sh

#BSUB -q c02516
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J part1
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 8:00
#BSUB -o part1_%J.out
#BSUB -e part1_%J.err

cd src

for dataset in ph2 drive; do
    uv run python main.py train --dataset $dataset --model encdec --loss bce
    for loss in bce focal weighted_bce; do
        uv run python main.py train --dataset $dataset --model unet --loss $loss
    done
done

uv run python main.py summarize
