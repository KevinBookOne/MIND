#!/bin/bash
cd "$(dirname "$0")"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate deepreg
python MIND.py
conda deactivate
python FvsW.py
python DI.py
