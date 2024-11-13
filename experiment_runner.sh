#!/bin/sh -x

#SBATCH --job-name=nda_snn
#SBATCH --account=eelsaisdc
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH --hint=multithread
#SBATCH --output=./logs/%j.out
#SBATCH --error=./logs/%j.err

# Load the required modules and activate the virtual environment
source ~/.bash_profile
load_v1
activate_evm

python3 main.py --dset dc10 --amp --nda --fold_idx $1
