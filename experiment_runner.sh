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
#SBATCH --output=/p/project1/eelsaisdc/bhisikar1/projects/NDA_SNN/logs/%j.out
#SBATCH --error=/p/project1/eelsaisdc/bhisikar1/projects/NDA_SNN/logs/%j.err

# Load the required modules and activate the virtual environment
source ~/init_scripts/init_event_mamba.sh

cd /p/project1/eelsaisdc/bhisikar1/projects/NDA_SNN
python main.py --dset dc10 --amp --nda --fold_idx $1