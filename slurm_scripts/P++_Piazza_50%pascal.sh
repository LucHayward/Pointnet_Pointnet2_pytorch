#!/bin/sh

#SBATCH --account=gpumk
#SBATCH --partition=gpumk
#SBATCH --nodes=1 --ntasks=8 --gres=gpu:pascal:1
#SBATCH --time=24:00:00
#SBATCH --job-name="P++_Piazza_50%"
#SBATCH --mail-user=hywluc001@myuct.ac.za
#SBATCH --mail-type=ALL
#SBATCH -e slurm-P++_Piazza_50%.err
#SBATCH -o slurm-P++_Piazza_50%.out

module load python/miniconda3-py39
source activate /scratch/hywluc001/conda-envs/pointnet

cd Pointnet_Pointnet2_pytorch
python3 train_masters.py --data_path data/PatrickData/Piazza/50% --epoch 60 --log_dir final/Piazza_50% --decay_rate 1e-2 --log_merged_validation --log_merged_training_set --sample_all_validation --shuffle_training_data

