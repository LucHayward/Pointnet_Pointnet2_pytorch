#!/bin/sh

#SBATCH --account=gpumk
#SBATCH --partition=gpumk
#SBATCH --nodes=1 --ntasks=8 --gres=gpu:pascal:1
#SBATCH --time=12:00:00
#SBATCH --job-name="P++_Lunnahoja_2.5%-s3dis"
#SBATCH --mail-user=hywluc001@myuct.ac.za
#SBATCH --mail-type=ALL
#SBATCH -e slurm-P++_Lunnahoja_2.5%-s3dis.err
#SBATCH -o slurm-P++_Lunnahoja_2.5%-s3dis.out

module load python/miniconda3-py39
source activate /scratch/hywluc001/conda-envs/pointnet

cd Pointnet_Pointnet2_pytorch
python3 train_masters.py --data_path data/PatrickData/Lunnahoja/2.5% --epoch 60 --log_dir final/Lunnahoja_2.5%_s3dis --decay_rate 1e-2 --log_merged_validation --log_merged_training_set --sample_all_validation --shuffle_training_data
