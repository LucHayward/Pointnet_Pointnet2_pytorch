#!/bin/sh

#SBATCH --account=gpumk
#SBATCH --partition=gpumk
#SBATCH --nodes=1 --ntasks=8 --gres=gpu:pascal:1
#SBATCH --time=24:00:00
#SBATCH --job-name="RF_Piazza_2.5%"
#SBATCH --mail-user=hywluc001@myuct.ac.za
#SBATCH --mail-type=ALL
#SBATCH -e slurm-RF_Piazza_2.5%.err
#SBATCH -o slurm-RF_Piazza_2.5%.out

module load python/miniconda3-py39
source activate /scratch/hywluc001/conda-envs/pointnet

cd Pointnet_Pointnet2_pytorch
python3 train_rf.py --data_path data/PatrickData/Piazza/2.5% --log_dir Piazza_2.5% --xgboost

