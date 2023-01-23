#!/bin/bash

cd /home/luc/PycharmProjects/Pointnet_Pointnet2_pytorch/
for dataset in Church Lunnahoja Monument Bagni_Nerone Montelupo Piazza
do
  for split in "2.5%" "5%" "25%" "50%"
  do
#  mkdir -p log/final/${dataset}_${split}/checkpoints
#  mkdir -p log/final/${dataset}_${split}_s3dis/checkpoints
#  cp log/temp/checkpoints/best_model.pth log/final/${dataset}_${split}_s3dis/checkpoints/best_model.pth

  echo "#!/bin/sh

#SBATCH --account=gpumk
#SBATCH --partition=gpumk
#SBATCH --nodes=1 --ntasks=8 --gres=gpu:pascal:1
#SBATCH --time=24:00:00
#SBATCH --job-name=\"RF_${dataset}_${split}\"
#SBATCH --mail-user=hywluc001@myuct.ac.za
#SBATCH --mail-type=ALL
#SBATCH -e slurm-RF_${dataset}_${split}.err
#SBATCH -o slurm-RF_${dataset}_${split}.out

module load python/miniconda3-py39
source activate /scratch/hywluc001/conda-envs/pointnet

cd Pointnet_Pointnet2_pytorch
python3 train_rf.py \
--data_path "data/PatrickData/${dataset}/${split}" \
--log_dir ${dataset}_${split} \
--xgboost
" > slurm_scripts/RF_${dataset}_${split}.sh

  done
done

