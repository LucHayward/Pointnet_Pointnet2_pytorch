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
#SBATCH --time=12:00:00
#SBATCH --job-name=\"P++_${dataset}_${split}\"
#SBATCH --mail-user=hywluc001@myuct.ac.za
#SBATCH --mail-type=ALL
#SBATCH -e slurm-P++_${dataset}_${split}.err
#SBATCH -o slurm-P++_${dataset}_${split}.out

module load python/miniconda3-py39
source activate /scratch/hywluc001/conda-envs/pointnet

cd Pointnet_Pointnet2_pytorch
python3 train_masters.py \
--data_path "data/PatrickData/${dataset}/${split}" \
--epoch 60 \
--log_dir final/${dataset}_${split} \
--decay_rate 1e-2 \
--log_merged_validation \
--log_merged_training_set \
--sample_all_validation \
--shuffle_training_data
" > slurm_scripts/P++_${dataset}_${split}'pascal'.sh

echo "#!/bin/sh

#SBATCH --account=gpumk
#SBATCH --partition=gpumk
#SBATCH --nodes=1 --ntasks=8 --gres=gpu:pascal:1
#SBATCH --time=12:00:00
#SBATCH --job-name=\"P++_${dataset}_${split}-s3dis\"
#SBATCH --mail-user=hywluc001@myuct.ac.za
#SBATCH --mail-type=ALL
#SBATCH -e slurm-P++_${dataset}_${split}-s3dis.err
#SBATCH -o slurm-P++_${dataset}_${split}-s3dis.out

module load python/miniconda3-py39
source activate /scratch/hywluc001/conda-envs/pointnet

cd Pointnet_Pointnet2_pytorch
python3 train_masters.py \
--data_path "data/PatrickData/${dataset}/${split}" \
--epoch 60 \
--log_dir final/${dataset}_${split}_s3dis \
--decay_rate 1e-2 \
--log_merged_validation \
--log_merged_training_set \
--sample_all_validation \
--shuffle_training_data
" > slurm_scripts/P++_${dataset}_${split}_'s3dispascal'.sh



  done
done

