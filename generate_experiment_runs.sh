#!/bin/bash

cd /home/luc/PycharmProjects/Pointnet_Pointnet2_pytorch/
for dataset in Church Lunnahoja Monument Bagni_Nerone Montelupo Piazza
do
  for split in "2.5%" "5%" "25%" "50%"
  do
  mkdir -p log/final/${dataset}_${split}/checkpoints
  mkdir -p log/final/${dataset}_${split}_s3dis/checkpoints
  cp log/temp/checkpoints/best_model.pth log/final/${dataset}_${split}_s3dis/checkpoints/best_model.pth

  echo python3 train_Masters.py \
--data_path "data/PatrickData/${dataset}/${split}" \
--epoch 60 \
--log_dir final/${dataset}_${split}_s3dis \
--decay_rate 1e-2 \
--log_merged_validation \
--log_merged_training_set \
--sample_all_validation \
--shuffle_training_data

 echo python3 train_Masters.py \
--data_path "data/PatrickData/${dataset}/${split}" \
--epoch 60 \
--log_dir final/${dataset}_${split} \
--decay_rate 1e-2 \
--log_merged_validation \
--log_merged_training_set \
--sample_all_validation \
--shuffle_training_data
  done
done

