for dataset in "Piazza"
do
  for split in "50%"
  do
    echo ${dataset}_${split}
    python3 train_masters.py \
    --data_path "data/PatrickData/${dataset}/50%" \
    --epoch 66 \
    --log_dir final/${dataset}_${split} \
    --decay_rate 1e-2 \
    --log_merged_validation \
    --log_merged_training_set \
    --sample_all_validation \
    --shuffle_training_data \
    --validate_only

    echo ${dataset}_${split}_s3dis
    python3 train_masters.py \
    --data_path "data/PatrickData/${dataset}/50%" \
    --epoch 66 \
    --log_dir final/${dataset}_${split}_s3dis \
    --decay_rate 1e-2 \
    --log_merged_validation \
    --log_merged_training_set \
    --sample_all_validation \
    --shuffle_training_data \
    --validate_only
  done
done

