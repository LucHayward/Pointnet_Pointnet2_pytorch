import numpy as np
import pptk
from pathlib import Path
import wandb

from data_utils.MastersDataset import MastersDataset

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, jaccard_score

import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # Conform to the current convention on logging and data paths
    parser.add_argument('--log_dir')
    parser.add_argument('--data_path')

    # RF HParams
    parser.add_argument('--n_estimators', default=32, help='Number of trees to train')
    parser.add_argument('--max_depth', default=32, help='Maximum depth of the tree')
    parser.add_argument('--min_samples_split', default=20, help='')

    return parser.parse_args()


def setup_wandb_metrics():
    # Train classification metrics
    wandb.define_metric('Train/TP', summary='max')
    wandb.define_metric('Train/FP', summary='min')
    wandb.define_metric('Train/TN', summary='max')
    wandb.define_metric('Train/FN', summary='min')

    wandb.define_metric('Train/category-TP', summary='max')
    wandb.define_metric('Train/category-FP', summary='min')
    wandb.define_metric('Train/category-TN', summary='max')
    wandb.define_metric('Train/category-FN', summary='min')

    wandb.define_metric('Train/Precision', summary='max')
    wandb.define_metric('Train/Recall', summary='max')
    wandb.define_metric('Train/F1', summary='max')
    wandb.define_metric('Train/mIoU', summary='max')
    wandb.define_metric('Train/accuracy', summary='max')
    wandb.define_metric('Train/mean_loss', summary='min')

    # Validation Classification metrics
    wandb.define_metric('validation/TP', summary='max')
    wandb.define_metric('validation/FP', summary='min')
    wandb.define_metric('validation/TN', summary='max')
    wandb.define_metric('validation/FN', summary='min')

    wandb.define_metric('validation/category-TP', summary='max')
    wandb.define_metric('validation/category-FP', summary='min')
    wandb.define_metric('validation/category-TN', summary='max')
    wandb.define_metric('validation/category-FN', summary='min')

    wandb.define_metric('Validation/Precision', summary='max')
    wandb.define_metric('Validation/Recall', summary='max')
    wandb.define_metric('Validation/F1', summary='max')
    wandb.define_metric('Validation/eval_point_mIoU', summary='max')
    wandb.define_metric('Validation/eval_point_accuracy', summary='max')
    wandb.define_metric('Validation/eval_point_avg_class_accuracy', summary='max')
    wandb.define_metric('Validation/eval_mean_loss', summary='min')


def log_metrics(target, preds, prefix=None) -> None:
    """
    Log the confusion matrix, the confusion matrix normalized over true_labels (category),
    and the precision, recall, accuracy and mIoU/Jaccard macro average.
    :param target: target binary labels
    :param preds: predicted binary labels
    :param prefix: Train/Validation
    """


def main(config):
    train_data = MastersDataset('train', config['data_path'], sample_all_points=True)
    val_data = MastersDataset('validate', config['data_path'], sample_all_points=True)

    classifier = RandomForestClassifier(n_estimators=config["n_estimators"],
                                        max_depth=config['max_depth'],
                                        min_samples_split=config['min_samples_split'])

    X_train, y_train = np.vstack(train_data.data_segment), np.hstack(train_data.labels_segment)
    classifier.fit(X=X_train, y=y_train)
    preds_train = classifier.predict(X_train)


if __name__ == '__main__':
    args = parse_args()
    # os.environ["WANDB_MODE"] = "dryrun"
    wandb.init(project="Masters-RF", config=args,
               # name='',
               # notes=''
               )
    wandb.run.log_code(".")
    main(wandb.config)
    wandb.finish()
