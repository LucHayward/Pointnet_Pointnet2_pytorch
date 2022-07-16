import logging
from pprint import pprint, pformat

import torch.utils.data
from joblib import dump

import numpy as np
import pptk
from pathlib import Path
import wandb

from data_utils.MastersDataset import MastersDataset

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, jaccard_score

import argparse

from train_masters import setup_logging_dir, setup_logger, setup_wandb_classification_metrics, _log_string


def parse_args():
    parser = argparse.ArgumentParser()

    # Conform to the current convention on logging and data paths
    parser.add_argument('--log_dir', default='test')
    parser.add_argument('--data_path', help='The path to the folder containing the data in .npy formats',
                        default='data/PatrickData/Church/MastersFormat/hand_selected_reversed')

    # RF HParams
    parser.add_argument('--n_estimators', default=32, help='Number of trees to train')
    parser.add_argument('--max_depth', default=32, help='Maximum depth of the tree')
    parser.add_argument('--min_samples_split', default=20, help='')

    # Expected values
    parser.add_argument('--model', default='RF', help='name of the model, expected for logger')


    # Debugging
    parser.add_argument('--active_learning', action='store_true', default=None)
    parser.add_argument('--validation_repeats', default=None)



    return parser.parse_args()


def log_metrics(target, preds, prefix=None, logger=None) -> None:
    """
    Log the confusion matrix, the confusion matrix normalized over true_labels (category),
    and the precision, recall, accuracy and mIoU/Jaccard macro average.
    :param target: target binary labels
    :param preds: predicted binary labels
    :param prefix: Train/Validation
    """
    tn, fp, fn, tp = confusion_matrix(target, preds).ravel()
    cat_tn, cat_fp, cat_fn, cat_tp = confusion_matrix(target, preds, normalize='true').ravel()
    precision = precision_score(target, preds)
    recall = recall_score(target, preds)
    f1 = f1_score(target, preds)
    accuracy = accuracy_score(target, preds)
    keepIoU, discardIoU = jaccard_score(target, preds, average=None)
    mIoU = jaccard_score(target, preds, average='macro')

    metrics_dict = {'TP': tp,
                    'FP': fp,
                    'TN': tn,
                    'FN': fn,
                    'category-TP': cat_tp,
                    'category-FP': cat_fp,
                    'category-TN': cat_tn,
                    'category-FN': cat_fn,
                    'Precision': precision,
                    'Recall': recall,
                    'F1': f1,
                    'accuracy': accuracy,
                    'mIoU': mIoU}
    wandb.log({prefix: metrics_dict})
    metrics_dict['keepIoU'] = keepIoU
    metrics_dict['discardIoU'] = discardIoU
    _log_string(pformat(metrics_dict), logger)


def main(config):
    def log_string(str):
        _log_string(str, logger)

    # Setup logging and WandB metrics
    experiment_dir, log_dir, checkpoints_dir = setup_logging_dir(config, exp_dir='RF')
    logger = logging.getLogger("Model")
    setup_logger(logger, log_dir, config)
    log_string('PARAMETERS ...')
    log_string(config)
    setup_wandb_classification_metrics()

    # Setup training/validation data
    TRAIN_DATASET = MastersDataset('train', Path(config['data_path']), sample_all_points=True)
    VAL_DATASET = MastersDataset('validate', Path(config['data_path']), sample_all_points=True)
    val_data_loader = torch.utils.data.DataLoader(VAL_DATASET, batch_size=1, shuffle=False, num_workers=0)
    X_train, y_train = np.vstack(TRAIN_DATASET.data_segment), np.hstack(TRAIN_DATASET.labels_segment)
    X_val, y_val = np.vstack(VAL_DATASET.data_segment), np.hstack(VAL_DATASET.labels_segment)
    log_string(f"Training data: {X_train.shape}")
    log_string(f"Validation data: {X_val.shape}")

    # Setup classifier, train and perform predictions
    classifier = RandomForestClassifier(n_estimators=config["n_estimators"],
                                        max_depth=config['max_depth'],
                                        min_samples_split=config['min_samples_split'],
                                        n_jobs=4,
                                        verbose=1)
    classifier.fit(X=X_train, y=y_train)
    dump(classifier, checkpoints_dir / 'random_forest.joblib')
    log_string(f"Feature importances:\n{classifier.feature_importances_}")

    preds_train = classifier.predict(X_train)
    log_metrics(y_train, preds_train, 'Train', logger)

    # for i, sample in enumerate(val_data_loader):
    preds_val = classifier.predict(X_val)
    log_metrics(y_train, preds_train, 'Validation', logger)

    if config["active_learning"]:
        preds_vals = [preds_val]
        for i in range(int(config["validation_repeats"])-1):
            classifier.fit(X=X_train, y=y_train)
            preds_vals.append(classifier.predict(X_val))

        # here we could easily include probabilities or log_probs from the model instead of prediction variance
        preds_vals = np.array(preds_vals)
        pred_variances = preds_vals.var(axis=0)  # This should be the prediction variance at each point

        # Should be able to convert the points into cells by going backwards over samples_per_cell
        # for idx, segments_in_cell in enumerate(VAL_DATASET.grid_cell_to_segment)

        # Log the metrics and predictions for later use in active_learning.py
        log_string('Save model training predictions...')
        savepath = str(experiment_dir) + '/train_predictions.npz'
        np.savez_compressed(savepath, points=X_train, preds=preds_train, target=y_train)
        log_string('Saved model training predictions.')

        log_string('Save model validation predictions...')
        savepath = str(experiment_dir) + '/val_predictions.npz'
        log_string('Saving at %s' % savepath)
        np.savez_compressed(savepath, points=X_val,
                            preds=preds_val,
                            target=y_val, variance=None,  # cell_variance
                            point_variance=pred_variances,
                            grid_mask=VAL_DATASET.grid_mask,
                            features=X_val)


if __name__ == '__main__':
    import os

    args = parse_args()
    os.environ["WANDB_MODE"] = "dryrun"
    wandb.init(project="Masters-RF", config=args,
               # name='',
               # notes=''
               )
    wandb.run.log_code(".")
    main(wandb.config)
    wandb.finish()
