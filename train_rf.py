import argparse
import logging
import time
# import pptk
from pathlib import Path
from pprint import pformat

import numpy as np
import torch.utils.data
import wandb
from joblib import dump, load
# from line_profiler_pycharm import profile
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, jaccard_score
from tqdm import tqdm

from train_masters import setup_logging_dir, setup_logger, setup_wandb_classification_metrics, _log_string
from data_utils.MastersDataset import MastersDataset


def parse_args():
    parser = argparse.ArgumentParser()

    # Conform to the current convention on logging and data paths
    parser.add_argument('--log_dir', default='test')
    parser.add_argument('--data_path', help='The path to the folder containing the data in .npy formats',
                        default='data/PatrickData/Church/MastersFormat/hand_selected_reversed')

    # RF HParams
    parser.add_argument('--n_estimators', type=int, default=32, help='Number of trees to train')
    parser.add_argument('--max_depth', type=int, default=32, help='Maximum depth of the tree')
    parser.add_argument('--min_samples_split', type=int, default=20, help='')
    parser.add_argument('--xgboost', action='store_true', help='Fit XGBoost as well as sklearn RF')

    # Expected values
    parser.add_argument('--model', default='RF', help='name of the model, expected for logger')


    # Debugging
    parser.add_argument('--active_learning', action='store_true', default=None)
    parser.add_argument('--validation_repeats', default=None)


    return parser.parse_args()

def setup_wandb_classification_metrics():
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

    # Validation Classification metrics
    wandb.define_metric('Validation/TP', summary='max')
    wandb.define_metric('Validation/FP', summary='min')
    wandb.define_metric('Validation/TN', summary='max')
    wandb.define_metric('Validation/FN', summary='min')

    wandb.define_metric('Validation/category-TP', summary='max')
    wandb.define_metric('Validation/category-FP', summary='min')
    wandb.define_metric('Validation/category-TN', summary='max')
    wandb.define_metric('Validation/category-FN', summary='min')

    wandb.define_metric('Validation/Precision', summary='max')
    wandb.define_metric('Validation/Recall', summary='max')
    wandb.define_metric('Validation/F1', summary='max')
    wandb.define_metric('Validation/mIoU', summary='max')
    wandb.define_metric('Validation/accuracy', summary='max')

    # XGBoost Classification metrics
    wandb.define_metric('XGB_Train/TP', summary='max')
    wandb.define_metric('XGB_Train/FP', summary='min')
    wandb.define_metric('XGB_Train/TN', summary='max')
    wandb.define_metric('XGB_Train/FN', summary='min')

    wandb.define_metric('XGB_Train/category-TP', summary='max')
    wandb.define_metric('XGB_Train/category-FP', summary='min')
    wandb.define_metric('XGB_Train/category-TN', summary='max')
    wandb.define_metric('XGB_Train/category-FN', summary='min')

    wandb.define_metric('XGB_Train/Precision', summary='max')
    wandb.define_metric('XGB_Train/Recall', summary='max')
    wandb.define_metric('XGB_Train/F1', summary='max')
    wandb.define_metric('XGB_Train/mIoU', summary='max')
    wandb.define_metric('XGB_Train/accuracy', summary='max')

    # Validation Classification metrics
    wandb.define_metric('XGB_Validation/TP', summary='max')
    wandb.define_metric('XGB_Validation/FP', summary='min')
    wandb.define_metric('XGB_Validation/TN', summary='max')
    wandb.define_metric('XGB_Validation/FN', summary='min')

    wandb.define_metric('XGB_Validation/category-TP', summary='max')
    wandb.define_metric('XGB_Validation/category-FP', summary='min')
    wandb.define_metric('XGB_Validation/category-TN', summary='max')
    wandb.define_metric('XGB_Validation/category-FN', summary='min')

    wandb.define_metric('XGB_Validation/Precision', summary='max')
    wandb.define_metric('XGB_Validation/Recall', summary='max')
    wandb.define_metric('XGB_Validation/F1', summary='max')
    wandb.define_metric('XGB_Validation/mIoU', summary='max')
    wandb.define_metric('XGB_Validation/accuracy', summary='max')

# @profile
def log_metrics(target, preds, prefix=None, logger=None) -> None:
    """
    Log the confusion matrix, the confusion matrix normalized over true_labels (category),
    and the precision, recall, accuracy and mIoU/Jaccard macro average.
    :param target: target binary labels
    :param preds: predicted binary labels
    :param prefix: Train/Validation
    """
    #    pn pp
    # an tn fp
    # ap fn tp
    tn, fp, fn, tp = confusion_matrix(target, preds).ravel()
    cat_tn, cat_fp, cat_fn, cat_tp = confusion_matrix(target, preds, normalize='true').ravel()
    precision = precision_score(target, preds)
    recall = recall_score(target, preds)
    f1 = f1_score(target, preds)
    accuracy = accuracy_score(target, preds)
    keepIoU, discardIoU = jaccard_score(target, preds, average=None)
    mIoU = jaccard_score(target, preds, average='macro')

    metrics_dict = {f'{prefix}/TP': tp,
                    f'{prefix}/FP': fp,
                    f'{prefix}/TN': tn,
                    f'{prefix}/FN': fn,
                    f'{prefix}/category-TP': cat_tp,
                    f'{prefix}/category-FP': cat_fp,
                    f'{prefix}/category-TN': cat_tn,
                    f'{prefix}/category-FN': cat_fn,
                    f'{prefix}/Precision': precision,
                    f'{prefix}/Recall': recall,
                    f'{prefix}/F1': f1,
                    f'{prefix}/accuracy': accuracy,
                    f'{prefix}/mIoU': mIoU}
    wandb.log(metrics_dict)
    metrics_dict[f'{prefix}/keepIoU'] = keepIoU
    metrics_dict[f'{prefix}/discardIoU'] = discardIoU
    _log_string(pformat(metrics_dict), logger)

def classification_confidence_from_diverging_probability(probs):
    """
    Given a set of probabilities [0,1] for a binary classifier (closer to 0/1 is more probable of that value),
    converts these to classification certainties.
    i.e. the bigger the value [0,1] the more confident the model was in its prediction (regrdless of class)
    :param probs: Diverging probabilities (probability of class 1)
    :return: classification confidence irrespective of class.
    """
    # Shift the numbers from [0,1] to [-.5,.5] -> [-1,1] -> [0,1]
    return np.abs((probs-0.5)*2)

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
    if config["active_learning"]:
        TRAIN_DATASET = MastersDataset('train', Path(config['data_path']), sample_all_points=True)
        VAL_DATASET = MastersDataset('validate', Path('/'.join(config['data_path'].split('/')[:-1])+"/50%"), sample_all_points=True)
        # val_data_loader = torch.utils.data.DataLoader(VAL_DATASET, batch_size=1, shuffle=False, num_workers=0)
        # CHECK might need to undo this to make getting the variance back easier
        X_train, y_train = TRAIN_DATASET.segment_points[0], TRAIN_DATASET.segment_labels[0]
        X_val, y_val = VAL_DATASET.segment_points[0], VAL_DATASET.segment_labels[0]
    else:
        # Setup training/validation data
        TRAIN_DATASET = np.load(
            "/home/luc/PycharmProjects/Pointnet_Pointnet2_pytorch/data/PatrickData/Piazza/2.5%/train.npy")
        VAL_DATASET = np.load(
            "/home/luc/PycharmProjects/Pointnet_Pointnet2_pytorch/data/PatrickData/Piazza/2.5%/validate.npy")
        # val_data_loader = torch.utils.data.DataLoader(VAL_DATASET, batch_size=1, shuffle=False, num_workers=0)
        # CHECK might need to undo this to make getting the variance back easier
        X_train, y_train = TRAIN_DATASET[:, :3], TRAIN_DATASET[:, -1]
        X_val, y_val = VAL_DATASET[:, :3], VAL_DATASET[:, -1]
    log_string(f"Training data: {X_train.shape}")
    log_string(f"Validation data: {X_val.shape}")

    st = time.time()
    print("Training Random Forest")
    if (checkpoints_dir/'random_forest.joblib').exists():
        log_string("Loading prefit random forest")
        classifier = load(checkpoints_dir/'random_forest.joblib')
    else:
        # Setup classifier, train and perform predictions
        classifier = RandomForestClassifier(n_estimators=config["n_estimators"],
                                            max_depth=config['max_depth'],
                                            min_samples_split=config['min_samples_split'],
                                            n_jobs=8,
                                            verbose=1)
        classifier.fit(X=X_train, y=y_train)
        dump(classifier, checkpoints_dir / 'random_forest.joblib')
        wandb.save(str(checkpoints_dir / 'random_forest.joblib'))
    preds_train, preds_val = train_predict(X_train, X_val, classifier, log_string, logger, y_train, y_val)
    print(f"RF: {time.time() - st}")

    if config['xgboost']:
        st = time.time()
        import xgboost as xgb
        from xgboost import XGBClassifier

        if (checkpoints_dir/'xgboost.model').exists():
            log_string("Loading prefit XGBoost model")
            xgboost_model = xgb.Booster()
            xgboost_model.load_model(str(checkpoints_dir/'xgboost.model'))
        else:
            log_string("Training XGBoost")
            xgboost_model = XGBClassifier(n_estimators=config["n_estimators"],
                                    max_depth=config['max_depth'],
                                    n_jobs=8,
                                    )
            # xgboost = XGBClassifier()
            xgboost_model.fit(X=X_train, y=y_train)
            xgboost_model.save_model(str(checkpoints_dir / 'xgboost.model'))
            wandb.save(str(checkpoints_dir / 'xgboost.model'))

        preds_train_xgboost, preds_val_xgboost = train_predict(X_train, X_val, xgboost_model, log_string, logger, y_train, y_val, True)
        print(f"XGBoost: {time.time() - st}")


    print()
    if config["active_learning"]:
        if config["uncertainty_metric"] == "variance":
            preds_vals = [preds_val]
            for i in range(int(config["validation_repeats"])-1):
                classifier.fit(X=X_train, y=y_train)
                preds_vals.append(classifier.predict(X_val))

            # here we could easily include probabilities or log_probs from the model instead of prediction variance
            preds_vals = np.array(preds_vals)
            pred_uncertainty = preds_vals.var(axis=0)  # This should be the prediction variance at each point
        elif config["uncertainty_metric"] == "probability":
            pred_probs = classifier.predict_proba(X_val)
            # 1- to make more confident predictions closer to 0 (uncertainty)
            pred_uncertainty = 1-classification_confidence_from_diverging_probability(pred_probs[:,1])

        # Should be able to convert the points into cells by going backwards over samples_per_cell
        # Collect all the points together per cell, get the mean variance
        cell_uncertainty = []
        cell_features = []
        for cell_idx in tqdm(np.unique(VAL_DATASET.grid_mask), desc='Rebuilding cells'):
            cell_idx_mask = VAL_DATASET.grid_mask == cell_idx
            cell_uncertainty.append(np.mean(pred_uncertainty[cell_idx_mask]))
            cell_features.append(np.mean(X_val[cell_idx_mask], axis=0))

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
                            target=y_val, variance=cell_uncertainty,  # cell_variance OR pred_probs?
                            point_variance=pred_uncertainty,
                            grid_mask=VAL_DATASET.grid_mask,
                            features=cell_features)


def train_predict(X_train, X_val, classifier, log_string, logger, y_train, y_val, xgboost=False):
    if xgboost: print("Producing results for XGBoost")
    log_string(f"Feature importances:\n{classifier.feature_importances_}")
    preds_train = classifier.predict(X_train)
    prefix = 'Train' if not xgboost else 'XGB_Train'
    log_metrics(y_train, preds_train, prefix, logger)
    # for i, sample in enumerate(val_data_loader):
    preds_val = classifier.predict(X_val)
    prefix = 'Validation' if not xgboost else 'XGB_Validation'
    log_metrics(y_val, preds_val, prefix, logger)
    return preds_train, preds_val


def retrain_rf(dataset_path):
    def log_metrics(target, preds, prefix=None, logger=None) -> None:
        """
        Log the confusion matrix, the confusion matrix normalized over true_labels (category),
        and the precision, recall, accuracy and mIoU/Jaccard macro average.
        :param target: target binary labels
        :param preds: predicted binary labels
        :param prefix: Train/Validation
        """
        #    pn pp
        # an tn fp
        # ap fn tp
        tn, fp, fn, tp = confusion_matrix(target, preds).ravel()
        cat_tn, cat_fp, cat_fn, cat_tp = confusion_matrix(target, preds, normalize='true').ravel()
        precision = precision_score(target, preds)
        recall = recall_score(target, preds)
        f1 = f1_score(target, preds)
        accuracy = accuracy_score(target, preds)
        keepIoU, discardIoU = jaccard_score(target, preds, average=None)
        mIoU = jaccard_score(target, preds, average='macro')
        print(accuracy)
        print(mIoU)

    def train_predict(X_train, X_val, classifier, y_train, y_val, xgboost=False):
        preds_val = classifier.predict(X_val)
        log_metrics(y_val, preds_val)
        return preds_val

    st = time.time()
    # Setup training/validation data
    TRAIN_DATASET = np.load(
        f"{dataset_path}train.npy")
    VAL_DATASET = np.load(
        f"{dataset_path}validate.npy")
    # val_data_loader = torch.utils.data.DataLoader(VAL_DATASET, batch_size=1, shuffle=False, num_workers=0)
    # CHECK might need to undo this to make getting the variance back easier
    X_train, y_train = TRAIN_DATASET[:, :3], TRAIN_DATASET[:, -1]
    X_val, y_val = VAL_DATASET[:, :3], VAL_DATASET[:, -1]
    print("Training Random Forest")
    # Setup classifier, train and perform predictions
    classifier = RandomForestClassifier(n_estimators=32,
                                        max_depth=32,
                                        min_samples_split=20,
                                        n_jobs=8,
                                        verbose=1)
    classifier.fit(X=X_train, y=y_train)
    preds_val = train_predict(X_train, X_val, classifier, y_train, y_val)
    print(f"RF: {time.time() - st}")


if __name__ == '__main__':
    import os

    # python --data_path data/PatrickData/Bagni_Nerone/2.5% --log_dir Bagni_Nerone_2.5% --n_estimators {32, 64} --max_depth ={16,32,64,128,256}
    args = parse_args()
    os.environ["WANDB_MODE"] = "dryrun"
    wandb.init(project="Masters-RF", config=args, resume=False,
               name=f"{'-'.join(args.data_path.split('/')[-2:])}_50%Val",
               notes="",
               group="50%Validation")
    wandb.run.log_code(".")
    setup_wandb_classification_metrics()
    main(wandb.config)
    wandb.finish()
