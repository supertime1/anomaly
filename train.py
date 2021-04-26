from model import models
import mlflow
import argparse
import numpy as np
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
import pandas as pd
import os


def train(model_name, ds_name):
    train_ds_path = os.path.join(ds_name, 'training.csv')
    test_ds_path = os.path.join(ds_name, 'testing.csv')
    train_set = pd.read_csv(train_ds_path)
    test_set = pd.read_csv(test_ds_path)
    train_X = None
    print('Loading data file successfully!\n')
    if ds_name == 'breast_cancer':
        train_X = train_set.drop(['index', 'id', 'class'], axis=1)
        test_X = test_set.drop(['index', 'id', 'class'], axis=1)
    if ds_name == 'heart_disease':
        train_X = train_set.drop(['class'], axis=1)
        test_X = test_set.drop(['class'], axis=1)
    train_Y = train_set['class'].copy()
    test_Y = test_set['class'].copy()

    # don't need  data normalization
    model_dic = {'gm': models.gaussian_mixture(),
                 'bgm': models.bayesian_gm(),
                 'pca': models.pca(),
                 'iso': models.iso_forest(),
                 }

    config_file = model_dic[model_name]

    print('Current training params are:')
    print(config_file)
    print('\n')

    with mlflow.start_run() as run:
        bic_lst = []
        model_lst = []
        if model_name == 'gm':
            for n_component in config_file['n_components']:
                for cov_type in config_file['covariance_type']:
                    for n_init in config_file['n_init']:
                        gm = GaussianMixture(n_components=n_component,
                                             covariance_type=cov_type,
                                             n_init=n_init)
                        gm = gm.fit(train_X)
                        bic_lst.append(gm.bic(train_X))
                        model_lst.append(gm)

            min_bic = min(bic_lst)
            best_model = model_lst[bic_lst.index(min_bic)]
            mlflow.log_param("bic", min_bic)
            mlflow.sklearn.log_model(best_model, 'gm_model')

    print(best_model.get_params())

    def evaluate_model(ds_X, ds_Y, anomaly_label: int):
        densities = best_model.score_samples(ds_X)
        density_threshold = np.percentile(densities, 3)
        pred_anomaly = np.where(densities < density_threshold)
        true_anomaly = np.where(ds_Y == anomaly_label)
        pred_anomaly = np.array(pred_anomaly)[0]
        true_anomaly = np.array(true_anomaly)[0]

        tp = fp = 0
        for pred in pred_anomaly:
            if pred in true_anomaly:
                tp += 1
            else:
                fp += 1
        fn = len(true_anomaly) - tp
        tn = len(train_Y) - len(true_anomaly) - fp
        accuracy = round((tp + tn) / (tp + tn + fp + fn), 2)
        precision = round(tp / (tp + fp), 2)
        recall = round(tp / (tp + fn), 2)

        return accuracy, precision, recall

    # evaluate on training data
    train_acc, train_pre, train_rec = evaluate_model(train_X, train_Y, 4)
    print('\nEvaluation on TRAINING data... \n')
    print(f'accuracy {train_acc}')
    print(f'precision {train_pre}')
    print(f'recall {train_rec}')

    # evaluate on testing data
    test_acc, test_pre, test_rec = evaluate_model(test_X, test_Y, 4)
    print('\nEvaluation on TESTING data... \n')
    print(f'accuracy {test_acc}')
    print(f'precision {test_pre}')
    print(f'recall {test_rec}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-gm', action='store_true', help='select Gaussian Mixture Model as training model')
    parser.add_argument('--bgm', action='store_true', help='select Bayesian Gaussian Mixture Model as training model')
    parser.add_argument('--pca', action='store_true', help='select PCA as training model')
    parser.add_argument('--iso', action='store_true', help='select Isolation Forest as training model')
    parser.add_argument('--ds', action='store', help='choose which data set to train',
                        default='breast_cancer', type=str)
    args = parser.parse_args()
    # set a default model type

    model_name = ('gm' if args.gm else
                  'bgm' if args.bgm else
                  'pca' if args.pca else
                  'iso' if args.iso else
                  'gm')

    print('Start training...\n')
    train(model_name, args.ds)
