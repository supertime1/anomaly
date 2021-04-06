from model import models
import mlflow
import argparse
import numpy as np
from sklearn.mixture import GaussianMixture
import pandas as pd


def train(model_name):
    train_set = pd.read_csv('data/training.csv')
    print('Loading data file successfully!\n')
    train_X = train_set.drop(['index', 'id', 'class'], axis=1)
    train_Y = train_set['class'].copy()

    # don't need data normalization
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

    densities = best_model.score_samples(train_X)
    density_threshold = np.percentile(densities, 3)
    pred_anomaly = np.where(densities < density_threshold)
    true_anomaly = np.where(train_Y == 4)
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

    print('\nEvaluation on TRAINING data... \n')
    print(f'accuracy {accuracy}')
    print(f'precision {precision}')
    print(f'recall {recall}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-gm', action='store_true', help='select Gaussian Mixture Model as training model')
    parser.add_argument('--bgm', action='store_true', help='select Bayesian Gaussian Mixture Model as training model')
    parser.add_argument('--pca', action='store_true', help='select PCA as training model')
    parser.add_argument('--iso', action='store_true', help='select Isolation Forest as training model')

    args = parser.parse_args()
    # set a default model type

    model_name = ('gm' if args.gm else
                  'bgm' if args.bgm else
                  'pca' if args.pca else
                  'iso' if args.iso else
                  'gm')

    print('Start training...\n')
    train(model_name)
