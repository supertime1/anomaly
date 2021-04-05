from model import models
import mlflow
import argparse
import numpy as np
from sklearn.mixture import GaussianMixture
import pandas as pd


def train(model_name):

    train_set = pd.read_csv('data/train.csv')
    print('Loading data file successfully!\n')
    # heart_labels = train_set['target'].copy()
    heart = train_set.drop('target', axis=1)

    # don't need data normalization
    model_dic = {'gm': models.gaussian_mixture(),
                 'bgm': models.bayesian_gm(),
                 'pca': models.pca(),
                 'iso': models.iso_tree(),
                 }

    config_file = model_dic[model_name]

    print('Current training params are:')
    print(config_file)
    print('\n')


    with mlflow.start_run() as run:
        if model_name == 'gm':
            for n_component in config_file['n_components']:
                for cov_type in config_file['covariance_type']:
                    for n_init in config_file['n_init']:
                        gm = GaussianMixture(n_components=n_component,
                                             covariance_type=cov_type,
                                             n_init=n_init)
                        gm.fit(heart)

                        mlflow.log_param("bic", gm.bic(heart))
                        mlflow.sklearn.log_model(gm, 'gm_model')


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



