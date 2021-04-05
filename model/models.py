import numpy as np


def gaussian_mixture():
    # Number of mixture components
    n_components = np.arange(1, 10)
    # Number of covariance_type
    covariance_type = ['full', 'tied', 'diag', 'spherical']
    # Number of Initialization to perform
    n_init = np.arange(1, 10)

    gm_random_grid = {'n_components': n_components,
                      'covariance_type': covariance_type,
                      'n_init': n_init
                      }

    return gm_random_grid


def bayesian_gm():
    pass


def pca():
    pass


def iso_forest():
    pass

