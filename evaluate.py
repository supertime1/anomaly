import pickle
import pandas as pd
import numpy as np


def evaluate():
    best_model = pickle.load(open('model/best_model.pkl', 'rb'))
    print(best_model.get_params())

    test_set = pd.read_csv('data/testing.csv')
    print('\nLoading test data file successfully!\n')
    test_X = test_set.drop(['index', 'id', 'class'], axis=1)
    test_Y = test_set['class'].copy()

    # import train set to calculate density threshold
    train_set = pd.read_csv('data/training.csv')
    print('Loading test data file successfully!\n')
    train_X = train_set.drop(['index', 'id', 'class'], axis=1)

    train_densities = best_model.score_samples(train_X)
    density_threshold = np.percentile(train_densities, 3)


    test_densities = best_model.score_samples(test_X)
    pred_anomaly = np.where(test_densities < density_threshold)
    true_anomaly = np.where(test_Y == 4)
    pred_anomaly = np.array(pred_anomaly)[0]
    true_anomaly = np.array(true_anomaly)[0]

    tp = fp = 0
    for pred in pred_anomaly:
        if pred in true_anomaly:
            tp += 1
        else:
            fp += 1

    fn = len(true_anomaly) - tp
    tn = len(test_Y) - len(true_anomaly) - fp

    accuracy = round((tp + tn) / (tp + tn + fp + fn), 2)
    precision = round(tp / (tp + fp), 2)
    recall = round(tp / (tp + fn), 2)

    print('\nEvaluation on TESTING data... \n')
    print(f'accuracy {accuracy}')
    print(f'precision {precision}')
    print(f'recall {recall}')


if __name__ == '__main__':

    print('Start evaluating...\n')
    evaluate()