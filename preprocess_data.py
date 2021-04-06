import pandas as pd
import numpy as np
import argparse


def preprocess_data(data_path, normal_ratio, anom_ratio):
    """

    :param data_path: data file path
    :param normal_ratio: normal data ratio (among overall nomral data) in training data
    :param anom_ratio: anomaly data ratio (among normal data in training data) in training
    data
    :return: train and test dataset
    """
    col_name = ['id', 'ct', 'size', 'shape', 'adhesion', ' single', 'bn', 'bc', 'nn', 'mitose', 'class']
    df = pd.read_table(data_path, sep=',', names=col_name)
    df.replace({'?': None}, inplace=True)
    df.dropna(inplace=True)
    df = df.reset_index()
    total_num_normal = np.count_nonzero(df['class'] == 2)
    total_num_anomaly = np.count_nonzero(df['class'] == 4)

    np.random.seed(10)
    random_normal_index = np.random.choice(np.arange(total_num_normal),
                                           int(normal_ratio * total_num_normal),
                                           replace=False)
    normal_index = df[df['class'] == 2].index[random_normal_index]

    np.random.seed(10)
    random_anomaly_index = np.random.choice(np.arange(total_num_anomaly),
                                            int(anom_ratio * normal_ratio * total_num_normal),
                                            replace=False)
    anomaly_index = df[df['class'] == 4].index[random_anomaly_index]

    training_data_index = np.concatenate((np.asarray(normal_index), np.asarray(anomaly_index)))
    training_data = df.iloc[training_data_index]
    bad_df = df.index.isin(training_data_index)
    testing_data = df[~bad_df]

    print(f'There are {len(training_data)} in training dataset, '
          f'with {len(normal_index)} normal data, and {len(anomaly_index)} anomaly data')
    print(f'There are {len(testing_data)} in testing dataset, '
          f'with {total_num_normal - len(normal_index)} normal data, '
          f'and {total_num_anomaly - len(anomaly_index)} anomaly data')

    return training_data, testing_data


if __name__ == '__main__':
    data_path = 'data/data'
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', action='store', help='input normal_ratio', type=float)
    parser.add_argument('--an', action='store', help='input anom_ratio', type=float)

    args = parser.parse_args()

    # set a default value
    normal_ratio = 0.9
    if args.n:
        normal_ratio = args.n

    anom_ratio = 0.03
    if args.an:
        anom_ratio = args.an

    training_data, testing_data = preprocess_data(data_path, normal_ratio, anom_ratio)
    training_data.to_csv('data/training.csv', index=False)
    testing_data.to_csv('data/testing.csv', index=False)
