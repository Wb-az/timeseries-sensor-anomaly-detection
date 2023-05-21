import os
import numpy as np
import pandas as pd
from natsort import natsorted
from scipy.fft import fft
from scipy.signal import detrend


def dir_list(path=None):
    """
    :param path: path to the dataset
    :return: an order directory list
    """
    list_dir = natsorted(os.listdir(path))
    return list_dir


def concat_raw_data(path=None, csv_path=None, dataset=None, fourier_tr=True,
                    detrends=True):
    """
    :param detrends: boolean to detrend the signal
    :param fourier_tr: boolean tranfor singal with furier transform
    :param path: a path to the dataset file
    :param dataset: number of the dataset to be processes from the three dataset available
    :param csv_path: path save the contenated files into a csv file
    :return: data dataset with average and std from each file at each time step
    """

    list_dir = dir_list(path)

    col_dual = list()
    for b in range(0, 4):
        b1 = f'b{b + 1}_ch{b * 2 + 1}'
        b2 = f'b{b + 1}_ch{b * 2 + 2}'
        col_dual.extend([b1, b2])

    col_names = [f'b{i + 1}_ch{i + 1}' for i in range(0, 4)]

    dataset_dict = {}

    for i, f in enumerate(list_dir):
        temp_df = pd.read_csv(os.path.join(path, f), sep='\t', header=None)
        if len(temp_df.columns) == 8:
            temp_df.columns = col_dual
        else:
            temp_df.columns = col_names
        temp_df.insert(0, 'date', len(temp_df) * [f])
        dataset_dict[f] = temp_df

    df = pd.concat(list(dataset_dict.values()), ignore_index=True)
    df.set_index('date', inplace=True)
    df.index = pd.to_datetime(df.index, format='%Y.%m.%d.%H.%M.%S')

    os.makedirs(csv_path, exist_ok=True)

    if fourier_tr:
        return fourier_transforms(df, path=csv_path, dataset=dataset, detrends=detrends)

    else:
        fname = os.path.join(csv_path, f'concat_dataset_{dataset}.csv')

        return df.to_csv(fname)


def fourier_transforms(data_frame, path=None, dataset=1, detrends=True):
    """
    :param data_frame: datafram with the concatenated raw data
    :param path: a path to store the csv file
    :param dataset: number of dataste processed
    :param detrends: a boolean to detrend before applyin fourier transformations
    :return: save dataframe as csv
    """

    os.makedirs(path, exist_ok=True)
    fname = os.path.join(path, f'fft_dataset_{dataset}.csv')
    df_fft = data_frame.copy()

    for col in df_fft.columns:
        if detrends:
            fft_col = fft(detrend(df_fft[col].values))
        else:
            fft_col = fft(df_fft[col].values)

        df_fft[col] = np.abs(fft_col)

    return df_fft.to_csv(fname)


def average_signal_dataset(path=None, dataset=1, csv_path=None):
    """
    :param path:  str a path to the dataset file
    :param dataset:  int the number of the daset to be process
    :param csv_path:  str a path to save the datframes to csv file
    :return: data dataset with average and std from each file at each time step
    """
    list_dir = dir_list(path)

    if dataset == 1:
        col_names = list()
        for b in range(0, 4):
            b1 = f'b{b + 1}_ch{b * 2 + 1}'
            b2 = f'b{b + 1}_ch{b * 2 + 2}'
            col_names.extend([b1, b2])
    else:
        col_names = [f'b{i + 1}_ch{i + 1}' for i in range(0, 4)]

    dataset_dict = {}

    for file in list_dir:
        temp_df = pd.read_csv(os.path.join(path, file), sep='\t', header=None)
        # mean_std_values = np.append(temp_df.abs().mean().values, temp_df.abs().std().values)
        mean = temp_df.abs().mean().values
        dataset_dict[file] = mean

    df = pd.DataFrame.from_dict(dataset_dict, orient='index', columns=col_names)
    df.index = pd.to_datetime(df.index, format='%Y.%m.%d.%H.%M.%S')
    os.makedirs(csv_path, exist_ok=True)

    fname = os.path.join(csv_path, f'avg_concat_dataset_{dataset}.csv')

    return df.to_csv(fname)


def generate_datasets(datasets_dict=None, csv_path=None):
    for i, (k, v) in enumerate(datasets_dict.items()):
        dataset_num = i + 1
        average_signal_dataset(v, dataset=dataset_num, csv_path=csv_path)
        concat_raw_data(v, csv_path, dataset=dataset_num, fourier_tr=True, detrends=False)


if __name__ == '__main__':
    datasets = {'dataset_path1': './ims_bearing/1st_test/1st_test',
                'dataset_path2': './ims_bearing/2nd_test/2nd_test',
                'dataset_path3': './ims_bearing/3rd_test/4th_test/txt'}

    csv_dir = os.path.join(os.getcwd(), 'csv_data')
    os.makedirs(csv_dir, exist_ok=True)

    generate_datasets(datasets, csv_path=csv_dir)