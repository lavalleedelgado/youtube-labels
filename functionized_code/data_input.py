import numpy as np
import pandas as pd
import random
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchtext import data
import torch.utils.data as d
import tokenization_dim_reduction as tdr

FEATURE_GROUPS = {"title": (tdr.cols_t1, -1, [0]), "tag": (tdr.cols_t2, 0, []), "title_tag": (tdr.cols_t3, 1, [0]), "full": (tdr.cols_t4, 1, [0,2])}

def load_data(data_dir, label_num, feature_g):
    '''
    The function is used to load data for the following
    analysis, which also seperates features and label
    Inputs:
        data_dir: the directory of the data
        label_num: number of specific category
        feature_g: the key for feature groups
    Returns:
        new_TEXT: loaded features
        new_label: loaded labels
        new_arr: loaded features with labels
    '''
    f_set = FEATURE_GROUPS[feature_g]
    _, dtext, dlabel = tdr.select_col(data_dir, f_set[0])
    new_TEXT = tdr.combine_text(dtext, f_set[1], f_set[2])
    new_label = tdr.multi_to_binary(dlabel, label_num)
    new_arr = np.concatenate((new_TEXT.reshape([len(new_TEXT),1]), new_label), axis=1)

    print((new_label[new_label == 1].shape[0] / new_label.shape[0]) * 100,
          " percent of videos are labelled as the selected category")
    print("the baseline precision is ", 
          (new_label[new_label == 1].shape[0] / new_label.shape[0]) * 100,
         " in this model")

    return new_TEXT, new_label, new_arr


def split_train_test(dt_size, train_valid_test_r):
    '''
    The function randomly selects the indices for
    training, validation, and testing sets
    Inputs:
        dt_size: number of rows
        train_valid_test_r: tuple of ratios
    Return: indices for each subset
    '''
    train, valid, testg = train_valid_test_r
    # Ensure the proportions sum to one.
    assert sum([train, valid, testg]) == 1
    # Calculate the size of each set.
    len_train = int(dt_size * train)
    len_valid = int(dt_size * valid)
    len_testg = dt_size - len_train - len_valid
    # Report the size of each set.
    message = 'The size of train, valid, and test sets are %d, %d, %d.'
    print(message % (len_train, len_valid, len_testg))
    # Proportionally allocate a random permutation of row indices.
    idx = np.random.permutation(dt_size)
    idx_train = idx[:len_train]
    idx_valid = idx[len_train:len_train + len_valid]
    idx_testg = idx[len_train + len_valid:]
    # Return the indices of each set.
    return idx_train, idx_valid, idx_testg


def split_data(path, arr, train_valid_test_r):
    '''
    The function split the data to train, validation and test
    sets with randomly selected indices and save them to seperated
    csv files
    Inputs:
        path: directory of the saved files
        arr: the whole dataset
        train_valid_test_r: tuple of ratios
    '''
    train_indices, valid_indices, test_indices = split_train_test(arr.shape[0], train_valid_test_r)
    pd.DataFrame(arr[train_indices]).to_csv(path + "\\train.csv", header=None, index=None)
    pd.DataFrame(arr[valid_indices]).to_csv(path + "\\valid.csv", header=None, index=None)
    pd.DataFrame(arr[test_indices]).to_csv(path + "\\test.csv", header=None, index=None)


def build_train_test(path, new_arr, train_valid_test_r, TEXT, LABEL):
    '''
    The function splits data into train, valid and test dataset
    Inputs:
        path: the directory of data
        new_arr: array with features and labels
        train_valid_test_r: tuple of ratios
        TEXT: TEXT object used in vocabulary building
        LABEL: LABEL object used in vocabulary building
    Returns: train, valid, and test set
    '''
    split_data(path, new_arr, train_valid_test_r)
    fields = [("text", TEXT), ("label", LABEL)]
    train_data, valid_data, test_data = data.TabularDataset.splits(
                                            path = path,
                                            train = 'train.csv',
                                            validation = 'valid.csv',
                                            test = 'test.csv',
                                            format = 'csv',
                                            fields = fields,
                                            skip_header = True)

    print(f'Number of training examples: {len(train_data)}')
    print(f'Number of testing examples: {len(test_data)}')
    print(f'Number of validation examples:{len(valid_data)}')

    return train_data, valid_data, test_data


def build_iterator(batch_size, device, train_data, valid_data, test_data):
    '''
    The function splits train, valid, and test dataset with multiple batches
    Inputs:
        batch_size: batch size
        device: the device object assigned for the operation
        train_data: training data set
        valid_data: validation data set
        test_data: testing data set
    Returns: train_iterator, valid_iterator, test_iterator with batches
    '''
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
            (train_data, valid_data, test_data), 
             batch_size = batch_size,
             sort_key = lambda x: len(x.text),
             sort_within_batch = True, 
             device = device)

    return train_iterator, valid_iterator, test_iterator


# The following function is for the data loading for TFIDF and Machine Learning
def get_tfidf_matrix(data_dir, label_num, feature_g, topk):
    '''
    The function creates the tfidf matrix from specific dataset
    and select the features with the highest topk average TFIDF
    Inputs:
        data_dir: the directory of the data
        label_num: number of specific category
        feature_g: the key for feature groups
        topk: topk for highest average TFIDF
    Returns:
        new_TEXT: loaded features
        new_label: loaded labels
        new_arr: loaded features with labels
    '''
    f_set = FEATURE_GROUPS[feature_g]
    _, dtext, dlabel = tdr.select_col(data_dir, f_set[0])
    new_TEXT = tdr.combine_text(dtext, f_set[1], f_set[2])

    txt_tfidf = tdr.tfidf_tokenization(new_TEXT)
    new_TEXT = txt_tfidf.toarray()
    new_label = tdr.multi_to_binary(dlabel, label_num)
    new_arr = np.concatenate((new_TEXT, new_label), axis=1)

    top5k_indices = np.argsort(np.apply_along_axis(np.mean, 0, new_TEXT))[ -topk: ]
    new_TEXT = new_TEXT[:, top5k_indices]
    new_arr = np.concatenate((new_TEXT, new_label), axis=1)
    print("the current shape of the reduced data is ", new_TEXT.shape)

    return new_TEXT, new_label, new_arr


def split_reduced_data(path, arr, y, k=500, train_valid_test_r=(0.4, 0.4, 0.2)):
    '''
    The function split the data to train, validation and test
    sets with dimensional reduction and randomly selected indices.
    The datasets are saved to seperated csv files
    Inputs:
        path: directory of the saved files
        arr: the whole dataset
        y: the label array
        k: traget dimension for dimensional reduction
        train_valid_test_r: tuple of ratios
    Return: updated sets for train, validation and test
    '''
    train_indices, valid_indices, test_indices = split_train_test(arr.shape[0], train_valid_test_r)
    
    red_train, red_valid = tdr.dimensional_reduction(arr[train_indices], k, True, arr[valid_indices])
    red_train, red_test = tdr.dimensional_reduction(arr[train_indices], k, True, arr[test_indices])
    
    pd.DataFrame(red_train).to_csv(path + "\\train.csv", header=None, index=None)
    pd.DataFrame(red_valid).to_csv(path + "\\valid.csv", header=None, index=None)
    pd.DataFrame(red_test).to_csv(path + "\\test.csv", header=None, index=None)
    
    return (red_train, y[train_indices], red_valid, y[valid_indices], 
            red_test, y[test_indices])

