from typing import Tuple, List, Dict, Set, Any, Type
import os
import argparse
import yaml
import pickle
import numpy as np
import pandas as pd
from torch import nn, optim
from functionized_code import data_pipeline as dp, nnmodels, train_eval


# Identify paths to the data.
PATH_VIDEOS = 'data/USvideos.csv'
PATH_CAPTIONS = 'data/transcripts.txt'


# Identify labels in the data.
LABEL_TO_IDX = {
    'Film & Animation': 1,
    'Autos & Vehicles': 2,
    'Music': 10,
    'Pets & Animals': 15,
    'Sports': 17,
    'Short Movies': 18,
    'Travel & Events': 19,
    'Gaming': 20,
    'Videoblogging': 21,
    'People & Blogs': 22,
    'Comedy': 23,   # Also has unused id #34.
    'Entertainment': 24,
    'News & Politics': 25,
    'Howto & Style': 26,
    'Education': 27,
    'Science & Technology': 28,
    'Nonprofits & Activism': 29,
    'Movies': 30,
    'Anime/Animation': 31,
    'Action/Adventure': 32,
    'Classics': 33,
    'Documentary': 35,
    'Drama': 36,
    'Family': 37,
    'Foreign': 38,
    'Horror': 39,
    'Sci-Fi/Fantasy': 40,
    'Thriller': 41,
    'Shorts': 42,
    'Shows': 43,
    'Trailers': 44
}


# Identify default data pipeline parameters for when none are given.
DEFAULT_DATA_PARAMS = {
    'col_labels': 'category_id',
    'col_corpus': ['title', 'tags', 'description', 'caption'],
    'splits': [0.4, 0.4, 0.2],
    'ngram_size': 1,
    'vocab_size': 25000,
    'batch_size': 64,
    'cbow': False,
}


def get_data():
    return pd.merge(
        pd.read_csv(PATH_VIDEOS),
        pd.read_table(PATH_CAPTIONS, names=['video_id', 'caption']),
        on='video_id'
    ).drop_duplicates('video_id', 'first', ignore_index=True)


def get_column_indices(
    data: pd.DataFrame,
    col_labels: str,
    col_corpus: List[str],
) -> Tuple[int, List[int]]:
    '''
    Identify the indices of the labels and text features.

    data (pd.DataFrame): raw data from which to extract labels and corpus.
    col_labels (str): name of the column that has the label.
    col_corpus (list): names of the columns that have the text features.

    Return label index and corpus indices (int, list of ints).
    '''
    idx_labels = data.columns.get_loc(col_labels)
    idx_corpus = [data.columns.get_loc(column) for column in col_corpus]
    return idx_labels, idx_corpus


def get_label_values(
    data: pd.DataFrame,
    label_lookup: Dict[str, int],
    label_target: str,
    label_others: List[str],
) -> Tuple[int, List[int]]:
    '''
    Identify the values of the target label and other labels to keep.

    data (pd.DataFrame): raw data from which to extract labels and corpus.
    label_target (str): name of the positive class.
    label_others (list): names of the other classes to include for modeling.

    Return target label and other label values (int, list of ints).
    '''
    label_target = label_lookup[label_target]
    label_others = [label_lookup[label] for label in label_others]
    return label_target, label_others


def run_data_pipeline(
    data: pd.DataFrame,
    col_labels: str,
    col_corpus: List[str],
    label_target: str,
    label_others: List[str],
    splits: Tuple[float, float, float],
    ngram_size: int,
    vocab_size: int,
    batch_size: int,
    cbow: bool
) -> Tuple[Tuple[Dict[str, int], Dict[int, str]], List[List[dp.TorchTextLike]]]:
    '''
    Run the complete data pipeline, bringing the raw data to the neural network
    models. Build a vocabulary and training, validation, and testing datasets
    from the labels and corpus in the data.

    data (pd.DataFrame): raw data from which to extract labels and corpus.
    col_labels (str): name of the column that has the label.
    col_corpus (list): names of the columns that have the text features.
    label_target (str): name of the positive class.
    label_others (list): names of the other classes to include for modeling.
    splits (tuple of floats): proportional allocations of each respective set.
    ngram_size (int): length of n-gram, i.e. number of words in one token.
    vocab_size (int): number of most frequent words to keep.
    batch_size (int): number of sentences with each batch.
    cbow (bool): whether to represent sentences as continuous bags of words.

    Return vocab (tuple of dicts), batches (list of lists of TorchTextLike).
    '''
    # Identify the indices of the labels and text features.
    idx_labels, idx_corpus = get_column_indices(data, col_labels, col_corpus)
    # Identify the values of the target label and other labels to keep.
    label_target, label_others = get_label_values(
        data, LABEL_TO_IDX, label_target, label_others)
    # Get the labels and corpus from the data.
    labels, corpus = dp.get_labels_and_corpus(
        data, idx_labels, idx_corpus, label_target, label_others, silent=True)
    # Return the vocabulary and train, valid, and test set batches.
    return dp.get_vocab_and_batches(
        labels, corpus, splits, ngram_size, vocab_size, batch_size, cbow, silent=True)


def get_models(model_kwargs: Dict[str, dict]) -> Dict[str, Type[nn.Module]]:
    '''
    Initialize empty models by name from parameterizations.

    model_kwargs (dict): mapping of model class names to parameterizations.

    Return mapping of model names to model objects (dict).
    '''
    return {
        model: getattr(nnmodels, model)(**kwargs)
        for model, kwargs in model_kwargs.items()
    }


def run_modeling_pipeline(
    word_to_idx: Dict[str, int],
    idx_to_word: Dict[int, str],
    itr_train: List[dp.TorchTextLike],
    itr_valid: List[dp.TorchTextLike],
    itr_testg: List[dp.TorchTextLike],
    model_kwargs: Dict[str, dict]
) -> Dict[str, dict]:
    '''
    Run the complete modeling pipeline, training and evaluating language models.
    Build models and report the performance of each on the data. The metrics
    dictionary has the results of each validation epoch and testing evaluation:

    metrics = {
        'model_name': {
            0: {'loss': 0., 'accuracy': 0., 'precision': 0., 'recall': 0.},
            1: {'loss': 0., 'accuracy': 0., 'precision': 0., 'recall': 0.},
            'test': {'loss': 0., 'accuracy': 0., 'precision': 0., 'recall': 0.},
            'best': nn.Module.state_dict()
        }
    }

    word_to_idx, idx_to_word (dict): mappings of words the vocabulary.
    itr_train, itr_valid, itr_testg (list of TorchTextLike): dataset batches.
    model_kwargs (dict): mappings of model classes to parameterizations.

    Return model metrics (dict).
    '''
    # Update parameters with knowledge from the data.
    add = {'input_dim': len(word_to_idx), 'pad_idx': word_to_idx[dp.PAD]}
    for model in model_kwargs:
        model_kwargs[model].update(add)
    # Get the models per their parameterizations.
    models = get_models(model_kwargs)
    # Train and evaluate the models on training and validation datasets.
    metrics = {}
    for name, model in models.items():
        tm = train_eval.Training_module(model, silent=True)
        metrics[name] = tm.train_model(itr_train, itr_valid)
    # Evaluate the best models on the testing dataset.
    for name, model in models.items():
        tm = train_eval.Training_module(model, silent=True)
        tm.model.load_state_dict(metrics[name]['best'])
        metrics[name]['test'] = tm.evaluate(itr_testg)
    # Return the metrics dictionary.
    return metrics


def read_config(filename: str) -> List[Dict[str, Any]]:
    '''
    Read the configuration from file and update each experiment specified
    with any unset defaults.

    filename (str): location at which to read the configuration.

    Return experiemnts (list of dicts).
    '''
    experiments = []
    # Open the file at the given filename.
    with open(filename) as f:
        # Consider each document in the YAML file.
        for config in yaml.safe_load_all(f):
            # Skip blank documents.
            if config is None:
                continue
            # Update data parameters with any unset defaults.
            for param, setting in DEFAULT_DATA_PARAMS.items():
                if param not in config['data']:
                    config['data'][param] = setting
            # Include this experiment with all others.
            experiments.append(config)
    # Return the experiment configurations.
    return experiments


def dump_results(filename: str, results: Dict[str, Any]) -> None:
    '''
    Serialize the results and write to disk.

    filename (str): location at which to write the results.
    results (dict): metrics and models from experiments.
    '''
    # Ensure intermediate directories exist.
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        print(directory)
        os.makedirs(directory)
    # Open the file at the given filename.
    with open(filename, mode='wb') as f:
        # Serialize the results into the file.
        pickle.dump(results, f)


def run(args):
    # Read the config file for experiments to run.
    experiments = read_config(args.config)
    # Read the data.
    raw_data = get_data()
    # Run each experiment.
    for params in experiments:
        name = params.pop('name')
        # Run the data pipeline.
        vocab, batches = run_data_pipeline(raw_data, **params['data'])
        # Run the modeling pipeline.
        metrics = run_modeling_pipeline(*vocab, *batches, params['models'])
        # Write these metrics to file.
        dump_results(args.out + '/' + name, metrics)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description='Run YouTube video and caption data language models.')
    parser.add_argument(
        '--config',
        required=True,
        help='Identify the YAML file that specifies the experiments.',
        dest='config')
    parser.add_argument(
        '--out',
        required=True,
        help='Identify directory in which to pickle the model results.',
        dest='out')
    run(parser.parse_args())
