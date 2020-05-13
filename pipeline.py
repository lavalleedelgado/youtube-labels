import os
from typing import Generator, List, Tuple
import csv
import re
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# Identify constants for vocabulary operations.
DELIMITERS = re.compile(r' |\\n|\|')
PUNCTUATION = re.compile(r'[.:;,?!\"|#()-]|^\'|\'$')
STOP_PATTERN = re.compile(r'http|www')
STOP_WORDS = set(['the', 'a', 'and', 'or', 'of'])
UNK = '<unk>'
PAD = '<pad>'


def make_dataset(
    filename: str,
    idx_label: int = None,
    idx_text: tuple = None,
    idx_num: tuple = None,
    binarize_label: str = None,
    **kwargs
) -> np.ndarray:
    '''
    Collect the features and label from each row in the data by column index.
    Concatenate text features. Binarize labels on the given positive class. The
    order of the result is label, text feature, numeric features.

    filename (str): location of the data.
    idx_label (int): column index of the label.
    idx_text (tuple): column indices of text data to concatenate.
    idx_num (tuple): column indices of numeric data.
    binarize_label (str): positive class in the label.
    **kwargs (dict): keyword arguments to pass to read_data().

    Return dataset np.ndarray.
    '''
    # Initialize a container for the dataset.
    dataset = []
    # Get a generator that reads the data from the file.
    rows = read_data(filename, **kwargs)
    # Consider each row in the data.
    while True:
        row = next(rows, False)
        if not row:
            break
        r = []
        # Collect the label and binarize if requested.
        if idx_label is not None:
            label = row[idx_label]
            if binarize_label is not None:
                label = 1 if label == binarize_label else 0
            r.append(label)
        # Concatenate the text features indicated.
        if idx_text is not None:
            if len(idx_text) > 1:
                text = ' '.join(row[idx] for idx in idx_text)
            else:
                text = row[idx_text[0]]
            r.append(text)
        # Collect the numeric features indicated.
        if idx_num is not None:
            num = [row[idx] for idx in idx_num]
            r.append(num)
        # Append this row to the dataset.
        dataset.append(np.array(r))
    # Return the dataset.
    return np.vstack(dataset)


def read_data(
    filename: str,
    delimiter: str = ',',
    has_header: bool = True
) -> Generator[List[str], None, None]:
    '''
    Read the data in a file and tokenize its records on the field delimiter.

    filename (str): location of the data.
    delimiter (str): field separator, default is comma ','.
    has_header (bool): whether the data has a header row, default is True.

    Return tokenized rows in the data (Generator of lists of strings).
    '''
    # Verify the file exists.
    if not os.path.exists(filename):
        raise FileNotFoundError
    # Read the file.
    with open(filename, mode='r') as f:
        data = csv.reader(f, delimiter=delimiter)
        # Skip the header row if indicated.
        if has_header:
            next(data)
        # Yield each row in the data
        for row in data:
            yield row


def split_data(data: np.ndarray, train: float, valid: float, test: float) -> List[np.ndarray]:
    '''
    Split the data proportionally into training, validation, and testing sets.

    train, valid, test (float): proportional allocations of the data.

    Return splits (list of np.ndarray objects).
    '''
    # Ensure the proportions sum to one.
    assert sum([train, valid, test]) == 1
    # Allocate a random permutation of row indices proportionally.
    n = data.shape[0]
    idx = np.random.permutation(n)
    idx_train = idx[:int(n * train)]
    idx_valid = idx[len(idx_train):len(idx_train) + int(n * valid)]
    idx_testg = idx[len(idx_train) + len(idx_valid):]
    # Return the subsets of data.
    return data[idx_train, ], data[idx_valid, ], data[idx_testg, ]


def make_vocabulary(
    corpus: np.ndarray,
    n: int,
    tokenize_pattern: re.Pattern = DELIMITERS,
    clean_pattern: re.Pattern = PUNCTUATION,
    stop_words: set = STOP_WORDS,
    drop_pattern: re.Pattern = STOP_PATTERN,
) -> Tuple[List[tuple], dict]:
    '''
    Collect the vocabulary in the corpus and tally the frequency of each word.
    Identify words on the delimiter and clean them with a regular expression.

    corpus (np.ndarray): n x 1 array of text to parse.
    n (int): size of the vocabulary chosen from most frequent words.
    tokenize_pattern (re.Pattern): regular expression for delimiters.
    clean_pattern (re.Pattern): regular expression for substrings to remove.
    stop_words (set): words to remove, i.e. articles, prepositions.
    drop_pattern (re.Pattern): regular expression for words to remove.

    Return cleaned corpus, mapping of words to indices (list, dict).
    '''
    vocabulary = {}
    # Consider each sentence in the corpus.
    for s in range(corpus.shape[0]):
        sentence = tokenize_words(corpus[s], tokenize_pattern)
        # Consider each word in the sentence.
        for w in range(len(sentence)):
            # Clean this word.
            word = clean_words(sentence[w], clean_pattern)
            # Update the frequency of this word in the vocabulary.
            if word not in vocabulary:
                vocabulary[word] = 0
            vocabulary[word] += 1
            # Update this word in the sentence.
            sentence[w] = word
        # Update this sentence in the corpus.
        corpus[s] = ' '.join(word for word in sentence if word != '')
    # Drop stop words and keep only the n most frequent words.
    vocabulary = drop_words(vocabulary, n, stop_words, drop_pattern)
    # Map words in the vocabulary to indices with entries for padding, unknown.
    vocabulary = make_index(vocabulary)
    # Return the updated corpus and the vocabulary.
    return corpus, vocabulary


def tokenize_words(text: str, pattern: re.Pattern) -> List[str]:
    '''
    Tokenize text on the delimiter.

    text (str): collection of characters to tokenize.
    pattern (re.Pattern): regular expression for delimiters.

    Return tokenized text (list of strings).
    '''
    return re.split(pattern, text)


def clean_words(text: str, pattern: re.Pattern) -> str:
    '''
    Clean text of substrings.

    text (str): collection of characters to clean.
    pattern (re.Pattern): regular expression for substrings to remove.

    Return cleaned text (string).
    '''
    return re.sub(pattern, '', text).lower()


def drop_words(vocabulary: dict, n: int, stop_words: set, pattern: re.Pattern) -> set:
    '''
    Remove words from the vocabulary that are among the stop words or match a
    regular expression and keep only the n most frequent.

    vocabulary (dict): mapping of words to frequencies.
    n (int): the number of most frequent words to keep.
    stop_words (set): words to remove.
    pattern (re.Pattern): regular expression for words to remove.

    Return vocabulary (set).
    '''
    # Find words that match the stop pattern.
    for word in vocabulary:
        if re.search(pattern, word):
            stop_words.add(word)
    # Remove stop words from the vocabulary.
    for word in stop_words:
        del vocabulary[word]
    # Return the most frequent words.
    return {word for word, _ in Counter(vocabulary).most_common(n)}


def make_index(vocabulary: set) -> dict:
    '''
    Collect indices for each word in the vocabulary. Include tokens for unknown
    words and padding.

    vocabulary (set): words in the vocabulary.

    Return mapping of words to indices (dict).
    '''
    vocabulary.update([UNK, PAD])
    return {word: i for i, word in enumerate(vocabulary)}


def make_vectorization(
    corpus: np.ndarray,
    vocabulary: dict,
    tokenize_pattern: re.Pattern = DELIMITERS
) -> np.ndarray:
    '''
    Represent the corpus as an n x m matrix that encodes each of n sentences
    as a vector with the frequencies of each of m words in the vocabulary.

    corpus (np.ndarray): n x 1 array of text to vectorize.
    vocabulary (dict): mapping of m words to indices.
    tokenize_pattern (re.Pattern): regular expression for delimiters.

    Return n x m matrix representation of the corpus (np.ndarray).
    '''
    # Initialize an n x m array for the data.
    n = corpus.shape[0]
    m = len(vocabulary)
    data = np.zeros(shape=(n, m))
    # Get the length of the longest sentence.
    k = max(len(sentence) for sentence in corpus)
    # Get the index of the unknown word.
    idx_unk = vocabulary[UNK]
    # Consider each sentence in the corpus.
    for s in range(n):
        sentence = tokenize_words(corpus[s], tokenize_pattern)
        # Pad this sentence to the length of the longest.
        sentence += [PAD] * (k - len(sentence))
        # Map the index of each word to its frequency in the sentence.
        words = Counter(vocabulary.get(word, idx_unk) for word in sentence)
        # Update the matrix representation.
        for w, frequency in words.items():
            data[s, w] = frequency
    # Return the vectorized corpus.
    return data

