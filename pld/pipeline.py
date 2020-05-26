import os
from typing import Type, Generator, Tuple, List, Dict
import csv
import copy
import re
from collections import Counter
import numpy as np


# Identify constants for vocabulary operations.
DELIMITERS = re.compile(r' |\\n|\|')
PUNCTUATION = re.compile(r'[.:;,?!\"|#()-_•]|^\'|\'$')
STOP_PATTERN = re.compile(r'http|www|^\s*$')
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
    **kwargs (dict): keyword arguments to pass to __read_data().

    Return dataset (np.ndarray).
    '''
    # Initialize a container for the dataset.
    dataset = []
    # Get a generator that reads the data from the file.
    rows = __read_data(filename, **kwargs)
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


def __read_data(
    filename: str,
    delimiter: str = ',',
    has_header: bool = True
) -> Generator[List[str], None, None]:
    '''
    Read the data in a file and tokenize its records on the field delimiter.

    filename (str): location of the data.
    delimiter (str): field separator, default is comma ','.
    has_header (bool): whether the data has a header row, default is True.

    Return tokenized rows in the data (generator of lists of strings).
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

    Return splits (list of np.ndarray).
    '''
    # Ensure the proportions sum to one.
    assert sum([train, valid, test]) == 1
    # Allocate a random permutation of row indices proportionally.
    n = data.shape[0]
    idx = np.random.permutation(n)
    idx_train = idx[:int(n * train)]
    idx_valid = idx[int(n * train):int(n * (train + valid))]
    idx_testg = idx[int(n * (train + valid)):]
    # Return the subsets of data.
    return data[idx_train, ], data[idx_valid, ], data[idx_testg, ]


def make_vectorization(
    corpus: np.ndarray,
    n: int,
    tokenize_pattern: re.Pattern = DELIMITERS,
    clean_pattern: re.Pattern = PUNCTUATION,
    stop_words: set = STOP_WORDS,
    drop_pattern: re.Pattern = STOP_PATTERN
) -> Tuple[List[np.ndarray], Dict[str, int]]:
    '''
    Collect the vector representation of each sentence in the corpus from the
    n most frequent words in its vocabulary.

    corpus (np.ndarray): n x 1 array of text to parse.
    n (int): size of the vocabulary chosen from most frequent words.
    tokenize_pattern (re.Pattern): regular expression for delimiters.
    clean_pattern (re.Pattern): regular expression for substrings to remove.
    stop_words (set): words to remove, i.e. articles, prepositions.
    drop_pattern (re.Pattern): regular expression for words to remove.

    Return the vectorized corpus and vocabulary (list, dict).
    '''
    # Make the vocabulary and clean the corpus.
    corpus, vocab = __make_vocab(corpus, tokenize_pattern, clean_pattern)
    # Drop stop words and keep only the n most frequent words.
    vocab = __drop_words(vocab, n, stop_words, drop_pattern)
    # Map words in the vocabulary to indices with entries for padding, unknown.
    vocab = __make_index(vocab)
    # Map each word in the corpus to its index in the vocabulary.
    corpus = __numericize_words(corpus, vocab)
    # Return the vectorized corpus and its vocabulary.
    return corpus, vocab


def __make_vocab(
    corpus: np.ndarray,
    tokenize_pattern: re.Pattern,
    clean_pattern: re.Pattern
) -> Tuple[np.ndarray, dict]:
    '''
    Collect the vocabulary in the corpus and tally the frequency of each word.
    Identify words on the delimiter and clean them with a regular expression.

    corpus (np.ndarray): n x 1 array of text to parse.
    tokenize_pattern (re.Pattern): regular expression for delimiters.
    clean_pattern (re.Pattern): regular expression for substrings to remove.

    Return cleaned corpus, mapping of words to indices (np.ndarray, dict).
    '''
    vocab = {}
    # Make a deep copy of the corpus because it arrives to scope by reference.
    corpus = copy.deepcopy(corpus)
    # Consider each sentence in the corpus.
    for s in range(corpus.shape[0]):
        sentence = __tokenize_words(corpus[s], tokenize_pattern)
        # Consider each word in the sentence.
        for w in range(len(sentence)):
            # Clean this word.
            word = __clean_words(sentence[w], clean_pattern)
            # Update the frequency of this word in the vocabulary.
            if word not in vocab:
                vocab[word] = 0
            vocab[word] += 1
            # Update this word in the sentence.
            sentence[w] = word
        # Update this sentence in the corpus.
        corpus[s] = ' '.join(word for word in sentence if word != '')
    # Return the updated corpus and the vocabulary.
    return corpus, vocab


def __tokenize_words(text: str, pattern: re.Pattern) -> List[str]:
    '''
    Tokenize text on the delimiter.

    text (str): collection of characters to tokenize.
    pattern (re.Pattern): regular expression for delimiters.

    Return tokenized text (list of strings).
    '''
    return re.split(pattern, text)


def __clean_words(text: str, pattern: re.Pattern) -> str:
    '''
    Clean text of substrings.

    text (str): collection of characters to clean.
    pattern (re.Pattern): regular expression for substrings to remove.

    Return cleaned text (str).
    '''
    return re.sub(pattern, '', text.lower())


def __drop_words(vocab: dict, n: int, stop_words: set, pattern: re.Pattern) -> set:
    '''
    Remove words from the vocabulary that are among the stop words or match a
    regular expression and keep only the n most frequent.

    vocab (dict): mapping of words to frequencies.
    n (int): the number of most frequent words to keep.
    stop_words (set): words to remove.
    pattern (re.Pattern): regular expression for words to remove.

    Return vocabulary (set).
    '''
    # Find words that match the stop pattern.
    for word in vocab:
        if re.search(pattern, word):
            stop_words.add(word)
    # Remove stop words from the vocabulary.
    for word in stop_words:
        del vocab[word]
    # Return the most frequent words.
    return {word for word, _ in Counter(vocab).most_common(n)}


def __make_index(vocab: set) -> Dict[str, int]:
    '''
    Collect indices for each word in the vocabulary. Include tokens for unknown
    words and padding.

    vocab (set): words in the vocabulary.

    Return mapping of words to indices (dict).
    '''
    vocab.update([UNK, PAD])
    return {word: i for i, word in enumerate(vocab)}


def __numericize_words(
    corpus: np.ndarray,
    vocab: dict,
    tokenize_pattern: re.Pattern = DELIMITERS,
) -> List[np.ndarray]:
    '''
    Represent each word in the corpus with its index in the vocabulary.

    corpus (np.ndarray): n x 1 array of text to parse.
    vocab (dict): mapping of words to indices.
    tokenize_pattern (re.Pattern): regular expression for delimiters.

    Return the numericized corpus (list).
    '''
    data = []
    # Consider each sentence in the corpus.
    for s in range(corpus.shape[0]):
        words = __tokenize_words(corpus[s], tokenize_pattern)
        # Map each word in the sentence to its index in the vocabulary.
        sentence = np.array([vocab.get(word, UNK) for word in words])
        # Add this sentence to the data.
        data.append(sentence)
    # Return the numericized corpus.
    return data


def make_batches(
    labels: np.ndarray,
    corpus: List[np.ndarray],
    default: int,
    n: int
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    '''
    Generate label, data pairs of the data of length n. Pad sentences with a
    default value such that each in the same batch is of the same length.

    Return paired labels and data (generator of tuples of np.ndarray).
    '''
    # Verify the size of the labels and corpus match.
    assert labels.shape[0] == corpus.shape[0]
    # Calculate the number of batches to generate.
    batches = (labels.shape[0] + n - 1) // n
    # Generate each batch.
    for i in range(batches):
        idx = i * n
        # Initialize this batch with the length of its longest sentence.
        m = max(corpus[idx + j].shape[0] for j in range(n))
        batch = np.full(shape=(n, m), fill_value=default)
        # Load each sentence in this batch.
        for j in range(n):
            batch[j, :corpus[idx + j].shape[0]] = corpus[idx + j]
        # Yield this batch.
        yield labels[idx:idx + n], batch

