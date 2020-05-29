from typing import Type, Generator, Iterable, Tuple, List, Dict, Set, Any
import re
from sklearn.feature_extraction import stop_words
from collections import Counter

import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim

label_to_ix = {0.0: 0, 1.0: 1}


# Identify constants for vocabulary operations.
DELIMITERS = re.compile(r' |\\n|\|')
PUNCTUATION = re.compile(r'[.:;,?!\"|#()-_•]|^\'|\'$')
STOP_WORDS = stop_words.ENGLISH_STOP_WORDS
STOP_PATTERN = re.compile(r'http|www|^\s*$')
UNK = '<unk>'
PAD = '<pad>'


def get_ngrams(
    corpus: np.ndarray,
    n: int,
    k: int = 10000,
    tokenize_pattern: re.Pattern = DELIMITERS,
    clean_pattern: re.Pattern = PUNCTUATION,
    stop_words: Set[str] = STOP_WORDS,
    drop_pattern: re.Pattern = STOP_PATTERN
) -> Dict[str, int]:
    '''
    Collect the indices of the k most frequent n-grams in the vocabulary.

    corpus (np.ndarray): n x 1 array of text to parse.
    n (int): length of n-gram, i.e. number of words to represent with one token.
    k (int): number of most frequent words to keep.
    tokenize_pattern (re.Pattern): regular expression for delimiters.
    clean_pattern (re.Pattern): regular expression for substrings to remove.
    stop_words (set): words to remove, i.e. articles, prepositions.
    drop_pattern (re.Pattern): regular expression for words to remove.

    Return mapping of words to indices (dict).
    '''
    # Collect the vocabulary in the corpus.
    vocab = __make_vocab(
        corpus, n, tokenize_pattern, clean_pattern, stop_words, drop_pattern)
    # Return the mapping of the k most frequent words to their indices.
    return __make_index(vocab, k)


def get_bag_of_ngrams(
    vocab: Dict[str, int],
    sentence: str,
    n: int,
    tokenize_pattern: re.Pattern = DELIMITERS,
    clean_pattern: re.Pattern = PUNCTUATION,
    stop_words: Set[str] = STOP_WORDS,
    drop_pattern: re.Pattern = STOP_PATTERN
) -> torch.FloatTensor:
    '''
    Map n-grams in a sentence to their indices in the vocabulary.

    vocab (dict): mapping of words to indices.
    sentence (str): collection of words to process.
    n (int): length of n-gram, i.e. number of words to represent with one token.
    tokenize_pattern (re.Pattern): regular expression for delimiters.
    clean_pattern (re.Pattern): regular expression for substrings to remove.
    stop_words (set): words to remove, i.e. articles, prepositions.
    drop_pattern (re.Pattern): regular expression for words to remove.

    Return the bag of n-grams (torch.FloatTensor)
    '''
    # Numericize the sentence.
    sentence = __numericize_sentence(
        vocab, sentence, n, tokenize_pattern, clean_pattern, stop_words, drop_pattern)
    # Count each n-gram in the vocabulary that appears in the sentence.
    bag_of_ngrams = torch.zeros(len(vocab))
    for idx in sentence:
        bag_of_ngrams[idx] += 1
    # Return the bag of n-grams.
    return bag_of_ngrams


def __make_vocab(
    corpus: np.ndarray,
    n: int,
    tokenize_pattern: re.Pattern,
    clean_pattern: re.Pattern,
    stop_words: Set[str],
    drop_pattern: re.Pattern
) -> Dict[str, int]:
    '''
    Collect the vocabulary in the corpus and tally the frequency of each word.
    Identify words on the delimiter and clean them with a regular expression.

    corpus (np.ndarray): n x 1 array of text to parse.
    n (int): length of n-gram, i.e. number of words to represent with one token.
    tokenize_pattern (re.Pattern): regular expression for delimiters.
    clean_pattern (re.Pattern): regular expression for substrings to remove.
    stop_words (set): words to remove, i.e. articles, prepositions.
    drop_pattern (re.Pattern): regular expression for words to remove.

    Return mapping of words to frequencies (dict).
    '''
    vocab = {}
    # Consider each sentence in the corpus.
    for sentence in corpus:
        # Preprocess the sentence, i.e. tokenize, clean, drop, pad.
        sentence = __make_sentence(
            sentence, n, tokenize_pattern, clean_pattern, stop_words, drop_pattern)
        # Consider each word, or possibly n-gram, in the sentence.
        for w in range(len(sentence) - n):
            # Collect this word or n-gram.
            word = ' '.join(sentence[w:w + n])
            # Update the frequency of this word or n-gram in the vocabulary.
            if word not in vocab:
                vocab[word] = 0
            vocab[word] += 1
    # Return the mapping of words to frequencies.
    return vocab


def __make_sentence(
    sentence: str,
    n: int,
    tokenize_pattern: re.Pattern,
    clean_pattern: re.Pattern,
    stop_words: Set[str],
    drop_pattern: re.Pattern
) -> List[str]:
    '''
    Preprocess the sentence by tokenizing, cleaning, dropping stop words, and
    including any padding.

    sentence (str): collection of words to process.
    n (int): length of n-gram, i.e. number of words to represent with one token.
    tokenize_pattern (re.Pattern): regular expression for delimiters.
    clean_pattern (re.Pattern): regular expression for substrings to remove.
    stop_words (set): words to remove, i.e. articles, prepositions.
    drop_pattern (re.Pattern): regular expression for words to remove.

    Return processed sentence (list).
    '''
    # Tokenize the sentence on the delimiter.
    sentence = __tokenize_words(sentence, tokenize_pattern)
    # Clean words in the sentence.
    sentence = [__clean_words(word, clean_pattern) for word in sentence]
    # Drop any stop words in the sentence.
    sentence = __drop_words(sentence, stop_words, drop_pattern)
    # Pad both sides of the sentence.
    sentence = [PAD] * (n - 1) + sentence + [PAD] * (n - 1)
    # Return the sentence.
    return sentence


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


def __drop_words(
    words: List[str],
    stop_words: Set[str],
    pattern: re.Pattern
) -> List[str]:
    '''
    Remove words that are among the stop words or match a regular expression.

    words (list): collection of words, i.e. a sentence.
    stop_words (set): words to remove.
    pattern (re.Pattern): regular expression for words to remove.

    Return words (list).
    '''
    return [
        word for word in words
        if word not in stop_words and not re.search(pattern, word)
    ]


def __make_index(vocab: Dict[str, int], n: int) -> Dict[str, int]:
    '''
    Collect indices for the most frequent words in the vocabulary. Include
    tokens for unknown words and padding.

    vocab (dict): mapping of words to frequencies.
    n (int): number of most frequent words to keep.

    Return mapping of most frequent words to indices (dict).
    '''
    # Collect most the frequent words into a set.
    vocab = {word for word, _ in Counter(vocab).most_common(n)}
    # Include tokens for unknown words and padding.
    vocab.update([UNK, PAD])
    # Return the mapping of words to indices.
    return {word: i for i, word in enumerate(vocab)}


def __numericize_sentence(
    vocab: Dict[str, int],
    sentence: str,
    n: int,
    tokenize_pattern: re.Pattern,
    clean_pattern: re.Pattern,
    stop_words: Set[str],
    drop_pattern: re.Pattern
) -> List[int]:
    '''
    Represent each word in the sentence with its index in the vocabulary.

    vocab (dict): mapping of words to indices.
    sentence (str): collection of words to process.
    n (int): length of n-gram, i.e. number of words to represent with one token.
    tokenize_pattern (re.Pattern): regular expression for delimiters.
    clean_pattern (re.Pattern): regular expression for substrings to remove.
    stop_words (set): words to remove, i.e. articles, prepositions.
    drop_pattern (re.Pattern): regular expression for words to remove.

    Return the numericized sentence (list).
    '''
    new_sentence = []
    # Preprocess the sentence, i.e. tokenize, clean, drop, pad.
    sentence = __make_sentence(
        sentence, n, tokenize_pattern, clean_pattern, stop_words, drop_pattern)
    # Consider each word, or possibly n-gram, in the sentence.
    for w in range(len(sentence) - n):
        # Collect this word or n-gram.
        word = ' '.join(sentence[w:w + n])
        # Add the index of this word or n-gram to the new sentence.
        new_sentence.append(vocab.get(word, vocab[UNK]))
    # Return the numericized sentence.
    return new_sentence


def make_target(label, label_to_ix):
    '''
    Helper function for label tensor creation
    '''
    return torch.LongTensor([label_to_ix[label]])


def run_bow_ngram(model, word_to_idx, wtorch, X_train, y_train, 
                  X_valid, y_valid, X_test, y_test, n):
    '''
    The function is used to run the neural network classifier
    with ngrams and bag of words setting
    Inputs:
        model: model objects for neural network
        word_to_idx: word to indices dictionary 
        wtorch: bag of words tensor
        X_train, X_train, y_train, X_valid, y_valid, X_test, y_test
        n: n for ngrams
    Returns: the performance dictionary
    '''
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    
    best_acc = 0
    perf_dict = {}
    for epoch in range(5):
    
        for idx in range(X_train.shape[0]):
        
            model.zero_grad()
        
            # Complete forward propagation.
            x = get_bag_of_ngrams(word_to_idx, X_train[idx], n)
            y_hat = model(x.view(1, -1))
            # Calculate loss.
            y = make_target(y_train[idx, 0], label_to_ix)
            loss = loss_function(y_hat, y)
            # Complete backward propagation.
            loss.backward()
            optimizer.step()
    
        acc_count = 0
        with torch.no_grad():
            for idx in range(X_valid.shape[0]):
                # Complete forward propagation.
                x = get_bag_of_ngrams(word_to_idx, X_valid[idx], n)
                y_hat = model(x.view(1, -1))
                # Increment the count of correct predictions.
                y_pred = np.argmax(y_hat[0].detach().numpy())
                if y_valid[idx, 0] == y_pred:
                    acc_count += 1

        print("For epoch number ", epoch, ", the accuracy for validation set is ", 
              acc_count / X_valid.shape[0])
    
        if (acc_count / X_valid.shape[0]) > best_acc:
            best_model = model
        
        perf_dict["valid_epoch_acc " + str(epoch)] = acc_count / X_valid.shape[0]

    acc_count = 0
    yreal_count = 0
    ypred_count = 0
    ypred1_count = 0
    with torch.no_grad():
        for idx in range(X_test.shape[0]):
            # Complete forward propagation.
            x = get_bag_of_ngrams(word_to_idx, X_test[idx], n)
            y_hat = model(x.view(1, -1))
            # Increment the count of correct predictions.
            y_pred = np.argmax(y_hat[0].detach().numpy())

            if y_test[idx, 0] == y_pred:
                acc_count += 1
        
            yreal_count += y_test[idx, 0]
            ypred_count += y_pred
        
            if (y_test[idx, 0] == y_pred) and (y_pred == 1):
                ypred1_count += 1

    print("the accuracy for test set is ", acc_count / X_test.shape[0])
    print("the presision for test set is ", ypred1_count / ypred_count)
    print("the recall for test set is ", ypred1_count / yreal_count)

    perf_dict["best_test_acc"] = acc_count / X_test.shape[0]
    perf_dict["best_test_precision"] = ypred1_count / ypred_count
    perf_dict["best_test_recall"] = ypred1_count / yreal_count

    return perf_dict