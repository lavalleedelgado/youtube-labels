import numpy as np
import torch
import copy
import torch.nn as nn
import torch.optim as optim

label_to_ix = {0.0: 0, 1.0: 1}


def make_target(label, label_to_ix):
    '''
    Helper function for label tensor creation
    '''
    return torch.LongTensor([label_to_ix[label]])


def run_bow_ngram(model, itr_train, itr_valid, itr_test, n):
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
    
        for batch in itr_train:
        
            model.zero_grad()
        
            # Complete forward propagation.
            y_hat = model(batch.text)
            # Calculate loss.
            y = make_target(batch.label, label_to_ix)
            loss = loss_function(y_hat, y)
            # Complete backward propagation.
            loss.backward()
            optimizer.step()
    
        acc_count = 0
        with torch.no_grad():
            for batch in itr_valid:
                # Complete forward propagation.
                y_hat = model(batch.text)
                # Increment the count of correct predictions.
                y_pred = np.argmax(y_hat[0].detach().numpy())
                if batch.label == y_pred:
                    acc_count += 1

        print("For epoch number ", epoch, ", the accuracy for validation set is ", 
              acc_count / itr_valid.shape[0])
    
        if (acc_count / itr_valid.shape[0]) > best_acc:
            best_model = model
        
        perf_dict["valid_epoch_acc " + str(epoch)] = acc_count / itr_valid.shape[0]

    acc_count = 0
    yreal_count = 0
    ypred_count = 0
    ypred1_count = 0
    with torch.no_grad():
        for idx in range(X_test.shape[0]):
            # Complete forward propagation.
            y_hat = model(batch.text)
            # Increment the count of correct predictions.
            y_pred = np.argmax(y_hat[0].detach().numpy())

            if batch.label == y_pred:
                acc_count += 1
        
            yreal_count += batch.label
            ypred_count += y_pred
        
            if (batch.label == y_pred) and (y_pred == 1):
                ypred1_count += 1

    print("the accuracy for test set is ", acc_count / X_test.shape[0])
    print("the presision for test set is ", ypred1_count / ypred_count)
    print("the recall for test set is ", ypred1_count / yreal_count)

    perf_dict["best_test_acc"] = acc_count / X_test.shape[0]
    perf_dict["best_test_precision"] = ypred1_count / ypred_count
    perf_dict["best_test_recall"] = ypred1_count / yreal_count

    return perf_dict