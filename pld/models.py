from typing import Type, Generator, List, Tuple, Dict
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tud


class YouTubeBagOfWords(nn.Module):
    
    def __init__(self, d_in, d_out):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out)


    def forward(self, x):
        return self.linear(x)


class YouTubeWordEmbeddings(nn.Module):
    
    def __init__(self, d_in, d_e, d_h, d_out):
        super().__init__()
        self.embedding = nn.Embedding(d_in, d_e)
        self.linear1 = nn.Linear(d_e, d_h)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_h, d_out)


    def forward(self, x):
        layer_e = self.embedding(x)
        layer_h = self.relu(self.linear1(layer_e))
        return self.linear2(layer_h)


class YouTubeModelTrainer():

    def __init__(self, model: Type[nn.Module], loss):
        self.model = model
        self.loss = loss
        self.optimizer = optim.Adam(model.parameters(), lr=0.1)


    def __train_epoch(self, train_data: np.ndarray, batch_size: int = 5) -> Type[nn.Module]:
        '''
        Train the model with one pass of the training data.

        train_data (np.ndarray): training data.
        batch_size (int): number of observations with which to train at a time.

        Return the trained model (nn.Module).
        '''
        # Set the module to training mode.
        self.train()
        # Consider each batch of examples in the training data.
        data = tud.DataLoader(train_data, batch_size=batch_size)
        for x, y in data:
            # Clear the gradients from any previous optimization.
            self.optimizer.zero_grad()
            # Forward pass: get predictions from the model.
            y_hat = self.model(x).squeeze(1)
            l = self.loss(y_hat, y)
            # Backward pass: compute gradients and update model parameters.
            l.backward()
            self.optimizer.step()
        # Return the trained model.
        return self.model


    def __evaluate_epoch(self, valid_data: np.ndarray) -> Tuple[float, Dict[str, List[float]]]:
        '''
        Evaluate the model with one pass of the validation data.

        valid_data (np.ndarray): validation data.

        Return loss, metrics (tuple).
        '''
        # Initialize the loss aggregator and confusion matrix.
        loss = 0
        n_classes = np.unique(valid_data[:, 0]).shape[0]
        cm = ConfusionMatrix(n_classes)
        # Set the module to evaluation mode.
        self.eval()
        # Disable gradient calculation.
        with torch.no_grad():
            # Consider each batch of examples in the validation data.
            data = tud.DataLoader(valid_data, batch_size=1)
            for x, y in data:
                # Get predictions from the model.
                y_hat = self.model(x).squeeze(1)
                l = self.loss(y_hat, y)
                # Increment loss.
                loss += l.item()
                # Update the confusion matrix.
                cm.update(y, y_hat)
        # Return loss and metrics.
        return loss / valid_data.shape[0], cm.get_metrics()


    def train_model(self,
        train_data: np.ndarray,
        valid_data: np.ndarray,
        epochs: int = 10,
        target: str = 'precision'
    ) -> Type[nn.Module]:
        '''
        Train and evaluate the model, selecting the performant across epochs by
        the metric indicated with respect to the base class.

        train_data (np.ndarray): training data to calibrate the model.
        valid_data (np.ndarray): validation data to evaluate the model.
        epochs (int): number of passes to make with training and evaluation.
        target (str): the metric by which to select the best model.

        Return the best model (nn.Module).
        '''
        # Initialize placeholders for the best model and metric.
        best_model, best_metric = None, None
        # Train the model for the number of epochs.
        for epoch in range(epochs):
            # Train the model with training data.
            model = self.__train_epoch(train_data)
            # Evaluate the model with validation data.
            loss, metrics = self.__evaluate_epoch(valid_data)
            # Check whether this model is an improvement over all others.
            if best_metric is None or metrics[target][0] > best_metric:
                best_model = copy.deepcopy(model)
                best_metric = metrics[target]
        # Return the best model of all epochs.
        return best_model


class ConfusionMatrix:

    def __init__(self, n: int = 1) -> None:
        '''
        Initialize a confusion matrix for n classes, represented in a
        multidimensional array as [[[TP, FN], [FP, TN]], …].

        n (int): number of classes in the data.

        Return confusion matrix (ConfusionMatrix).
        '''
        self.__n = n
        self.__M = np.zeros(shape=(n, 2, 2))
    

    def update(self, y, y_hat) -> None:
        '''
        Update the true and false positives and negatives, i.e. y ?= y_hat.

        y, y_hat (int): indices of the actual and predicted classes.
        '''
        if y == y_hat:
            # Increment true positives for the true class.
            self.__M[y, 0, 0] += 1
            # Increment true negatives for all other classes.
            for c in range(self.__n):
                if c == y:
                    continue
                self.__M[c, 1, 1] += 1
        else:
            # Increment false positives for the predicted class.
            self.__M[y_hat, 1, 0] += 1
            # Increment false negatives for the true class.
            self.__M[y, 0, 1] += 1


    def get_metrics(self) -> Dict[str, List[float]]:
        '''
        Calculate accuracy, precision, recall, and F1 from the confusion matrix.

        Return metrics by class (dict of lists of floats).
        '''
        # Initialize containers for accuracy, precision, and recall.
        A, P, R, F = [], [], [], []
        # Calculate accuracy, precision, recall, F1 scores for each class.
        for c in range(self.__n):
            A.append((self.__M[c, 0, 0] + self.__M[c, 1, 1]) / self.__M[c, :, :].sum())
            P.append(self.__M[c, 0, 0] / self.__M[c, :, 0].sum())
            R.append(self.__M[c, 0, 0] / self.__M[c, 0, :].sum())
            F.append(2 * P[c] * R[c] / (P[c] + R[c]))
        # Return the metrics.
        return {'accuracy': A, 'precision': P, 'recall': R, 'f1': F}


    @property
    def matrix(self) -> np.ndarray:
        '''
        Get the confusion matrix.

        Return the matrix (np.ndarray).
        '''
        return self.__M

