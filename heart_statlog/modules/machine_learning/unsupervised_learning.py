import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torcheval.metrics import (
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    BinaryAUROC
    )


class NeuralNet2Layer:
    """Simple two-layer neural network built from scratch"""
    def __init__(self, input_layers, learn_rate, iterations):
        self.params = {}
        self.learn_rate = learn_rate
        self.iterations = iterations
        self.loss = []
        self.sample_size = None
        self.layers = input_layers
        self.X = None
        self.y = None

    def init_weights(self):
        # Initialize weights from a random normal distribution
        np.random.seed(1)
        self.params['W1'] = np.random.randn(self.layers[0], self.layers[1])
        self.params['b1'] = np.random.randn(self.layers[1])
        self.params['W2'] = np.random.randn(self.layers[1], self.layers[2])
        self.params['b2'] = np.random.randn(self.layers[2])

    def relu(self, Z):
        return np.maximum(0,Z)

    def dRelu(self, x):
        x[x<=0] = 0
        x[x>0] = 1
        return x

    def sigmoid(self, Z):
        return 1/(1+np.exp(-Z))

    def tanh(self, Z):
        return np.tanh(Z)

    def eta(self, input_val):
        eta = 0.0000000001
        return np.maximum(input_val, eta)

    def entropy_loss(self, input_y, input_y_hat):
        n_sample = len(input_y)
        y_inv = 1.0 - input_y
        y_hat_inv = 1.0 - input_y_hat

        # Clip values to avoid null values
        y_hat = self.eta(input_y_hat)
        y_hat_inv = self.eta(y_hat_inv)

        # Calculate loss
        loss = -1/n_sample * (
            np.sum(np.multiply(np.log(y_hat), input_y)
                   + np.multiply((y_inv), np.log(y_hat_inv))))
        return loss

    def binary_cross_entropy(self, input_y_true, input_y_pred):
        # Small epsilon value added to clip values to prevent log(0)
        epsilon = 1e-15
        y_pred = np.clip(input_y_pred, epsilon, 1-epsilon)
        loss = -(input_y_true*np.log(y_pred)+(1-input_y_true)*np.log(1-y_pred))
        return np.mean(loss)

    def categorical_cross_entropy(self, input_y_true, input_y_pred):
        epsilon = 1e-15
        y_pred = np.clip(input_y_pred, epsilon, 1.0)
        loss = -np.sum(input_y_true*np.log(y_pred), axis=1)
        return np.mean(loss)

    def forward_propagation(self):
        Z1 = self.X.dot(self.params['W1']) + self.params['b1']
        A1 = self.relu(Z1)
        Z2 = A1.dot(self.params['W2']) + self.params['b2']
        y_hat = self.sigmoid(Z2)
        loss = self.entropy_loss(self.y, y_hat)

        # Save calculated parameters
        self.params['Z1'] = Z1
        self.params['Z2'] = Z2
        self.params['A1'] = A1
        return y_hat, loss

    def back_propagation(self, input_y_hat):
        y_inv = 1 - self.y
        y_hat_inv = 1 - input_y_hat

        dl_wrt_y_hat = np.divide(y_inv, self.eta(y_hat_inv)) - np.divide(
            self.y, self.eta(input_y_hat))
        dl_wrt_sig = input_y_hat * y_hat_inv
        dl_wrt_z2 = dl_wrt_y_hat * dl_wrt_sig

        dl_wrt_A1 = dl_wrt_z2.dot(self.params['W2'].T)
        dl_wrt_w2 = self.params['A1'].T.dot(dl_wrt_z2)
        dl_wrt_b2 = np.sum(dl_wrt_z2, axis=0, keepdims=True)

        dl_wrt_z1 = dl_wrt_A1 * self.dRelu(self.params['Z1'])
        dl_wrt_w1 = self.X.T.dot(dl_wrt_z1)
        dl_wrt_b1 = np.sum(dl_wrt_z1, axis=0, keepdims=True)

        # Update the weights and bias
        self.params['W1'] = self.params['W1'] - (self.learn_rate * dl_wrt_w1)
        self.params['W2'] = self.params['W2'] - (self.learn_rate * dl_wrt_w2)
        self.params['b1'] = self.params['b1'] - (self.learn_rate * dl_wrt_b1)
        self.params['b2'] = self.params['b2'] - (self.learn_rate * dl_wrt_b2)

    def fit(self, input_X, input_y):
        self.X = input_X
        self.y = input_y
        self.init_weights()

        for i in range(self.iterations):
            y_hat, loss = self.forward_propagation()
            self.back_propagation(y_hat)
            self.loss.append(loss)

    def predict(self, input_X):
        Z1 = input_X.dot(self.params['W1']) + self.params['b1']
        A1 = self.relu(Z1)
        Z2 = A1.dot(self.params['W2']) + self.params['b2']
        pred = self.sigmoid(Z2)
        return np.round(pred)

    def calc_accuracy(self, input_y, input_y_hat):
        accuracy = int(sum(input_y.flatten() == input_y_hat.flatten())
                       / len(input_y.flatten()) * 100)
        return accuracy



class NeuralNetPyTorch:
    """Base class for neural networks built with PyTorch"""
    #----------------------------------------
    # CONVERSION FUNCTIONS
    #----------------------------------------

    def convert_to_tensor(self, input_data, mode):
        match mode:
            case 'pandas':
                tensor = torch.tensor(input_data.values, dtype=torch.float32)
            case 'numpy':
                tensor = torch.from_numpy(input_data).to(torch.float32)
            case 'pd_np':
                np_array = input_data.to_numpy().astype(np.float32)
                tensor = torch.from_numpy(np_array).to(torch.float32)
        return tensor

    def convert_tensor_to_numpy(self, input_tensor):
        if input_tensor.requires_grad:
            np_array = input_tensor.detach().numpy()
        else:
            np_array = input_tensor.numpy()
        return np_array

    def convert_prob_to_binary(self, input_tensor):
        val_list = []
        for x in input_tensor:
            match x.item():
                case n if n < 0.5:
                    val_list.append(0)
                case _:
                    val_list.append(1)
        output = torch.Tensor(val_list).int()
        return output


class TwoLayerNN(nn.Module, NeuralNetPyTorch):
    """Two-layer neural network built with PyTorch"""
    def __init__(self, size_input, size_hidden, size_output):
        super(TwoLayerNN, self).__init__()

        # First layer (input -> hidden) and activation function
        self.fc1 = nn.Linear(size_input, size_hidden)
        self.relu = nn.ReLU()

        # Second layer (hidden -> output)
        self.fc2 = nn.Linear(size_hidden, size_output)

    def forward(self, input_val):
        x = self.fc1(input_val)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class ThreeLayerNN(nn.Module, NeuralNetPyTorch):
    """Three-layer neural network built with PyTorch"""
    def __init__(self, size_input, size_hidden_1, size_hidden_2, size_output):
        super(ThreeLayerNN, self).__init__()
        self.size_input = size_input
        self.size_hidden_1 = size_hidden_1
        self.size_hidden_2 = size_hidden_2
        self.size_output = size_output
        self.define_layers()
        self.define_activation_function()

    def define_layers(self):
        self.fc1 = nn.Linear(self.size_input, self.size_hidden_1)
        self.fc2 = nn.Linear(self.size_hidden_1, self.size_hidden_2)
        self.fc3 = nn.Linear(self.size_hidden_2, self.size_output)

    def define_activation_function(self):
        self.activate = nn.ReLU()

    def forward(self, input_val):
        x = self.fc1(input_val)
        x = self.activate(x)
        x = self.fc2(x)
        x = self.activate(x)
        x = self.fc3(x)
        return x


class NeuralNetPerformance:
    """
    Calculates performance metrics for trained PyTorch-based neural networks
    """
    def __init__(self, input_model, input_preds_probs, input_target):
        self.model = input_model
        self.preds_probs = input_preds_probs
        self.target = input_target
        self.flatten_arrays()
        self.convert_prob_to_bin()

    def flatten_arrays(self):
        """Flattens tensors for performance evaluation calculations"""
        self.preds_flat = torch.flatten(self.preds_probs)
        self.targets_flat = torch.flatten(self.target)

    def convert_prob_to_bin(self):
        """Converts from probability to binary"""
        self.preds_probs_class = self.model.convert_prob_to_binary(
            self.preds_probs)

    def calc_metrics(self):
        # Accuracy
        self.accuracy = BinaryAccuracy()
        self.accuracy.update(self.preds_flat, self.targets_flat)
        print(f"\nAccuracy: {self.accuracy.compute()}")

        # Precision
        self.precision = BinaryPrecision()
        self.precision.update(self.preds_flat, self.targets_flat)
        print(f"Precision: {self.precision.compute()}")

        # Recall
        self.recall = BinaryRecall()
        self.recall.update(self.preds_probs_class, self.targets_flat.int())
        print(f"Recall: {self.recall.compute()}")

        # F1 Score
        self.f1_score = BinaryF1Score()
        self.f1_score.update(self.preds_flat, self.targets_flat)
        print(f"F1 Score: {self.f1_score.compute()}")

        #AUROC
        self.auroc = BinaryAUROC()
        self.auroc.update(self.preds_flat, self.targets_flat)
        print(f"AUROC: {self.auroc.compute()}")