"""
https://www.comet.com/site/blog/building-a-neural-network-from-scratch-using-python-part-1/
https://heartbeat.comet.ml/building-a-neural-network-from-scratch-using-python-part-2-testing-the-network-c1f0c1c9cbb0
https://medium.com/@pdquant/all-the-backpropagation-derivatives-d5275f727f60
"""


import copy
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


#------------------------------------------------------------------------------
# PART 1: Preprocess Data
#------------------------------------------------------------------------------

# Define file location and data headers
file_data = Path.cwd().joinpath(Path('data', 'heart.dat'))
col_list = ['age', 'sex', 'chest_pain', 'resting_blood_pressure',
           'serum_cholesterol', 'fasting_blood_sugar', 'resting_ecg_results',
           'max_heart_rate_achieved', 'exercise_induced_angina', 'oldpeak',
           'slope of the peak', 'num_of_major_vessels', 'thal',
           'heart_disease']

# Target variable
target_val = "heart_disease"

# Import data
df_raw = pd.read_csv(file_data, sep=' ', names=col_list)

# Display basic information about the dataset
print(f"\nFirst 10 rows of the dataset:\n{df_raw.head(10)}")
print(f"\nNumber of (rows, columns): {df_raw.shape}")
print(f"Recommended neural network size: "
      f"[{df_raw.shape[0]} x {df_raw.shape[1]-1}]")
print(f"\nCheck for nulls and datatype confirmation:\n{df_raw.isna().sum()}")

# Remove target column
col_mod = copy.deepcopy(col_list)
col_mod.remove(target_val)

# Split target column from data
X = df_raw[col_mod]
y = df_raw[target_val]

# Reassign target variable values
y = y.replace(1,0)
y = y.replace(2,1)
y = y.values.reshape(X.shape[0], 1)

# Split data between test and train data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2)

# Standardize dataset
sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

# Show results of data preprocessing
print(f"\nShape of train set is {X_train.shape}")
print(f"Shape of test set is {X_test.shape}")
print(f"Shape of train label is {y_train.shape}")
print(f"Shape of test labels is {y_test.shape}")


#------------------------------------------------------------------------------
# PART 2: Define Neural Network
#------------------------------------------------------------------------------

class NeuralNet2:
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


#------------------------------------------------------------------------------
# PART 3: Run Program
#------------------------------------------------------------------------------

layers = [13,8,1]
neural_net = NeuralNet2(layers, 0.001, 100)
neural_net.fit(X_train, y_train)

train_pred = neural_net.predict(X_train)
test_pred = neural_net.predict(X_test)

acc_train = neural_net.calc_accuracy(y_train, train_pred)
acc_test = neural_net.calc_accuracy(y_test, test_pred)


print(f"\nTrain Accuracy: {acc_train}")
print(f"Test Accuracy: {acc_test}")



from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

sk_net = MLPClassifier(
    hidden_layer_sizes=8, learning_rate_init=0.001, max_iter=100)
sk_net.fit(X_train, np.ravel(y_train))

pred_train = sk_net.predict(X_train)
pred_test = sk_net.predict(X_test)
acc_train = round(accuracy_score(pred_train, y_train),2)*100
acc_test = round(accuracy_score(pred_test, y_test),2)*100

print(f"\nTrain Accuracy (sklearn): {acc_train}")
print(f"Test Accuracy (sklearn): {acc_test}")