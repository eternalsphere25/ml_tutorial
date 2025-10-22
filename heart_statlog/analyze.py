"""
https://www.comet.com/site/blog/building-a-neural-network-from-scratch-using-python-part-1/
https://heartbeat.comet.ml/building-a-neural-network-from-scratch-using-python-part-2-testing-the-network-c1f0c1c9cbb0
https://medium.com/@pdquant/all-the-backpropagation-derivatives-d5275f727f60
"""


import copy
import numpy as np
import pandas as pd
from modules.machine_learning import unsupervised_learning as mlu
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


#------------------------------------------------------------------------------
# PART 3: Run Program
#------------------------------------------------------------------------------

layers = [13,8,1]
neural_net = mlu.NeuralNet2Layer(layers, 0.001, 100)
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







# https://pytorch.org/get-started/locally/#start-locally
import torch
import torch.nn as nn
import torch.optim as optim
from torcheval.metrics import (
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    BinaryAUROC
    )

# Generate model
#model = mlu.TwoLayerNN(13,8,1)
model = mlu.ThreeLayerNN(13, 9, 6, 1)
print(model)

# Convert data to PyTorch tensor format
X_train_tensor = model.convert_to_tensor(X_train, mode='numpy')
X_test_tensor = model.convert_to_tensor(X_test, mode='numpy')
y_train_tensor = model.convert_to_tensor(y_train, mode='numpy')
y_test_tensor = model.convert_to_tensor(y_test, mode='numpy')

# Train model
#criterion = nn.MSELoss()
criterion = nn.SmoothL1Loss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
cycles = 5000
for epoch in range(cycles):
    # Clear previous gradients
    optimizer.zero_grad()

    # Forward pass, calculate loss, then backward pass to compute gradients
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()

    # Update weights
    optimizer.step()
    print(f"Epoch [{epoch +1}/{cycles}], Loss: {loss.item():.4f}")

# Make prediction and convert output from logit to probability
preds = model(X_test_tensor)
preds_probs = torch.relu(preds)

# Evaluate model
performance = mlu.NeuralNetPerformance(model, preds_probs, y_test_tensor)
performance.calc_metrics()