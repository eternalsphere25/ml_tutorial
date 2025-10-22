# ml_tutorial: Autodidactic Machine Learning
This repository contains code used as part of self-study for machine learning

## Data Source
UCI Machine Learning Repository: https://archive.ics.uci.edu/

## General Parameter Benchmarks
- Accuracy: 70-90
- Precision: as high as possible; ideal >90
- Recall: as high as possible; ideal >90
- F1: >0.7
- ROC AUC: >0.7-0.8

## Neural Networks
- During training, the goal is to minimize a loss function by adjusting network weights
- The backpropagation algorithm calculates these gradients by propagating the error from the output layer to the input layer
- Vanishing gradient is when the gradients used to update the network become extremely small or 'vanish' as they are backpropagates from the output layers to the earlier layers

#### Notes:
The output from PyTorch is logit if there is no activation function in the last layer. In this case, a conversion from logit to probability will be necessary:
```
# Make prediction:
preds = model(X_test_tensor)

# Convert from logit to probability (either multiclass or binary)
prob_multiclass_classification = nn.functional.softmax(preds, dim=1)
prob_binary_classification = torch.relu(preds)
```
### PyTorch Activation Functions
#### Logistic (Sigmoid)
- `torch.sigmoid(x)` standalone; `nn.sigmoid()` when used in a layer
- Useful for binary classification
- Output range [0, 1]
#### Tanh
- `torch.tanh(x)` standalone; `nn.tanh()` when used in a layer
- Faster convergence but more computationally expensive
- Output range [-1, 1]
#### ReLU
- `torch.relu(x)` standalone; `nn.relu()` when used in a layer
- Not susceptible to vanishing gradient problem like other activation functions
- Output range [0, ∞]

### PyTorch Loss Functions
#### Regression (for continuous values)
- Mean Squared Error (MSE): `nn.MSELoss()`
    - Good for general tasks
- Mean Absolute Error (MAE): `nn.L1Loss()`
    - More robust to outliers
- Huber Loss: `nn.SmoothL1Loss()`, `nn.HuberLoss()`
    - Less sensitive to outliers, smoother gradient around zero

### Classification (for discrete output)
- Cross-Entropy: `nn.CrossEntropyLoss()`
    - Used for multi-class classification
    - Combination of `nn.LogSoftmax()` and `nn.NLLLoss()`
- Binary Cross-Entropy: `nn.BCELoss()`
    - Used for binary classification problems
    - Input must be probabilities between 0 and 1 (usualyl via sigmoid activation)
- Binary Cross-Entropy with Logits: `nn.BCEWithLogitsLoss()`
    - More numerically stable due to combining sigmiod function and BCELoss
    - Input is raw logit values

### Ranking (for relative distances or orderings)
- Margin Ranking: `nn.MarginRankingLoss()`
    - Used for ranking problems
    - Compares relative similarity or dissimilarity
- Triplet Margin: `nn.TripletMarginLoss()`
    - Used for face verification or similarity learning
    - Pulls similar samples closer and pushes dissimilar samples further
- Cosine Embedding: `nn.CosingEmbeddingLoss()`
    - Used when vector direction is more important than magnitude
    - Measures similarity or dissimilarity

## Resources:
- Backpropagation Derivatives:
    - https://medium.com/@pdquant/all-the-backpropagation-derivatives-d5275f727f60
- 2-Layer Neural Network:
    - https://www.comet.com/site/blog/building-a-neural-network-from-scratch-using-python-part-1/
    - https://heartbeat.comet.ml/building-a-neural-network-from-scratch-using-python-part-2-testing-the-network-c1f0c1c9cbb0
- Categorical cross-entropy loss:
    - https://neuralthreads.medium.com/categorical-cross-entropy-loss-the-most-important-loss-function-d3792151d05b
- Building a Neural Network with PyTorch:
    - https://www.codecademy.com/article/building-a-neural-network-using-pytorch