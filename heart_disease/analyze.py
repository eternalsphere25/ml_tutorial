import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from scipy.stats import pointbiserialr, spearmanr
from sklearn import svm
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def calculate_confusion_matrix(input_y_test, input_y_pred):
    # Generate confusion matrix
    cm = confusion_matrix(input_y_test, input_y_pred)

    #Calculate TP/FN/FP/TN
    tp = np.diag(cm)
    fn = cm.sum(axis=1) - np.diag(cm)
    fp = cm.sum(axis=0) - np.diag(cm)
    tn = np.sum(cm) - (fp + fn + tp)
    return cm, tp, fn, fp, tn

def calculate_correlation(input_df, target_name):
    # Calculate correlation
    columns = input_df.columns
    parameter = []
    correlation = []
    correlation_abs = []
    for x in columns:
        if x != target_name:
            if len(input_df[x].unique()) <= 2:
                r = spearmanr(input_df[target_name],input_df[x])[0]
            else:
                r = pointbiserialr(input_df[target_name],input_df[x])[0]
            parameter.append(x)
            correlation.append(r)
            correlation_abs.append(abs(r))

    # Assemble dataframe based on calculated correlation scores
    df_param = pd.DataFrame({'parameter': parameter,
                            'correlation': correlation,
                            'abs_corr': correlation_abs})
    return df_param


def calculate_metrics(tp, fn, fp, tn):
    # Calculate accuracy/precision/recall/F1
    accuracy = (tp[0]+tn[0])/(tp[0]+tn[0]+fp[0]+fn[0])
    precision = tp[0]/(tp[0]+fp[0])
    recall = tp[0]/(tp[0]+fn[0])
    f1 = 2*((precision*recall)/(precision+recall))
    return accuracy, precision, recall, f1

def calculate_roc_auc(input_y_test, input_y_pred):
    fpr, tpr, threshold = roc_curve(input_y_test, input_y_pred)
    auc = roc_auc_score(input_y_test, input_y_pred)
    return fpr, tpr, threshold, auc

def get_all_the_metrics(input_y_test, input_y_pred):
    # Calculate confusion matrix and related statistics
    #https://scikit-learn.org/stable/modules/
    # model_evaluation.html#confusion-matrix
    #https://stackoverflow.com/questions/31324218/
    # scikit-learn-how-to-obtain-true-positive-true-negative
    # -false-positive-and-fal
    cm, tp, fn, fp, tn = calculate_confusion_matrix(input_y_test, input_y_pred)
    #print(cm)
    #print(f"True Positives: {tp[0]}")
    #print(f"False Negatives: {fn[0]}")
    #print(f"False Positives: {fp[0]}")
    #print(f"True Negatives: {tn[0]}")

    # Calculate metrics
    #https://www.evidentlyai.com/classification-metrics/
    # accuracy-precision-recall
    #https://medium.com/analytics-vidhya/
    # confusion-matrix-accuracy-precision-recall-f1-score-ade299cf63cd
    #https://www.labelf.ai/blog/what-is-accuracy-precision-recall-and-f1-score
    accuracy, precision, recall, f1 = calculate_metrics(tp, fn, fp, tn)
    #print(f"Accuracy: {accuracy}")
    #print(f"Precision: {precision}")
    #print(f"Recall: {recall}")
    #print(f"F1: {f1}")

    # Calculate ROC AUC score
    #Inputs need to be numeric
    fpr, tpr, threshold, auc = calculate_roc_auc(input_y_test, input_y_pred)
    #print(f"FPR: {fpr}")
    #print(f"TPR: {tpr}")
    #print(f"Threshold: {threshold}")
    #print(f"AUC score: {auc}")
    return fpr, tpr, threshold, auc, cm

def print_cross_metrics(input_scores, mode, verbose=True):

    def print_metric_single(metric, parameter):
        print(f"{metric}: {parameter}")

    def print_metric_average(metric, parameter):
        print(f"{metric}: {round(parameter.mean(),2)} \u00B1 "
              f"{round(parameter.std(),2)}")

    if mode == "all":
        print_metric_single("\nModel", input_scores["estimator"])
        print_metric_single("Accuracy (Train)", input_scores["train_score"])
        print_metric_single("Accuracy (Test)", input_scores["test_score"])
        if verbose == True:
            print_metric_single("Fit Time", input_scores["fit_time"])
            print_metric_single("Score Time", input_scores["score_time"])

    elif mode == "average":
        print(f"\nModel: {input_scores['estimator'][0]}")
        print_metric_average("Accuracy (Train)", input_scores["train_score"])
        print_metric_average("Accuracy (Test)", input_scores["test_score"])
        if verbose == True:
            print_metric_average("Fit Time", input_scores["fit_time"])
            print_metric_average("Score Time", input_scores["score_time"])




def run_prediction_model(model, model_name):
    # Train model
    model.fit(X_train, y_train)

    # Make prediction
    y_pred = model.predict(X_test)

    # Evaluate model
    acc_train = model.score(X_train, y_train)
    acc_test = model.score(X_test, y_test)
    print(f"\nAccuracy on training data ({model_name}): {acc_train}")
    print(f"Accuracy on test data ({model_name}): {acc_test}")
    fpr, tpr, threshold, auc, cm = get_all_the_metrics(y_test, y_pred)
    print(f"AUC ({model_name}): {auc}")
    return acc_train, acc_test, fpr, tpr, threshold, auc, cm


#------------------------------------------------------------------------------
# PART 0: Define global constants
#------------------------------------------------------------------------------

# Data location
# http://archive.ics.uci.edu/dataset/45/heart+disease
input_data = Path(Path(__file__).parents[0], "data", "cleveland.data")

# Attribute descriptions:
"""
3 age: age in years
4 sex: sex (1 = male; 0 = female)
9 cp: chest pain type
-- Value 1: typical angina
-- Value 2: atypical angina
-- Value 3: non-anginal pain
-- Value 4: asymptomatic
10 trestbps: resting blood pressure (in mm Hg on admission to the hospital)
12 chol: serum cholestoral in mg/dl
16 fbs: (fasting blood sugar > 120 mg/dl)  (1 = true; 0 = false)
19 restecg: resting electrocardiographic results
-- Value 0: normal
-- Value 1: having ST-T wave abnormality (T wave inversions and/or ST
            elevation or depression of > 0.05 mV)
-- Value 2: showing probable or definite left ventricular hypertrophy by
            Estes' criteria
32 thalach: maximum heart rate achieved
38 exang: exercise induced angina (1 = yes; 0 = no)
40 oldpeak = ST depression induced by exercise relative to rest
41 slope: the slope of the peak exercise ST segment
-- Value 1: upsloping
-- Value 2: flat
-- Value 3: downsloping
44 ca: number of major vessels (0-3) colored by flourosopy
51 thal: 3 = normal; 6 = fixed defect; 7 = reversable defect
58 num: diagnosis of heart disease (angiographic disease status)
-- Value 0: < 50% diameter narrowing
-- Value 1: > 50% diameter narrowing
(in any major vessel: attributes 59 through 68 are vessels)
"""

# Column names for data
col_names = ["age",
             "sex",
             "cp",
             "trestbps",
             "chol",
             "fbs",
             "restecg",
             "thalach",
             "exang",
             "oldpeak",
             "slope",
             "ca",
             "thal",
             "num"]

# Categorical variable value key
val_guide = {"sex": {0: "female", 1: "male"},
             "cp": {1: "typical angina", 2: "atypical angina",
                    3: "non-anginal pain", 4: "asymptomatic"},
             "fbs": {0: "< 120 mg/dl", 1: "> 120 mg/dl"},
             "restecg": {0: "normal", 1: "ST-T wave abnormality",
                         2: "left ventricular hypertrophy"},
             "exang": {0: "no", 1: "yes"},
             "slope": {1: "upsloping", 2: "flat", 3: "downsloping"},
             "thal": {3: "normal", 6: "fixed defect", 7: "reversable defect"}}

# Value locations
val_index = [3, 4, 9, 10, 12, 16, 19, 32, 38, 40, 41, 44, 51, 58]

# Target variable
target_val = "num"

# Set dataframe display options
pd.set_option('display.max_rows', None)


#------------------------------------------------------------------------------
# PART 1: Import data
#------------------------------------------------------------------------------

raw_text = input_data.read_text(errors='ignore')

value_list = []
value = ""
for x in range(len(raw_text)):
    if raw_text[x] == "\n":
        value_list.append(value)
        value = ""
    elif raw_text[x] != " ":
        value = value + raw_text[x]
    elif raw_text[x] == " ":
        value_list.append(value)
        value = ""


#------------------------------------------------------------------------------
# PART 2: Preprocess and clean data
#------------------------------------------------------------------------------

# Separate into discrete entries
single_entry, separated_entry_list = [], []
current_index = 0
while current_index < len(value_list):
    for y in range(len(value_list)):
        current_value = value_list[y]
        if current_value != 'name':
            single_entry.append(current_value)
            current_index = y+1
        else:
            single_entry.append(current_value)
            current_index = y+1
            separated_entry_list.append(single_entry)
            single_entry = []

# Filter out rows that have too much / too little data
separated_entry_list = [x for x in separated_entry_list if len(x)==76]

# For each entry, extract only the 14 relevant variables
var_loc = [i-1 for i in val_index]

# Build dataframe from extracted data
df_data = pd.DataFrame(columns=col_names)
for entry in separated_entry_list:
    var_list = []
    for x in var_loc:
        var_list.append(entry[x])
    df_temp = pd.DataFrame([var_list], columns=col_names)
    df_data = pd.concat([df_data, df_temp])
df_data = df_data.reset_index(drop=True)

# Convert column datatypes to int/float
for item in col_names:
    if item != "oldpeak":
        df_data = df_data.astype({item: "int"})
    else:
        df_data = df_data.astype({item: "float"})

# Change "num" to 0 = No heart disease, 1 = heart disease
df_data.loc[df_data["num"] != 0, "num"] = 1

# Display results
print(f"Extracted Data:")
print(df_data)


#------------------------------------------------------------------------------
# PART 3: Feature selection
#------------------------------------------------------------------------------

#----------------------------------------
# PART 3-1: Split data
#----------------------------------------

# Remove target variable from the list
col_mod = col_names.copy()
col_mod.remove(target_val)

# Split data between test and train data
X = df_data[col_mod]
y = df_data[target_val]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

#----------------------------------------
# PART 3-2: Calculate variable
#           correlation with target
#----------------------------------------

# Add the target variable back in to calculate correlations
df_correlation = X_train.copy()
df_correlation[target_val] = y_train

# Calculate correlation
df_param = calculate_correlation(df_correlation, target_val)
print(f'\nCorrelation of variable with target "{target_val}":')
print(df_param.sort_values(by="correlation", ascending=False))


#------------------------------------------------------------------------------
# PART 4: Run prediction models
#------------------------------------------------------------------------------

#----------------------------------------
# PART 4-1: Gaussian Naive Bayes
#----------------------------------------

# Generate classifier object
gnb = GaussianNB()

# Train model and make predictions
gnb_results = run_prediction_model(gnb, "Gaussian Naive Bayes")

# Run k-fold cross validation
scores = cross_validate(gnb, X, y, cv=10, return_estimator=True,
                        return_train_score=True)
print_cross_metrics(scores, "average", verbose=False)

#----------------------------------------
# PART 4-2: Decision Tree
#----------------------------------------

# Generate object and train model
tree = DecisionTreeClassifier(criterion="gini", max_depth=None)

# Train model and make predictions
tree_results = run_prediction_model(tree, "Decision Tree")

# Run k-fold cross validation
scores = cross_validate(tree, X, y, cv=10, return_estimator=True,
                        return_train_score=True)
print_cross_metrics(scores, "average", verbose=False)

#----------------------------------------
# PART 4-3: Random Forest
#----------------------------------------

# Generate object and train model
rf = RandomForestClassifier()

# Train model and make predictions
rf_results = run_prediction_model(rf, "Random Forest")

# Run k-fold cross validation
scores = cross_validate(rf, X, y, cv=10, return_estimator=True,
                        return_train_score=True)
print_cross_metrics(scores, "average", verbose=False)

#----------------------------------------
# PART 4-4: Logistic Regression
#----------------------------------------

# Generate object and train model
log_reg = LogisticRegression(solver="liblinear")

# Train model and make predictions
log_reg_results = run_prediction_model(log_reg, "Logistic Regression")

# Run k-fold cross validation
scores = cross_validate(log_reg, X, y, cv=10, return_estimator=True,
                        return_train_score=True)
print_cross_metrics(scores, "average", verbose=False)

#----------------------------------------
# PART 4-5: K-Nearest Neighbors
#----------------------------------------

# Generate object and train model
knn = KNeighborsClassifier(n_neighbors=4)

# Train model and make predictions
knn_results = run_prediction_model(knn, "K-Nearest Neighbors")

# Run k-fold cross validation
scores = cross_validate(knn, X, y, cv=10, return_estimator=True,
                        return_train_score=True)
print_cross_metrics(scores, "average", verbose=False)

#----------------------------------------
# PART 4-6: Support Vector Machine
#----------------------------------------

# Generate object and train model
s_vector = svm.SVC(kernel='linear')

# Train model and make predictions
s_vector_results = run_prediction_model(s_vector, "Support Vector Machine")

# Run k-fold cross validation
scores = cross_validate(s_vector, X, y, cv=10, return_estimator=True,
                        return_train_score=True)
print_cross_metrics(scores, "average", verbose=False)


#----------------------------------------
# PART 4-7: Linear Regression
#----------------------------------------

try:
    # Generate object and train model
    lin_reg = LinearRegression()

    # Train model and make predictions
    lin_reg_results = run_prediction_model(lin_reg, "Linear Regression")

    # Run k-fold cross validation
    scores = cross_validate(lin_reg, X, y, cv=10, return_estimator=True,
                            return_train_score=True)
    print_cross_metrics(scores, "average", verbose=False)

except ValueError:
    print(f"\nWARNING: Linear Regression cannot be used for a mix of binary "
          f"and continuous targets!\nNOTICE: Linear Regression skipped; "
          f"proceeding to next machine learning algorithm...")

#----------------------------------------
# PART 4-8: K-Means Clustering
#----------------------------------------

"""
# Generate object and train model
kmeans = KMeans(n_clusters=50, n_init="auto")

# Train model and make predictions
kmeans_results = run_prediction_model(kmeans, "K-Means Clustering")

# Run k-fold cross validation
scores = cross_validate(kmeans, X, y, cv=10, return_estimator=True,
                        return_train_score=True)
print_cross_metrics(scores, "average", verbose=False)
"""

#----------------------------------------
# PART 4-9: Gradient Boosting
#----------------------------------------

# Generate object and train model
grad_boost = GradientBoostingClassifier()

# Train model and make predictions
grad_boost_results = run_prediction_model(grad_boost, "Gradient Boosting")

# Run k-fold cross validation
scores = cross_validate(grad_boost, X, y, cv=10, return_estimator=True,
                        return_train_score=True)
print_cross_metrics(scores, "average", verbose=False)

#----------------------------------------
# PART 4-10: Gradient Boosting (XGBoost)
#----------------------------------------

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)

    # Generate object and train model
    xg_boost = XGBClassifier(silent=True, verbosity=0)

    # Train model and make predictions
    xg_boost_results = run_prediction_model(xg_boost,
                                            "Gradient Boosting (XGBoost)")

    # Run k-fold cross validation
    scores = cross_validate(xg_boost, X, y, cv=10, return_estimator=True,
                            return_train_score=True)
    print_cross_metrics(scores, "average", verbose=False)

#----------------------------------------
# PART 4-11: Gradient Boosting (AdaBoost)
#----------------------------------------

# Generate object and train model (SVM)
ada_boost = AdaBoostClassifier(n_estimators=100)

# Train model and make predictions
ada_boost_results = run_prediction_model(ada_boost,
                                         "Gradient Boosting (AdaBoost)")

# Run k-fold cross validation
scores = cross_validate(ada_boost, X, y, cv=10, return_estimator=True,
                        return_train_score=True)
print_cross_metrics(scores, "average", verbose=False)

# Generate object and train model (SVM)
ada_boost = AdaBoostClassifier(estimator=s_vector, n_estimators=100,
                               algorithm="SAMME")

# Train model and make predictions
ada_boost_results = run_prediction_model(ada_boost,
                                         "Gradient Boosting (AdaBoost-SVM)")

# Run k-fold cross validation
scores = cross_validate(ada_boost, X, y, cv=10, return_estimator=True,
                        return_train_score=True)
print_cross_metrics(scores, "average", verbose=False)


# Generate object and train model (GNB)
ada_boost = AdaBoostClassifier(estimator=gnb, n_estimators=100)

# Train model and make predictions
ada_boost_results = run_prediction_model(ada_boost,
                                         "Gradient Boosting (AdaBoost-GNB)")

# Run k-fold cross validation
scores = cross_validate(ada_boost, X, y, cv=10, return_estimator=True,
                        return_train_score=True)
print_cross_metrics(scores, "average", verbose=False)