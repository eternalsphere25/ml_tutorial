import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
from scipy.stats import pointbiserialr, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, 
                             ConfusionMatrixDisplay, recall_score,
                             roc_curve, roc_auc_score)
from sklearn.model_selection import (cross_validate, cross_val_score,
                                     train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


def calculate_confusion_matrix(input_y_test, input_y_pred):
    #Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    #Calculate TP/FN/FP/TN
    tp = np.diag(cm)
    fn = cm.sum(axis=1) - np.diag(cm)
    fp = cm.sum(axis=0) - np.diag(cm)  
    tn = np.sum(cm) - (fp + fn + tp)
    return cm, tp, fn, fp, tn

def calculate_metrics(tp, fn, fp, tn):
    #Calculate accuracy/precision/recall/F1
    accuracy = (tp[0]+tn[0])/(tp[0]+tn[0]+fp[0]+fn[0])
    precision = tp[0]/(tp[0]+fp[0])
    recall = tp[0]/(tp[0]+fn[0])
    f1 = 2*((precision*recall)/(precision+recall))
    return accuracy, precision, recall, f1

def calculate_roc_auc(input_y_test, input_y_pred):
    fpr, tpr, threshold = roc_curve(input_y_test, input_y_pred)
    auc = roc_auc_score(input_y_test, input_y_pred)
    return fpr, tpr, threshold, auc

def filter_out_missing_values(input_df, input_condition):
    #Find missing values
    missing_rows = []
    for index, row in input_df.iterrows():
        value = row[input_condition]
        if value == "?":
                missing_rows.append(index)

    #Print out results
    print(f"\nFiltering out missing data points ({input_condition})")
    print(f"Number of data points before filtering: {input_df.shape[0]}")
    output_df = input_df.drop(index=missing_rows)
    print(f"Number of data points after filtering: {output_df.shape[0]}")
    return output_df

def get_all_the_metrics(input_y_test, input_y_pred):
    #Calculate confusion matrix and related statistics
    #https://scikit-learn.org/stable/modules/
    # model_evaluation.html#confusion-matrix
    #https://stackoverflow.com/questions/31324218/
    # scikit-learn-how-to-obtain-true-positive-true-negative
    # -false-positive-and-fal
    cm, tp, fn, fp, tn = calculate_confusion_matrix(input_y_test, input_y_pred)
    print(cm)
    print(f"True Positives: {tp[0]}")
    print(f"False Negatives: {fn[0]}")
    print(f"False Positives: {fp[0]}")
    print(f"True Negatives: {tn[0]}")

    #Calculate metrics
    #https://www.evidentlyai.com/classification-metrics/
    # accuracy-precision-recall
    #https://medium.com/analytics-vidhya/
    # confusion-matrix-accuracy-precision-recall-f1-score-ade299cf63cd
    #https://www.labelf.ai/blog/what-is-accuracy-precision-recall-and-f1-score
    accuracy, precision, recall, f1 = calculate_metrics(tp, fn, fp, tn)
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")

    #Calculate ROC AUC score
    #Inputs need to be numeric
    y_test_auc = np.where(y_test == '<=50K', 0, 1)
    y_pred_auc = np.where(y_pred == '<=50K', 0, 1)
    fpr, tpr, threshold, auc = calculate_roc_auc(y_test_auc, y_pred_auc)
    print(f"FPR: {fpr}")
    print(f"TPR: {tpr}")
    print(f"Threshold: {threshold}")
    print(f"AUC score: {auc}")
    return fpr, tpr, threshold, auc

def normalize_numerical_variables(input_df, col_name):
    input_df[col_name] = MinMaxScaler().fit_transform(
        input_df[col_name].to_numpy().reshape(-1,1))

def set_datatype(input_df, col_name, input_type):
    if input_type == 'numeric':
        input_df[col_name] = pd.to_numeric(input_df[col_name])


#------------------------------------------------------------------------------
# PART 0: Define global constants
#------------------------------------------------------------------------------

#Data location
#http://archive.ics.uci.edu/dataset/2/adult
input_data_1 = Path(Path(__file__).parents[0], "data", "adult.data")
input_data_2 = Path(Path(__file__).parents[0], "data", "adult.test")

#Set percentage of variables to use
features_to_use = 0.5

#Column names for data
col_names = ["age",
             "workclass",
             "fnlwgt",
             "education",
             "education-num",
             "marital-status",
             "occupation",
             "relationship",
             "race",
             "sex",
             "capital-gain",
             "capital-loss",
             "hours-per-week",
             "native-country",
             "income"]

num_var = ["age",
           "fnlwgt",
           "education-num",
           "capital-gain",
           "capital-loss", 
           "hours-per-week"]


#------------------------------------------------------------------------------
# PART 1: Import data
#------------------------------------------------------------------------------

#Import data
df_raw_1 = pd.read_csv(input_data_1, names=col_names, skipinitialspace=True)
df_raw_2 = pd.read_csv(input_data_2, names=col_names, skipinitialspace=True, 
                       skiprows=1)
df_raw = pd.concat([df_raw_1, df_raw_2])
df_raw["income"] = df_raw["income"].str.replace('.','')

#Display basic statistics
print(f"Number of Data Points: {df_raw.shape[0]}"
      f"\nNumber of Variables: {df_raw.shape[1]}")

#Since this is an income problem, show the incomes
income_over_50k = df_raw.loc[df_raw['income'] == '>50K'].shape[0]
income_under_50k = df_raw.loc[df_raw['income'] == '<=50K'].shape[0]
print(f"Number over 50k income: {income_over_50k} "
      f"({round((income_over_50k/df_raw.shape[0])*100,2)}%)")
print(f"Number at or below 50k income: {income_under_50k} "
      f"({round((income_under_50k/df_raw.shape[0])*100,2)}%)")


#------------------------------------------------------------------------------
# PART 2: Preprocess data
#------------------------------------------------------------------------------

#----------------------------------------
# PART 2-1: Filter and clean data
#----------------------------------------

#Find and remove records with missing values
df_data = filter_out_missing_values(df_raw, "workclass")
df_data = filter_out_missing_values(df_data, "occupation")
df_data = filter_out_missing_values(df_data, "native-country")

#Set datatypes
for item in num_var:
    set_datatype(df_data, item, 'numeric')

#Normalize numerical variables
for item in num_var:
    normalize_numerical_variables(df_data, item)

#----------------------------------------
# PART 2-2: Split data
#----------------------------------------

#Remove income from the list since it is the target variable
col_mod = col_names.copy()
col_mod.remove('income')

#Split data between test and train data
X = df_data[col_mod]
y = df_data['income']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)


#------------------------------------------------------------------------------
# PART 3: Feature selection
#------------------------------------------------------------------------------

#----------------------------------------
# PART 3-1: Encode categorial variables
#----------------------------------------

#One-hot encode categorial variables
df_encoded = pd.get_dummies(X_train)

#----------------------------------------
# PART 3-2: Calculate variable 
#           correlation with target
#----------------------------------------

#Use point biserial correlation when the variable is dichotomous (a variable
#with only 2 categories, ie gender is either male or female):
#https://statistics.laerd.com/statistical-guides/types-of-variable.php

#Otherwise use Spearman correlation to assess relationship between 2 variables

#Switch the income category to true/false in order to utilize statistics
income = np.where(y_train == '<=50K', 0, 1)
df_encoded['income'] = income

#Calculate correlation
columns = df_encoded.columns
parameter = []
correlation = []
correlation_abs = []
for x in columns:
    if x != 'income':
        if len(df_encoded[x].unique()) <= 2:
            r = spearmanr(df_encoded['income'],df_encoded[x])[0]
        else:
            r = pointbiserialr(df_encoded['income'],df_encoded[x])[0]
        parameter.append(x)
        correlation.append(r)
        correlation_abs.append(abs(r))

#Assemble dataframe based on calculated correlation scores
df_param = pd.DataFrame({'parameter': parameter,
                         'correlation': correlation,
                         'abs_corr': correlation_abs})

#Recombine all the one-hot-encoded variables back into single categories
#to determine the importance of the category as a whole
r_dict_merged = {}
for x in col_names:
    if x != 'income':
        col_val = {}
        #Gather all the correlation values for one-hot encoded variables
        for y in parameter:
            if f"{x}_" in y:
                col_val[y] = df_param.loc[
                    df_param['parameter'] == y]['correlation'].item()
        #Get the correlation values for the rest of the variables
        if len(col_val) == 0:
            col_val[x] = df_param.loc[
                df_param['parameter'] == x]['correlation'].item()
        #Merge all relevant categories
        r_dict_merged[x] = sum(col_val.values())

#Generate dataframe with merged values
df_param_merged = pd.DataFrame.from_dict(r_dict_merged, orient='index',
                                         columns=['correlation'])
df_param_merged['abs_corr'] = df_param_merged['correlation'].abs()

#Display sorted correlation values
df_param_sort = df_param_merged.sort_values(by='abs_corr', ascending=False)
df_param_sort = df_param_sort.reset_index(names="variable")
print(f"\nFeature relevance:")
print(df_param_sort)

#Display selected variables
num_vars = len(df_param_sort)*features_to_use
print(f"\nPercentage of variables to be selected: "
      f"{round(features_to_use*100)}% "
      f"(~{round(num_vars)} of {len(df_param_sort)})")

var_index_list = df_param_sort.index.tolist()[:int(num_vars)]
df_selected = df_param_sort.loc[df_param_sort.index[var_index_list]]
selected_features = df_selected["variable"].tolist()
print(f"Selected variables:")
print(df_selected)


#------------------------------------------------------------------------------
# PART 4: Run prediction models
#------------------------------------------------------------------------------

#One-hot encode categorial variables
X_train_encoded = pd.get_dummies(X_train)
X_test_encoded = pd.get_dummies(X_test)

#Confirm that both train and test have the same columns (variables)
not_in_test = X_train_encoded.columns.difference(X_test_encoded.columns)
not_in_train = X_test_encoded.columns.difference(X_train_encoded.columns)

for x in not_in_test:
    X_test_encoded[x] = False
for x in not_in_train:
    X_train_encoded[x] = False

#Rearrange columns to be in the same order
cols_train_list = X_train_encoded.columns.tolist()
X_test_encoded = X_test_encoded[cols_train_list]

#----------------------------------------
# PART 4-1: Gaussian Naive Bayes
#----------------------------------------

#Generate object
gnb = GaussianNB()

#Train model
gnb.fit(X_train_encoded, y_train)

#Make prediction
y_pred = gnb.predict(X_test_encoded)

print(f"\n------DEBUG-----")
print(y_test)
print(y_pred)

#Evaluate model
acc_train = gnb.score(X_train_encoded, y_train)
acc_test = gnb.score(X_test_encoded, y_test)
print(f"\nAccuracy on training data (Gaussian Naive Bayes): {acc_train}")
print(f"Accuracy on test data (Gaussian Naive Bayes): {acc_test}")
fpr_gnb, tpr_gnb, threshold_gnb, auc_gnb = get_all_the_metrics(y_test, y_pred)

#----------------------------------------
# PART 4-2: Decision Tree
#----------------------------------------

#Generate object and train model
tree = DecisionTreeClassifier(criterion="gini", max_depth=None)
tree = tree.fit(X_train_encoded, y_train)

#Make prediction
y_pred = tree.predict(X_test_encoded)

#Evaluate model
acc_test = accuracy_score(y_test, y_pred)
print(f"\nAccuracy on test data (Decision Tree): {acc_test}")
fpr_tree, tpr_tree, threshold_tree, auc_tree = get_all_the_metrics(
    y_test, y_pred)

#----------------------------------------
# PART 4-3: Random Forest
#----------------------------------------

#Generate object and train model
rf = RandomForestClassifier()
rf.fit(X_train_encoded, y_train)

#Make prediction
y_pred = rf.predict(X_test_encoded)

#Evaluate model
acc_test = accuracy_score(y_test, y_pred)
print(f"\nAccuracy on test data (Random Forest): {acc_test}")
fpr_rf, tpr_rf, threshold_rf, auc_rf = get_all_the_metrics(y_test, y_pred)

#----------------------------------------
# PART 4-4: Logistic Regression
#----------------------------------------

#Generate object and train model
logreg = LogisticRegression(solver="liblinear")
logreg.fit(X_train_encoded, y_train)

#Make prediction
y_pred = rf.predict(X_test_encoded)

#Evaluate model
acc_test = accuracy_score(y_test, y_pred)
print(f"\nAccuracy on test data (Logistic Regression): {acc_test}")
fpr_logreg, tpr_logreg, threshold_logreg, auc_logreg = get_all_the_metrics(
    y_test, y_pred)

#----------------------------------------
# PART 4-X: Generate and graph ROC curves
#----------------------------------------

fpr_random = np.arange(0, 1.1, 0.1)
tpr_random = np.arange(0, 1.1, 0.1)
fig, ax = plt.subplots()
ax.plot(fpr_random, tpr_random, linestyle="dashed", color="tab:orange")
ax.plot(fpr_gnb, tpr_gnb, color="tab:blue")
ax.plot(fpr_tree, tpr_tree, color="tab:olive")
ax.plot(fpr_rf, tpr_rf, color="tab:green")
ax.plot(fpr_logreg, tpr_logreg, color="tab:red")
ax.set_ylabel("TPR (True Positive Rate)")
ax.set_xlabel("FPR (False Positive Rate)")
fig.legend([f"Random  [AUC = 0.5]", 
            f"Gaussian Naive Bayes [AUC = {round(auc_gnb, 2)}]",
            f"Decision Tree [AUC = {round(auc_tree, 2)}]",
            f"Random Forest [AUC = {round(auc_rf, 2)}]",
            f"Logistic Regression [AUC = {round(auc_logreg, 2)}]"], 
            loc='lower right', bbox_to_anchor=[0.9, 0.11])
plt.show()