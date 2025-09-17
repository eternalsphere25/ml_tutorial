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
                                     GridSearchCV, RandomizedSearchCV, 
                                     RepeatedStratifiedKFold, 
                                     train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


def build_overfit_df(input_key_list, input_val_list):
    output_dict = {}
    for x in range(len(input_key_list)):
        output_dict[input_key_list[x]] = input_val_list[x]
    output_df = pd.DataFrame([output_dict])
    return output_df

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

def normalize_numerical_variables(input_df, col_name):
    input_df[col_name] = MinMaxScaler().fit_transform(
        input_df[col_name].to_numpy().reshape(-1,1))

def print_cross_metrics(input_scores, mode):

    def print_metric_single(metric, parameter):
        print(f"{metric}: {parameter}")

    def print_metric_average(metric, parameter):
        print(f"{metric}: {round(parameter.mean(),2)} \u00B1 "
              f"{round(parameter.std(),2)}")

    if mode == "all":
        print_metric_single("Model", input_scores["estimator"])
        print_metric_single("\nAccuracy (Train)", input_scores["train_score"])
        print_metric_single("Accuracy (Test)", input_scores["test_score"])
        print_metric_single("Fit Time", input_scores["fit_time"])
        print_metric_single("Score Time", input_scores["score_time"])

    elif mode == "average":
        print(f"Model: {input_scores['estimator'][0]}")
        print_metric_average("\nAccuracy (Train)", input_scores["train_score"])
        print_metric_average("Accuracy (Test)", input_scores["test_score"])
        print_metric_average("Fit Time", input_scores["fit_time"])
        print_metric_average("Score Time", input_scores["score_time"])

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
# PART 2-2: Cross-validation
#----------------------------------------

#Remove income from the list since it is the target variable
col_mod = col_names.copy()
col_mod.remove('income')
X = df_data[col_mod]
y = df_data['income']

#One-hot encode categorial variables
X_encoded = pd.get_dummies(X)

#Generate classifier object
gnb = GaussianNB()

#Cross validation
#https://neptune.ai/blog/cross-validation-in-machine-learning-how-to-do-it-right

scores = cross_val_score(gnb, X_encoded, y, cv=10)
print(f"Accuracy: {round(scores.mean(),2)} \u00B1 {round(scores.std(),2)}")

scores = cross_validate(gnb, X_encoded, y, cv=10, return_estimator=True,
                        return_train_score=True)
print_cross_metrics(scores, "all")
print_cross_metrics(scores, "average")

#----------------------------------------
# PART 2-3: Hyperparameter tuning
#----------------------------------------

#Generate classifier object
gnb = GaussianNB()

#Define cross-validation strategy
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)

#Define search space
space_grid = {}
space_grid["priors"] = [None]
space_grid["var_smoothing"] = np.logspace(-1,-12, num=100, base=10)

space_random = {}
space_random["priors"] = [None]
space_random["var_smoothing"] = np.logspace(-1,-12, num=100, base=10)

#Define search
search_grid = GridSearchCV(gnb, space_grid, scoring="accuracy", n_jobs=-1, 
                           cv=cv)
search_random = RandomizedSearchCV(gnb, space_random, scoring="accuracy", 
                                   n_jobs=-1, cv=cv, n_iter=10)

#Run grid search
result_grid = search_grid.fit(X_encoded, y)
print(f"\nResults (Grid Search):")
print(result_grid)
print(f"Best score: {result_grid.best_score_}")
print(f"Best parameters: {result_grid.best_params_}")

#Run random search
result_random = search_random.fit(X_encoded, y)
print(f"\nResults (Random Search):")
print(search_random)
print(f"Best score: {search_random.best_score_}")
print(f"Best parameters: {search_random.best_params_}")

"""
#Grid search overfit check
list_param = result_grid.cv_results_["param_var_smoothing"]
list_score = result_grid.cv_results_["mean_test_score"]

if len(list_param) == len(list_score):
    for x in range(len(list_param)):
        print(f"var_smoothing={list_param[x]}; score={list_score[x]}")
"""

#----------------------------------------
# PART 2-4: Overfit Check
#----------------------------------------

#Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

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

#Generate array of scores vs hyperparameters
scores_train, scores_test = [], []
for x in space_grid["var_smoothing"]:

    #Generate classifier object
    gnb_temp = GaussianNB(var_smoothing=x)

    #Train model
    gnb_temp.fit(X_train_encoded, y_train)

    #Make prediction
    y_pred = gnb_temp.predict(X_test_encoded)

    #Get accuracy scores for train and test
    scores_train.append(gnb_temp.score(X_train_encoded, y_train))
    scores_test.append(gnb_temp.score(X_test_encoded, y_test))

#Generate output dataframe
df_results = pd.DataFrame()
cols = ["var_smoothing", "train", "test", "slope_train", "slope_test"]
for x in range(len(space_grid["var_smoothing"])):
    if x == 0:
        val_list = [space_grid['var_smoothing'][x], scores_train[x],
                    scores_test[x], 0, 0]
        df_temp = build_overfit_df(cols, val_list)
        df_results = pd.concat([df_results, df_temp])
    else:
        dx = space_grid['var_smoothing'][x] - space_grid['var_smoothing'][x-1]
        dy_train = scores_train[x] - scores_train[x-1]
        dy_test = scores_test[x] - scores_test[x-1]
        slope_train = dy_train/dx
        slope_test = dy_test/dx
        val_list = [space_grid['var_smoothing'][x], scores_train[x],
                    scores_test[x], slope_train, slope_test]
        df_temp = build_overfit_df(cols, val_list)
        df_results = pd.concat([df_results, df_temp])

df_results = df_results.reset_index(drop=True)
pd.set_option('display.max_rows', None)
print(f"\nTrain vs Test Data:")
print(df_results)

#Generate graph to compare train and test
fig, ax = plt.subplots()

ax.plot(space_grid["var_smoothing"], scores_train, color="blue", marker='o')
ax.plot(space_grid["var_smoothing"], scores_test, color="green", marker='o')

ax.set_ylabel("score")
ax.set_xlabel("var_smoothing")

fig.set_size_inches(4,3)
fig.tight_layout()
fig.set_dpi(300)

plt.show()