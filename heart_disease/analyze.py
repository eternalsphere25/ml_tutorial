import pandas as pd
import warnings
from modules.machine_learning import algorithms as algo
from pathlib import Path
from sklearn import svm
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


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
# PART 3: Run prediction models
#------------------------------------------------------------------------------

#----------------------------------------
# PART 3-1: Gaussian Naive Bayes
#----------------------------------------

# Generate classifier object
gnb = algo.Model(df_data, target_val, GaussianNB())

# Train model, make predictions, and calculate performance metrics
gnb.preprocess_data()
gnb.print_correlation_metrics()
gnb.train_model()
gnb.evaluate_model()
gnb.print_metrics()

# Run k-fold cross validation
gnb.validate_model()
gnb.print_cross_metrics_average()

#----------------------------------------
# PART 3-2: Decision Tree
#----------------------------------------

# Generate classifier object
tree = algo.Model(df_data, target_val,
                  DecisionTreeClassifier(criterion="gini", max_depth=None))

# Train model, make predictions, and calculate performance metrics
tree.preprocess_data()
tree.print_correlation_metrics()
tree.train_model()
tree.evaluate_model()
tree.print_metrics()

# Run k-fold cross validation
tree.validate_model()
tree.print_cross_metrics_average()

#----------------------------------------
# PART 3-3: Random Forest
#----------------------------------------

# Generate classifier object
rf = algo.Model(df_data, target_val, RandomForestClassifier())

# Train model, make predictions, and calculate performance metrics
rf.preprocess_data()
rf.print_correlation_metrics()
rf.train_model()
rf.evaluate_model()
rf.print_metrics()

# Run k-fold cross validation
rf.validate_model()
rf.print_cross_metrics_average()

#----------------------------------------
# PART 3-4: Logistic Regression
#----------------------------------------

# Generate classifier object
log_reg = algo.Model(df_data, target_val,
                     LogisticRegression(solver="liblinear"))

# Train model, make predictions, and calculate performance metrics
log_reg.preprocess_data()
log_reg.print_correlation_metrics()
log_reg.train_model()
log_reg.evaluate_model()
log_reg.print_metrics()

# Run k-fold cross validation
log_reg.validate_model()
log_reg.print_cross_metrics_average()


#----------------------------------------
# PART 3-5: K-Nearest Neighbors
#----------------------------------------

# Generate classifier object
knn = algo.Model(df_data, target_val, KNeighborsClassifier(n_neighbors=4))

# Train model, make predictions, and calculate performance metrics
knn.preprocess_data()
knn.print_correlation_metrics()
knn.train_model()
knn.evaluate_model()
knn.print_metrics()

# Run k-fold cross validation
knn.validate_model()
knn.print_cross_metrics_average()

#----------------------------------------
# PART 3-6: Support Vector Machine
#----------------------------------------

# Generate classifier object
s_vector = algo.Model(df_data, target_val, svm.SVC(kernel='linear'))

# Train model, make predictions, and calculate performance metrics
s_vector.preprocess_data()
s_vector.print_correlation_metrics()
s_vector.train_model()
s_vector.evaluate_model()
s_vector.print_metrics()

# Run k-fold cross validation
s_vector.validate_model()
s_vector.print_cross_metrics_average()

#----------------------------------------
# PART 4-7: Linear Regression
#----------------------------------------

try:
    # Generate classifier object
    lin_reg = algo.Model(df_data, target_val, LinearRegression())

    # Train model, make predictions, and calculate performance metrics
    lin_reg.preprocess_data()
    lin_reg.print_correlation_metrics()
    lin_reg.train_model()
    lin_reg.evaluate_model()
    lin_reg.print_metrics()

    # Run k-fold cross validation
    lin_reg.validate_model()
    lin_reg.print_cross_metrics_average()
except ValueError:
    print(f"\nWARNING: Linear Regression cannot be used for a mix of binary "
          f"and continuous targets!\nNOTICE: Linear Regression skipped; "
          f"proceeding to next machine learning algorithm...")

#----------------------------------------
# PART 4-8: K-Means Clustering
#----------------------------------------
"""
# Generate classifier object
kmeans = algo.Model(df_data, target_val, KMeans(n_clusters=50, n_init="auto"))

# Train model, make predictions, and calculate performance metrics
kmeans.preprocess_data()
kmeans.print_correlation_metrics()
kmeans.train_model()
kmeans.evaluate_model()
kmeans.print_metrics()

# Run k-fold cross validation
kmeans.validate_model()
kmeans.print_cross_metrics_average()
"""
#----------------------------------------
# PART 4-9: Gradient Boosting
#----------------------------------------

# Generate classifier object
grad_boost = algo.Model(df_data, target_val, GradientBoostingClassifier())

# Train model, make predictions, and calculate performance metrics
grad_boost.preprocess_data()
grad_boost.print_correlation_metrics()
grad_boost.train_model()
grad_boost.evaluate_model()
grad_boost.print_metrics()

# Run k-fold cross validation
grad_boost.validate_model()
grad_boost.print_cross_metrics_average()

#----------------------------------------
# PART 4-10: Gradient Boosting (XGBoost)
#----------------------------------------

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)

    # Generate classifier object
    xg_boost = algo.Model(df_data, target_val,
                          XGBClassifier(silent=True, verbosity=0))

    # Train model, make predictions, and calculate performance metrics
    xg_boost.preprocess_data()
    xg_boost.print_correlation_metrics()
    xg_boost.train_model()
    xg_boost.evaluate_model()
    xg_boost.print_metrics()

    # Run k-fold cross validation
    xg_boost.validate_model()
    xg_boost.print_cross_metrics_average()

#----------------------------------------
# PART 4-11: Gradient Boosting (AdaBoost)
#----------------------------------------

# Generate classifier object
ada_boost = algo.Model(df_data, target_val,
                       AdaBoostClassifier(n_estimators=100))

# Train model, make predictions, and calculate performance metrics
ada_boost.preprocess_data()
ada_boost.print_correlation_metrics()
ada_boost.train_model()
ada_boost.evaluate_model()
ada_boost.print_metrics()

# Run k-fold cross validation
ada_boost.validate_model()
ada_boost.print_cross_metrics_average()

# Generate classifier object
# NOTE: For AdaBoost-SVM, a separate SVM classifier object needs to be made
ada_boost_svm = algo.Model(
    df_data, target_val, AdaBoostClassifier(
        estimator=s_vector.model, n_estimators=100))

# Train model, make predictions, and calculate performance metrics
ada_boost_svm.preprocess_data()
ada_boost_svm.print_correlation_metrics()
ada_boost_svm.train_model()
ada_boost_svm.evaluate_model()
ada_boost_svm.print_metrics()

# Run k-fold cross validation
ada_boost_svm.validate_model()
ada_boost_svm.print_cross_metrics_average()

# Generate object and train model (GNB)
# NOTE: For AdaBoost-GNB, a separate GNB classifier object needs to be made
ada_boost_gnb = algo.Model(
    df_data, target_val, AdaBoostClassifier(
        estimator=gnb.model, n_estimators=100))

# Train model, make predictions, and calculate performance metrics
ada_boost_gnb.preprocess_data()
ada_boost_gnb.print_correlation_metrics()
ada_boost_gnb.train_model()
ada_boost_gnb.evaluate_model()
ada_boost_gnb.print_metrics()

# Run k-fold cross validation
ada_boost_gnb.validate_model()
ada_boost_gnb.print_cross_metrics_average()