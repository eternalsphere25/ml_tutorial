import copy
import numpy as np
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP
from scipy.stats import pointbiserialr, spearmanr
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler


class FeatureSelection:
    def __init__(self, input_df, target_var):
        self.df_data = input_df
        self.col_names = list(self.df_data.columns)
        self.target_var = target_var

    def remove_target_var(self):
        # Remove target variable from the list
        self.col_mod = copy.deepcopy(self.col_names)
        self.col_mod.remove(self.target_var)

    def split_test_train(self, size=0.3):
        # Split data between test and train data
        self.X = self.df_data[self.col_mod]
        self.y = self.df_data[self.target_var]
        self.X_train, self.X_test, self.y_train, self.y_test = (
            train_test_split(self.X, self.y, test_size=size, random_state=0))

    def standardize_data_scale(self):
        sc = StandardScaler()
        sc.fit(self.X_train)
        self.X_train = sc.transform(self.X_train)
        self.X_test = sc.transform(self.X_test)

    def calculate_correlation(self):
        # Add the target variable back in to calculate correlations
        df_corr = copy.deepcopy(self.X_train)
        df_corr[self.target_var] = self.y_train

        # Calculate correlation
        columns = df_corr.columns
        parameter, correlation, correlation_abs = [], [], []
        for x in columns:
            if x != self.target_var:
                if len(df_corr[x].unique()) <= 2:
                    r = spearmanr(df_corr[self.target_var],df_corr[x])[0]
                else:
                    r = pointbiserialr(df_corr[self.target_var],df_corr[x])[0]
                parameter.append(x)
                correlation.append(r)
                correlation_abs.append(abs(r))

        # Assemble dataframe based on calculated correlation scores
        self.df_param = pd.DataFrame({'parameter': parameter,
                                      'correlation': correlation,
                                      'abs_corr': correlation_abs})

    def print_correlation_metrics(self):
        print(f'\nCorrelation of variable with target "{self.target_var}":')
        print(self.df_param.sort_values(by="correlation", ascending=False))


class EvaluateModel:
    def __init__(self):
        pass

    def round_num(self, input_float, round_to):
        output = Decimal(input_float).quantize(
            Decimal('0.' + ('0'*round_to)), rounding=ROUND_HALF_UP)
        return output

    def calculate_confusion_matrix(self, input_y_test, input_y_pred):
        # Generate confusion matrix and related statistics
        self.cm = confusion_matrix(input_y_test, input_y_pred)

        #Calculate TP/FN/FP/TN
        if len(self.cm) == 2:
            self.tp = self.cm[1][1]
            self.fn = self.cm[1][0]
            self.fp = self.cm[0][1]
            self.tn = self.cm[0][0]
        else:
            self.tp = np.diag(self.cm)
            self.fn = self.cm.sum(axis=1) - np.diag(self.cm)
            self.fp = self.cm.sum(axis=0) - np.diag(self.cm)
            self.tn = np.sum(self.cm) - (self.fp + self.fn + self.tp)

    def calculate_metrics(self):
        # Calculate accuracy/precision/recall/F1 metrics
        if len(self.cm) == 2:
            total = self.tp+self.tn+self.fp+self.fn
            true_all = self.tp+self.tn
            self.accuracy = true_all/total
            self.precision = self.tp/(self.tp+self.fp)
            self.recall = self.tp/(self.tp+self.fn)
            self.f1 = 2*((self.precision*self.recall)/(
                self.precision+self.recall))
        else:
            total = self.tp[0]+self.tn[0]+self.fp[0]+self.fn[0]
            true_all = self.tp[0]+self.tn[0]
            self.accuracy = true_all/total
            self.precision = self.tp[0]/(self.tp[0]+self.fp[0])
            self.recall = self.tp[0]/(self.tp[0]+self.fn[0])
            self.f1 = 2*((self.precision*self.recall)/(
                self.precision+self.recall))

    def calculate_roc_auc(self, input_y_test, input_y_pred):
        # Calculate ROC AUC score
        self.fpr, self.tpr, self.threshold = roc_curve(
            input_y_test, input_y_pred)
        self.auc = roc_auc_score(input_y_test, input_y_pred)

    def print_metrics(self, round_to=4):
        print(f"\nAccuracy Summary:")
        print(f"- Accuracy on training data: "
              f"{self.round_num(self.acc_train, round_to)}")
        print(f"- Accuracy on test data: "
              f"{self.round_num(self.acc_test, round_to)}")

        print(f"\nDetailed Results:")
        print(f"- Accuracy: {self.round_num(self.accuracy, round_to)}")
        print(f"- Precision: {self.round_num(self.precision, round_to)}")
        print(f"- Recall: {self.round_num(self.recall, round_to)}")
        print(f"- F1: {self.round_num(self.f1, round_to)}")
        print(f"- ROC AUC: {self.round_num(self.auc, round_to)}")

    def run_k_fold_cross_validation(self, input_model, input_X, input_y):
        self.scores = cross_validate(
            input_model, input_X, input_y, cv=10, return_estimator=True,
            return_train_score=True)

    def print_cross_metrics_single(self, verbose=True):
        print(f"\nCross-Validation Results:")
        print(f"- Model: {self.scores['estimator']}")
        print(f"- Accuracy (Train): {self.scores['train_score']}")
        print(f"- Accuracy (Test): {self.scores['test_score']}")
        if verbose == True:
            print(f"- Fit Time: {self.scores['fit_time']}")
            print(f"- Score Time: {self.scores['score_time']}")

    def print_cross_metrics_average(self, verbose=True):
        print(f"\nCross-Validation Results:")
        print(f"- Model: {self.scores['estimator'][0]}")
        print(f"- Accuracy (Train): "
              f"{round(self.scores["train_score"].mean(),2)} "
              f"\u00B1 {round(self.scores["train_score"].std(),2)}")
        print(f"- Accuracy (Test): "
              f"{round(self.scores["test_score"].mean(),2)} "
              f"\u00B1 {round(self.scores["test_score"].std(),2)}")
        if verbose == True:
            print(f"- Fit Time: {round(self.scores["fit_time"].mean(),2)} "
                  f"\u00B1 {round(self.scores["fit_time"].std(),2)}")
            print(f"- Score Time: {round(self.scores["score_time"].mean(),2)} "
                  f"\u00B1 {round(self.scores["score_time"].std(),2)}")


class Model(FeatureSelection, EvaluateModel):
    def __init__(self, input_df, target_var, input_model):
        super().__init__(input_df, target_var)
        self.model = input_model

    def preprocess_data(self):
        self.remove_target_var()
        self.split_test_train()
        self.calculate_correlation()

    def train_model(self):
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)

    def evaluate_model(self):
        self.acc_train = self.model.score(self.X_train, self.y_train)
        self.acc_test = self.model.score(self.X_test, self.y_test)
        self.calculate_confusion_matrix(self.y_test, self.y_pred)
        self.calculate_metrics()
        self.calculate_roc_auc(self.y_test, self.y_pred)

    def validate_model(self):
        self.run_k_fold_cross_validation(self.model, self.X, self.y)