# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
import warnings
import helpers
import matplotlib as mpl
import os

# Common imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import operator
from collections import OrderedDict

# Imports for ML
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier, PassiveAggressiveClassifier, RidgeClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import PolynomialFeatures, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
sns.set_style('whitegrid')

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "results"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)

# Helper functioins and structures
# Ignore useless warnings (see SciPy issue #5998)
warnings.filterwarnings(action="ignore", message="^internal gelsd")
IMAGES_PATH = "img"

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

def cv_learn(clf, X_train, y_train, X_test, y_test,
             cv_size=5, scoring_type="accuracy", average_type="macro"):
    # 5xCV
    y_scores = cross_val_score(clf, X_train, y_train,
                               cv=cv_size, scoring=scoring_type)
    # Test Prediction
    pred = cross_val_predict(clf, X_test, y_test, cv=cv_size)

    # Conf. Matrix
    matrix = confusion_matrix(y_test, pred)

    # CLF. Report
    report_str = classification_report(y_test, pred)
    report_dict = classification_report(y_test, pred, output_dict=True)

    return (y_scores, pred, matrix, report_str, report_dict)


def print_cv_scores(scores):
    print("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))


def print_conf_matrix(clf_name, matrix):
    print(clf_name, ":\n", matrix)


def plot_confusion_matrix(clf_name, class_names, cm, figsize):
    df_cm = pd.DataFrame(cm, index=[i for i in class_names],
                         columns=[i for i in class_names])
    plt.figure(figsize=figsize)
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="jet").set_title(
        "Confusion Matrix for: " + str(clf_name))


def print_learning_results_multiple(classifs, class_names, scores, matrices, reports):
    for item in zip(classifs, scores, matrices, reports):
        print(item[0])
        print(item[1])
        print_cv_scores(item[1])
        print_conf_matrix(item[0], item[2])
        # Plot non-normalized confusion matrix
        plot_confusion_matrix(item[0], class_names, item[2], (10, 7))
        print(item[3])
        print("\n")


def print_learning_results_single(clf, class_names, scores, matrix, report):
    print(clf)
    print(scores)
    print_cv_scores(scores)
    print_conf_matrix(clf, matrix)
    # Plot non-normalized confusion matrix
    plot_confusion_matrix(clf, class_names, matrix, (10, 7))
    print(report)
    print("\n")


class_names = ["GALAXY", "QSO", "STAR"]


class ResultSet:
    def __init__(self, clf_name, scores, predictions, matrix, report_str, report_dict):
        self.classifier_name = clf_name
        self.scores = scores
        self.predictions = predictions
        self.matrix = matrix
        self.report_str = report_str
        self.report_dict = report_dict


def print_kv_arr(ordered_by, arr):
    print(ordered_by)
    for kv in arr:
        print(kv[0], kv[1])
    print("\n")


def roc_auc_score_multiclass(actual_class, pred_class, average="macro"):
    # creating a set of all the unique classes using the actual class list
    unique_class = set(actual_class)
    roc_auc_dict = {}
    for per_class in unique_class:
        # creating a list of all the classes except the current class
        other_class = [x for x in unique_class if x != per_class]

        # marking the current class as 1 and all other classes as 0
        new_actual_class = [0 if x in other_class else 1 for x in actual_class]
        new_pred_class = [0 if x in other_class else 1 for x in pred_class]

        # using the sklearn metrics method to calculate the roc_auc_score
        roc_auc = roc_auc_score(
            new_actual_class, new_pred_class, average=average)
        roc_auc_dict[per_class] = roc_auc

    return roc_auc_dict


def order_results(result_sets):
    by_acc = {}
    by_prec_micro = {}
    by_recall_micro = {}
    by_f1_micro = {}

    by_acc_macro = {}
    by_prec_macro = {}
    by_recall_macro = {}
    by_f1_macro = {}

    for result_set in result_sets:
        by_acc[result_set.classifier_name] = result_set.scores.mean()

        micro_results = result_set.report_dict["micro avg"]
        macro_results = result_set.report_dict["macro avg"]

        by_prec_micro[result_set.classifier_name] = micro_results["precision"]
        by_prec_macro[result_set.classifier_name] = macro_results["precision"]

        by_recall_micro[result_set.classifier_name] = micro_results["recall"]
        by_recall_macro[result_set.classifier_name] = macro_results["recall"]

        by_f1_micro[result_set.classifier_name] = micro_results["f1-score"]
        by_f1_macro[result_set.classifier_name] = macro_results["f1-score"]

    sorted_acc = sorted(by_acc.items(), key=lambda kv: kv[1], reverse=True)

    sorted_prec_micro = sorted(by_prec_micro.items(), key=lambda kv: kv[1], reverse=True)
    sorted_recall_micro = sorted(by_recall_micro.items(), key=lambda kv: kv[1], reverse=True)
    sorted_f1_micro = sorted(by_f1_micro.items(), key=lambda kv: kv[1], reverse=True)

    sorted_prec_macro = sorted(by_prec_macro.items(), key=lambda kv: kv[1], reverse=True)
    sorted_recall_macro = sorted(by_recall_macro.items(), key=lambda kv: kv[1], reverse=True)
    sorted_f1_macro = sorted(by_f1_macro.items(), key=lambda kv: kv[1], reverse=True)

    print_kv_arr("By Acc:", sorted_acc)
    print_kv_arr("By Precision(avg=micro):", sorted_prec_micro)
    print_kv_arr("By Recall(avg=micro):", sorted_recall_micro)
    print_kv_arr("By F1(avg=micro):", sorted_f1_micro)
    print_kv_arr("By Precision(avg=macro):", sorted_prec_macro)
    print_kv_arr("By Recall(avg=macro):", sorted_recall_macro)
    print_kv_arr("By F1(avg=macro):", sorted_f1_macro)


def train_classif_single(clf, clf_name, class_names, X_train, y_train, X_test, y_test, result_sets):
    try:
        scores, predictions, matrix, report_str, report_dict = cv_learn(
        clf, X_train, y_train, X_test, y_test)

        # append results for later use
        result_sets.append(
            ResultSet(clf_name, scores, predictions, matrix, report_str, report_dict))

        # print results
        print_learning_results_single(
            clf_name, class_names, scores, matrix, report_str)
    except:
        print("Something bad had happened... Could not proceed for the: ", clf_name, "\n")


def train_classif_multiple(clfs, clf_names, class_names, X_train, y_train, X_test, y_test, result_sets):
    for clf in zip(clfs, clf_names):
        train_classif_single(clf[0], clf[1], class_names, X_train, y_train, X_test, y_test, result_sets)

    order_results(result_sets)
    print_roc_auc_scores(result_sets, y_test)
        

def print_roc_auc_scores(result_sets, y_true):
    for result_set in result_sets:
        print(result_set.classifier_name)
        # assuming your already have a list of actual_class and predicted_class
        score = roc_auc_score_multiclass(y_true, result_set.predictions)
        print(score)
        print("\n")

class DataSet:
    def __init__(self, type, X_train, y_train, X_test, y_test):
        self.type = type
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test


def learining_loop_for_sets(clfs, clf_names, class_names, data_sets):
    for data_set in data_sets:
        print("==========================================================")
        print("==========================================================")
        print("Data Set Type: ", data_set.type)
        result_sets = []
        train_classif_multiple(clfs, clf_names, class_names,
                               data_set.X_train, data_set.y_train, data_set.X_test, data_set.y_test, result_sets)
