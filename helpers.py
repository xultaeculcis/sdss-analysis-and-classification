# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
import warnings
import matplotlib as mpl
import os

# Common imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Imports for ML
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

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

# Helper functions and structures
# Ignore useless warnings (see SciPy issue #5998)
warnings.filterwarnings(action="ignore", message="^internal gelsd")
IMAGES_PATH = "img"


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def cv_learn(clf, x_train, y_train, x_test, y_test,
             cv_size=5, scoring_type="accuracy"):
    """
    Performs classifier learning process and validates the results.

    :param clf: The classifier.
    :param x_train: The training set.
    :param y_train: The labels for training set.
    :param x_test: The test set.
    :param y_test: The labels for test set.
    :param cv_size: Cross-Validation size. Default: 5.
    :param scoring_type: Scoring type for Cross-Val-Score. Default: accuracy.
    :return: Tuple with cross-val scores, predicted labels, confusion matrix, classification report (str. and dict.).
    """
    # 5xCV
    y_scores = cross_val_score(clf, x_train, y_train,
                               cv=cv_size, scoring=scoring_type)
    # Test Prediction
    pred = cross_val_predict(clf, x_test, y_test, cv=cv_size)

    # Conf. Matrix
    matrix = confusion_matrix(y_test, pred)

    # CLF. Report
    report_str = classification_report(y_test, pred)
    report_dict = classification_report(y_test, pred, output_dict=True)

    return y_scores, pred, matrix, report_str, report_dict


def print_cv_scores(scores):
    """
    Prints the cross-val acc. score.

    :param scores: The cross_val_score function result.
    :return:
    """
    print("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))


def print_conf_matrix(clf_name, matrix):
    """
    Prints the confusion matrix for specified classifier.

    :param clf_name: The classifier name.
    :param matrix: The confusion matrix.
    :return:
    """
    print(clf_name, ":\n", matrix)


def plot_confusion_matrix(clf_name, class_names, cm, figsize):
    """
    Plots the confusion matrix for specified classfier.

    :param clf_name: The classifier name.
    :param class_names: The sorted list (alphabetically) of labels for the predicted classes.
    :param cm: The confusion matrix.
    :param figsize: The size of the plot. e.g. figsize=(10,5)
    :return:
    """
    df_cm = pd.DataFrame(cm, index=[i for i in class_names],
                         columns=[i for i in class_names])
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax, fmt="d"); #annot=True to annotate cells

    # labels, title and ticks
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix for: ' + clf_name, fontsize=15); 
    ax.xaxis.set_ticklabels(class_names); ax.yaxis.set_ticklabels(class_names);
    plt.show()

def print_learning_results_multiple(classifs, class_names, scores, matrices, reports):
    """
    Prints learning results for multiple classifiers.

    :param classifs: The classifiers.
    :param class_names: The sorted list (alphabetically) of labels for the predicted classes.
    :param scores: List of results from the cross_val_score() function for each of classifiers.
    :param matrices: The confusion matrices.
    :param reports: The classification reports.
    :return:
    """
    for item in zip(classifs, scores, matrices, reports):
        print(item[0])
        print(item[1])
        print_cv_scores(item[1])
        print_conf_matrix(item[0], item[2])
        # Plot non-normalized confusion matrix
        plot_confusion_matrix(item[0], class_names, item[2], (5, 5))
        print(item[3])
        print("\n")


def print_learning_results_single(clf, class_names, scores, matrix, report):
    """
    Prints learning results for single classifier.

    :param clf: The classifier.
    :param class_names: The sorted list (alphabetically) of labels for the predicted classes.
    :param scores: Results from the cross_val_score() function for specified classifier.
    :param matrix: The confusion matrix.
    :param report: The classification report.
    :return:
    """
    #print(clf)
    print(scores)
    print_cv_scores(scores)
    print_conf_matrix(clf, matrix)
    # Plot non-normalized confusion matrix
    plot_confusion_matrix(clf, class_names, matrix, (5, 5))
    print(report)
    print("\n")


class ResultSet:
    """
    Represents result set for a single classifier.
    """

    def __init__(self, clf_name, scores, predictions, matrix, report_str, report_dict):
        """
        Initializes new instance of the ResultSet class.

        :param clf_name: The classifier's name.
        :param scores: Results from the cross_val_score() function for specified classifier.
        :param predictions: Predicted labels.
        :param matrix: The confusion matrix.
        :param report_str: The classification report represented as a string.
        :param report_dict: The classification report represented as a dictionary.
        """
        self.classifier_name = clf_name
        self.scores = scores
        self.predictions = predictions
        self.matrix = matrix
        self.report_str = report_str
        self.report_dict = report_dict


def print_kv_arr(ordered_by, arr):
    """
    Prints the array of key-value pairs, with specified ordering.

    :param ordered_by: The ordering name.
    :param arr: The array of key-value pairs.
    :return:
    """
    print(ordered_by)
    for kv in arr:
        print(kv[0], kv[1])
    print("\n")


def roc_auc_score_multiclass(actual_class, pred_class, average="macro"):
    """
    Computes ROC AUC score for multi-class classification.

    :param actual_class: The true class array.
    :param pred_class: The predicted class array.
    :param average: Type of the averaging. Default: "macro".
    :return: The dictionary with scores for each class.
    """
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
    """
    Performs ordering of the results in the result_sets array.


    :param result_sets: The ResultSet array.
    :return:
    """
    by_acc = {}

    by_prec_micro = {}
    by_recall_micro = {}
    by_f1_micro = {}

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


def train_classif_single(clf, clf_name, class_names, x_train, y_train, x_test, y_test, result_sets):
    """
    Performs learning process for a single classifier.

    :param clf: The classifier.
    :param clf_name: The classifier name.
    :param class_names: The sorted list (alphabetically) of labels for the predicted classes.
    :param x_train: The training set.
    :param y_train: The labels for training set.
    :param x_test: The test set.
    :param y_test: The labels for test set.
    :param result_sets: The result set array to which the results will be saved.
    :return:
    """
    try:
        scores, predictions, matrix, report_str, report_dict = cv_learn(
            clf, x_train, y_train, x_test, y_test)

        # append results for later use
        result_sets.append(
            ResultSet(clf_name, scores, predictions, matrix, report_str, report_dict))

        # print results
        print_learning_results_single(
            clf_name, class_names, scores, matrix, report_str)
    except:
        print("Something bad had happened... Could not proceed for the: ", clf_name, "\n")


def train_classif_multiple(clfs, clf_names, class_names, x_train, y_train, x_test, y_test, result_sets):
    """
    Performs learning process for multiple classifiers.

    :param clfs: The classifiers.
    :param clf_names: The classifier names.
    :param class_names: The sorted list (alphabetically) of labels for the predicted classes.
    :param x_train: The training set.
    :param y_train: The labels for training set.
    :param x_test: The test set.
    :param y_test: The labels for test set.
    :param result_sets: The result set array to which the results will be saved.
    :return:
    """
    for clf in zip(clfs, clf_names):
        train_classif_single(clf[0], clf[1], class_names, x_train, y_train, x_test, y_test, result_sets)

    order_results(result_sets)
    print_roc_auc_scores(result_sets, y_test)


def print_roc_auc_scores(result_sets, y_true):
    """
    Prints the ROC AUC scores.

    :param result_sets: The array of the ResultSet class instances.
    :param y_true:
    :return:
    """
    for result_set in result_sets:
        print(result_set.classifier_name)
        # assuming your already have a list of actual_class and predicted_class
        score = roc_auc_score_multiclass(y_true, result_set.predictions)
        print(score)
        print("\n")


class DataSet:
    """
    Represents the data set on which the learning process will take place.
    """

    def __init__(self, data_set_type, x_train, y_train, x_test, y_test):
        """

        :param data_set_type: The name of the data set type e.g. "StdScaled", "MinMaxScaled", "MaxAbsScaled".
        :param x_train: The training set.
        :param y_train: The labels for training set.
        :param x_test: The test set.
        :param y_test: The labels for test set.
        """
        self.type = data_set_type
        self.X_train = x_train
        self.X_test = x_test
        self.y_train = y_train
        self.y_test = y_test


def learning_loop_for_sets(clfs, clf_names, class_names, data_sets):
    """
    Performs learning of multiple classifiers on multiple data-sets.

    :param clfs: The classifiers.
    :param clf_names: The classifier names.
    :param class_names: The sorted list (alphabetically) of labels for the predicted classes.
    :param data_sets: The data sets.
    :return:
    """
    for data_set in data_sets:
        print("==========================================================")
        print("==========================================================")
        print("Data Set Type: ", data_set.type)
        result_sets = []
        train_classif_multiple(clfs, clf_names, class_names,
                               data_set.X_train, data_set.y_train, data_set.X_test, data_set.y_test, result_sets)
        
    return result_sets
