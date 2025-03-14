import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display,clear_output
from scipy.stats import sem
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, 
    classification_report, confusion_matrix
)


'''
def KNN_accuracy(rep,labels):
    accuracy= []

    for random_state in np.random.randint(1,10000,50):
        X_train, X_test, y_train, y_test = train_test_split(rep, labels, test_size=0.2, random_state=random_state)

        # Define the model

        neigh = KNeighborsClassifier(n_neighbors = 5)

        #Train

        neigh.fit(X_train, y_train)

        acc = sum([neigh.predict(X_test) == y_test][0])/len(y_test)

        accuracy.append(acc)
    m_accuracy = np.mean(accuracy)
    var = sem(accuracy)

    
    return round(m_accuracy*100,2),round(var*100,2)
'''


def KNN_accuracy(rep, labels):
    """
    Evaluates K-Nearest Neighbors (KNN) accuracy on an imbalanced dataset 
    with multiple trials using stratified sampling.

    Parameters:
        rep (numpy.ndarray): Feature representations.
        labels (numpy.ndarray): Corresponding class labels.

    Returns:
        dict: Dictionary containing mean and standard error of key metrics.
    """
    accuracies = []
    f1_scores_macro, f1_scores_weighted = [], []
    precisions_macro, precisions_weighted = [], []
    recalls_macro, recalls_weighted = [],[]
    f1_scores_grade_1, f1_scores_grade_2 = [], []
    confusion_matrices = []
    class_reports = []

    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"Label Distribution: {dict(zip(unique_labels, counts))}")

    for random_state in np.random.randint(1, 10000, 50):
        X_train, X_test, y_train, y_test = train_test_split(
            rep, labels, test_size=0.2, random_state=random_state, stratify=labels  
        )

        # Define and train the KNN model
        neigh = KNeighborsClassifier(n_neighbors=5)
        neigh.fit(X_train, y_train)

        # Predict
        y_pred = neigh.predict(X_test)

        # Calculate overall metrics
        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)  # Macro (treats all classes equally)
        f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)  # Weighted (accounts for class imbalance)
        precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
        precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
        recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)

        # Store overall metrics
        accuracies.append(acc)
        f1_scores_macro.append(f1_macro)
        f1_scores_weighted.append(f1_weighted)
        precisions_macro.append(precision_macro)
        precisions_weighted.append(precision_weighted)
        recalls_macro.append(recall_macro)
        recalls_weighted.append(recall_weighted)

        # Compute classification report
        class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        # Extract F1 scores for Grade 1 and Grade 2
        f1_grade_1 = class_report.get("1", {}).get("f1-score", 0)
        f1_grade_2 = class_report.get("2", {}).get("f1-score", 0)
        f1_scores_grade_1.append(f1_grade_1)
        f1_scores_grade_2.append(f1_grade_2)

        # Store confusion matrix & classification report
        confusion_matrices.append(confusion_matrix(y_test, y_pred))
        #class_reports.append(class_report)

    metrics = {
        "Accuracy": (np.mean(accuracies), sem(accuracies)),
        "F1 Macro": (np.mean(f1_scores_macro), sem(f1_scores_macro)),
        "F1 Weighted": (np.mean(f1_scores_weighted), sem(f1_scores_weighted)),
        "F1 Grade 1": (np.mean(f1_scores_grade_1), sem(f1_scores_grade_1)),
        "F1 Grade 2": (np.mean(f1_scores_grade_2), sem(f1_scores_grade_2)),
        "Precision Macro": (np.mean(precisions_macro), sem(precisions_macro)),
        "Precision Weighted": (np.mean(precisions_weighted), sem(precisions_weighted)),
        "Recall Macro": (np.mean(recalls_macro), sem(recalls_macro)),
        "Recall Weighted": (np.mean(recalls_weighted), sem(recalls_weighted)),
        # "Confusion Matrix Mean": np.mean(confusion_matrices, axis=0).astype(int) if confusion_matrices else None,
    }

    # Print summary
    print(f"Accuracy: {metrics['Accuracy'][0]:.2%} ± {metrics['Accuracy'][1]:.2%}")
    print(f"F1 (Macro): {metrics['F1 Macro'][0]:.2%} ± {metrics['F1 Macro'][1]:.2%}")
    print(f"F1 (Weighted): {metrics['F1 Weighted'][0]:.2%} ± {metrics['F1 Weighted'][1]:.2%}")
    print(f"F1 (Grade 1): {metrics['F1 Grade 1'][0]:.2%} ± {metrics['F1 Grade 1'][1]:.2%}")
    print(f"F1 (Grade 2): {metrics['F1 Grade 2'][0]:.2%} ± {metrics['F1 Grade 2'][1]:.2%}")
    print(f"Precision (Macro): {metrics['Precision Macro'][0]:.2%} ± {metrics['Precision Macro'][1]:.2%}")
    print(f"Recall (Macro): {metrics['Recall Macro'][0]:.2%} ± {metrics['Recall Macro'][1]:.2%}")
    print("\n Mean Confusion Matrix:\n",  np.mean(confusion_matrices, axis=0).astype(int))
    
    return metrics

# def KNN_accuracy(rep, labels):
#     accuracies = []
#     f1_scores = []
#     precisions = []
#     recalls = []

#     for random_state in np.random.randint(1, 10000, 50):
#         X_train, X_test, y_train, y_test = train_test_split(rep, labels, test_size=0.2, random_state=random_state)

#         # Define the model
#         neigh = KNeighborsClassifier(n_neighbors=5)

#         # Train
#         neigh.fit(X_train, y_train)

#         # Predict
#         y_pred = neigh.predict(X_test)

#         # Calculate metrics
#         acc = accuracy_score(y_test, y_pred)
#         f1 = f1_score(y_test, y_pred, average='weighted')
#         precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
#         recall = recall_score(y_test, y_pred, average='weighted')

#         accuracies.append(acc)
#         f1_scores.append(f1)
#         precisions.append(precision)
#         recalls.append(recall)

#     # Calculate mean and standard error of the mean for each metric
#     m_accuracy = np.mean(accuracies)
#     var_accuracy = sem(accuracies)

#     m_f1 = np.mean(f1_scores)
#     var_f1 = sem(f1_scores)

#     m_precision = np.mean(precisions)
#     var_precision = sem(precisions)

#     m_recall = np.mean(recalls)
#     var_recall = sem(recalls)

#     print(f' Accuracy {round(m_accuracy * 100, 2)}, F1 {round(m_f1 * 100, 2)}, Precision {round(m_precision * 100, 2)}, Recall {round(m_recall * 100, 2)}')

#     return {
#         "accuracy": (round(m_accuracy * 100, 2), round(var_accuracy * 100, 2)),
#         "f1": (round(m_f1 * 100, 2), round(var_f1 * 100, 2)),
#         "precision": (round(m_precision * 100, 2), round(var_precision * 100, 2)),
#         "recall": (round(m_recall * 100, 2), round(var_recall * 100, 2))
#     }


def silhuoette(rep,labels):

    #umap = viz.umap(rep,dim = 10,scatter = False)
    umap = viz.pca(data = rep,n_components = min([512,len(rep[1])]), variance = 0.95)
        
    sil = metrics.silhouette_score(rep,labels, metric = "euclidean", n_jobs = -1)
    
    return sil