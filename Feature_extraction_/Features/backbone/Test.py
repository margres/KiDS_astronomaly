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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score




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
    accuracies = []
    f1_scores = []
    precisions = []
    recalls = []

    for random_state in np.random.randint(1, 10000, 50):
        X_train, X_test, y_train, y_test = train_test_split(rep, labels, test_size=0.2, random_state=random_state)

        # Define the model
        neigh = KNeighborsClassifier(n_neighbors=5)

        # Train
        neigh.fit(X_train, y_train)

        # Predict
        y_pred = neigh.predict(X_test)

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted')

        accuracies.append(acc)
        f1_scores.append(f1)
        precisions.append(precision)
        recalls.append(recall)

    # Calculate mean and standard error of the mean for each metric
    m_accuracy = np.mean(accuracies)
    var_accuracy = sem(accuracies)

    m_f1 = np.mean(f1_scores)
    var_f1 = sem(f1_scores)

    m_precision = np.mean(precisions)
    var_precision = sem(precisions)

    m_recall = np.mean(recalls)
    var_recall = sem(recalls)

    print(f' Accuracy {round(m_accuracy * 100, 2)}, F1 {round(m_f1 * 100, 2)}, Precision {round(m_precision * 100, 2)}, Recall {round(m_recall * 100, 2)}')

    return {
        "accuracy": (round(m_accuracy * 100, 2), round(var_accuracy * 100, 2)),
        "f1": (round(m_f1 * 100, 2), round(var_f1 * 100, 2)),
        "precision": (round(m_precision * 100, 2), round(var_precision * 100, 2)),
        "recall": (round(m_recall * 100, 2), round(var_recall * 100, 2))
    }


def silhuoette(rep,labels):

    #umap = viz.umap(rep,dim = 10,scatter = False)
    umap = viz.pca(data = rep,n_components = min([512,len(rep[1])]), variance = 0.95)
        
    sil = metrics.silhouette_score(rep,labels, metric = "euclidean", n_jobs = -1)
    
    return sil