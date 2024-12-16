from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import accuracy_score
import numpy as np
import os
from datetime import datetime

def load_datasets(input_path: str):
    files = [f for f in os.listdir(input_path)]
    data_sets = []
    for file in files:
        data  = pd.read_csv(os.path.join(input_path, file), header=None, delimiter=',')
        x = data.iloc[:,:-1]
        y = data.iloc[:, -1]
        data_sets.append((x, y))
    return data_sets

def run_kfold(clf, X_all, y_all):
    rkf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)
    outcomes = []
    fold = 0
    for train_index, test_index in rkf.split(X_all):
        fold += 1
        X_train, X_test = X_all.values[train_index], X_all.values[test_index]
        y_train, y_test = y_all.values[train_index], y_all.values[test_index]
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        outcomes.append(accuracy)
    mean = np.mean(outcomes)
    std = np.std(outcomes)
    return mean, std



classifiers = [
    AdaBoostClassifier(algorithm="SAMME", random_state=42),
    GaussianNB(),
]
input_file = '/home/student/lab1/datasets/'
    
        

if __name__ == "__main__":
    start_time = datetime.now()
    sets = load_datasets(input_file)
    results = []
    for clf in classifiers:
        for x, y in sets:
            results.append(run_kfold(clf, x, y))
    print(results)
    print(f'Elapsed time: {datetime.now() - start_time}')