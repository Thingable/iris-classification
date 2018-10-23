"""
This experiment seeks to answer how the size of the training set affects the results of a Gaussian Naive Bayes classifier.

"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


def main():
    results = []
    for i in range(0, 100):
        row = []
        for j in np.arange(.05, .95, .01):
            train, test = build_datasets('../iris_data.csv', j, i)

            train_targets = train['class']
            train_data = train.drop(columns=['class'])
            model = build_model(train_data, train_targets)

            test_targets = test['class']
            test_data = test.drop(columns=['class'])
            result = test_model(model, test_data, test_targets)
            row.append(result)
        results.append(row)
    results = pd.DataFrame(results, columns=np.arange(.05, .95, .01))
    print(results)
    plot_results(results)


def build_datasets(file, test_size, random_state):
    irides = pd.read_csv(file)
    irides.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    train, test = train_test_split(irides, test_size=test_size, random_state=random_state)
    return train, test


def build_model(train_data, targets):
    model = GaussianNB()
    model.fit(train_data, targets)
    return model


def test_model(model, test_data, targets):
    predictions = model.predict(test_data)
    j = 0
    for i, target in enumerate(targets):
        if target == str(predictions[i]):
            j = j+1
    return j/len(targets)


def plot_results(results):
    plt.figure()
    results.boxplot()
    plt.show()



if __name__ == '__main__':
    main()