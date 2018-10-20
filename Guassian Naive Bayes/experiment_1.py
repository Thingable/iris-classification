"""
This experiment seeks to answer which independent variable, when modeled with the Gaussian Naive Bayes classifier,
produces the most accurate classification.

The iris data (found in iris_data.csv) was split into 1000 random selections of 2/3 training data and 1/3 test data. These
data sets were then modeled and tested and compared against each other.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


def main():
    accuracy = pd.DataFrame(columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

    for i in range(0, 1000):
        train, test = build_datasets('../iris_data.csv', i)
        models = []

        for column in train:
            if column is not 'class' and column is not 'zeros':
                train_data = pd.concat([train[column], train['zeros']], axis=1, keys=[column, 'zeros'])
                model = build_model(train_data.values, train['class'])
                models.append(model)

        results = []
        for j, column in enumerate(test):
            if column is not 'class' and column is not 'zeros':
                test_data = pd.concat([test[column], test['zeros']], axis=1, keys=[column, 'zeros'])
                result = test_model(models[j], test_data, test['class'])
                results.append(result)
        accuracy.loc[i] = results
    print(accuracy.head())
    plot_accuracy(accuracy)
    stats(accuracy)


def build_datasets(file, random_state):
    irides = pd.read_csv(file)
    irides.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    irides = irides.assign(zeros=np.zeros(len(irides)))
    train, test = train_test_split(irides, test_size=.33, random_state=random_state)

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


def stats(accuracy):
    stats = pd.DataFrame(columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
                         index=['mean', 'mode', 'median', 'std', 'min', 'max'])
    mean = []
    median = []
    mode = []
    std = []
    min = []
    max = []
    for i, column in enumerate(accuracy):
        mean.append(accuracy[column].mean())
        median.append(accuracy[column].median())
        mode.append(accuracy[column].mode()[0])
        std.append(accuracy[column].std())
        min.append(accuracy[column].min())
        max.append(accuracy[column].max())

    stats.loc['mean'] = mean
    stats.loc['median'] = median
    stats.loc['mode'] = mode
    stats.loc['std'] = std
    stats.loc['min'] = min
    stats.loc['max'] = max

    x = stats.reset_index()
    print(x.to_string(index=None))


def plot_accuracy(accuracy):
    plt.figure()
    accuracy.plot.hist(bins=30, alpha=0.5)
    plt.show()


if __name__ == '__main__':
    main()