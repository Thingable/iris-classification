import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


def main():
    irides, train, test = build_datasets('../iris_data.csv')
    irides.plot()
    plt.show()

    models = []
    accuracy = []

    for column in train:
        if column is not 'class' and column is not 'zeros':
            train_data = pd.concat([train[column], train['zeros']], axis=1, keys=[column, 'zeros'])
            model = build_model(train_data.values, train['class'])
            models.append(model)

    for i, column in enumerate(test):
        if column is not 'class' and column is not 'zeros':
            test_data = pd.concat([test[column], test['zeros']], axis=1, keys=[column, 'zeros'])
            result = test_model(models[i], test_data, test['class'])
            accuracy.append((column, result))

    print(accuracy)

def build_datasets(file):
    irides = pd.read_csv(file)
    irides.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    irides = irides.assign(zeros=np.zeros(len(irides)))
    print(irides.head())
    train, test = train_test_split(irides, test_size=.33, random_state=1)
    train.reset_index()
    print(train.head())
    test.reset_index()
    return irides, train, test


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


if __name__ == '__main__':
    main()