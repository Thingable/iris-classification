import pandas as pd
import matplotlib.pyplot as plt

def main():
    irides = pd.read_csv('iris_data.csv')
    irides.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

    setosa = irides[irides['class'] == 'Iris-setosa']
    versicolor = irides[irides['class'] == 'Iris-versicolor']
    virginica = irides[irides['class'] == 'Iris-virginica']

    ax = setosa.plot.scatter(x='sepal_length', y='petal_width', c='black')
    versicolor.plot.scatter(x='sepal_length', y='petal_width', c='red', ax=ax)
    virginica.plot.scatter(x='sepal_length', y='petal_width', c='green', ax=ax)
    plt.show()



if __name__ == '__main__':
    main()
