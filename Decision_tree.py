import numpy as np
import pandas as pd

def csv_prueba():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    columnas = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'especie']
    df = pd.read_csv(url, names=columnas)
    # ignoramos 'Iris-virginica'
    df = df[df['especie'].isin(['Iris-setosa', 'Iris-versicolor'])]
    # Mapeamos a 0 (Negativo) y 1 (Positivo)
    mapeo_especies = {
        'Iris-setosa': 0,
        'Iris-versicolor': 1
    }
    df['especie'] = df['especie'].map(mapeo_especies)
    # Extraemos características (X) y etiquetas (y)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    print(df.isnull().sum())
    
    return X, y

def dataset(filename):
    df = pd.read_csv(filename)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    return X, y

class DecisionTree:
    def __init__(self, min_samples_split, max_depth):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    def gini(self, y):
        """ Impureza de Gini para un arreglo"""
        """ 1 - sum(p_i²) """
        m = len(y)

        if m == 0:
            return 0.0

        _, counts = np.unique(y, return_counts=True)
        probabilidades = counts/m
        print("Probabilidades \n")
        print(probabilidades)

        gini = 1.0 - np.sum(probabilidades**2)

        return gini

    def mejor_split(self, X, y):
        n_samples, n_features = X.shape
        mejor_gini = float('inf')
        mejor_feature = None
        mejor_threshold = None

        for feature in range(n_features):
            valores_unicos = np.unique(X[:, feature])
            thresholds = (valores_unicos[:-1] + valores_unicos[1:]) / 2

            for threshold in thresholds:
                mascara_izq = X[:feature] <= threshold
                mascara_der = X[:feature] > threshold

                y_izq = y[mascara_izq]
                y_der = y[mascara_der]

                print(y_izq.shape)
                print(y_der.shape)
                


def main():
    X, y = csv_prueba() 

    print("X")
    print(type(X))
    print(X.shape)

    print("y")
    print(type(y))
    print(y.shape)

    arbol = DecisionTree(2, 5)
    gini = arbol.gini(y)

    split = arbol.mejor_split(X, y)

    print(gini)
    print(split)



if __name__ == "__main__":
    main()