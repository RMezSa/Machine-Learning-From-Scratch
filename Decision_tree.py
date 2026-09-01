import numpy as np
import pandas as pd

def csv_prueba():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    columnas = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'especie']
    df = pd.read_csv(url, names=columnas)
    # ignoramos 'Iris-virginica'
    df = df[df['especie'].isin(['Iris-versicolor', 'Iris-virginica'])]
    # Mapeamos a 0 (Negativo) y 1 (Positivo)
    mapeo_especies = {
        'Iris-versicolor': 0,
        'Iris-virginica': 1
    }
    df['especie'] = df['especie'].map(mapeo_especies)
    # Extraemos características (X) y etiquetas (y)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    #print(df.isnull().sum())
    
    return X, y

def dataset(filename):
    df = pd.read_csv(filename)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    return X, y

def split_dataset(X, y, test_size = .2, random_seed=42):
    np.random.seed(random_seed)
    idx = np.random.permutation(len(X))
    corte = int(len(X)*(1-test_size)) #80%
    train_idx, test_idx = idx[:corte], idx[corte:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx] 


class TreeNode:
    def __init__(self, threshold=None, feature=None, left=None, right=None, valor=None):
        self.left = left
        self.right = right
        self.threshold = threshold
        self.feature = feature
        self.valor = valor

    def es_hoja(self):
        return self.valor is not None
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
        #print("Probabilidades \n")
        #print(probabilidades)

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
                mascara_izq = X[:,feature] <= threshold
                mascara_der = X[:,feature] > threshold

                y_izq = y[mascara_izq]
                y_der = y[mascara_der]

                print(y_izq.shape)
                print(y_der.shape)

                if y_izq.size == 0 or y_der.size == 0:
                    continue

                gini_izq = self.gini(y_izq)
                gini_der = self.gini(y_der)
                n_izq = len(y_izq)
                n_der = len(y_der)
                n_total = len(y)

                gini_ponderado = (n_izq / n_total)* gini_izq + (n_der / n_total)*gini_der


                #print(f"Gini ponderado: {gini_ponderado}")

                if  gini_ponderado < mejor_gini:
                    mejor_gini = gini_ponderado
                    mejor_feature = feature
                    mejor_threshold = threshold

                print(f" mejor gini: {mejor_gini}")
                #print(f"mejor feature: {mejor_feature}")
                #print(f" mejor threshold: {mejor_threshold}")

        return mejor_threshold, mejor_feature, mejor_gini


    def construir_arbol(self, X, y, profundidad = 0):
        #parar
        if self.gini(y) == 0 or profundidad > self.max_depth or len(y) < self.min_samples_split:
            valores, conteos = np.unique(y, return_counts=True)
            indice_mayor = np.argmax(conteos)
            valor_hoja = valores[indice_mayor]
            return TreeNode(valor=valor_hoja)
        #seguir partiendo
        threshold, feature, _ = self.mejor_split(X, y)
        if feature is None:
            valores, conteos = np.unique(y, return_counts=True)
            indice_mayor = np.argmax(conteos)
            valor_hoja = valores[indice_mayor]
            return TreeNode(valor=valor_hoja)

        mascara_izq = X[:,feature] <= threshold
        mascara_der = X[:, feature] > threshold

        izquierda = self.construir_arbol(X[mascara_izq], y[mascara_izq], profundidad+1)
        derecha = self.construir_arbol(X[mascara_der], y[mascara_der], profundidad+1)

        return TreeNode(feature=feature, threshold=threshold, left=izquierda, right=derecha)

    def fit(self, X, y):
        self.root = self.construir_arbol(X, y)

    def predict_row(self, row):
        assert self.root is not None, "Llamar fit antes que predict"
        nodo = self.root
        while not nodo.es_hoja():
            if row[nodo.feature] <= nodo.threshold:
                nodo = nodo.left
            else:
                nodo = nodo.right
        return nodo.valor
            
    def predict(self, X):
        predicciones  = []
        for i in X:
            predicciones.append(self.predict_row(i))
        return np.array(predicciones)

    def imprimir_arbol(self, nodo=None, espacio=""):
        if nodo is None:
            nodo = self.root
            if nodo is None:
                print("no hay arbol")
                return

        if nodo.es_hoja():
            print(f"Hoja: {nodo.valor}")
            return

        print(f" feature {nodo.feature} <= threshold {nodo.threshold}")

        print(f"{espacio} |-- SI - ", end="")
        self.imprimir_arbol(nodo.left, espacio + " |   ")

        print(f"{espacio} |--- NO - ", end="")
        self.imprimir_arbol(nodo.right, espacio + "    ")

def main():
    X, y = csv_prueba() 

    print("X")
    print(type(X))
    print(X.shape)

    print("y")
    print(type(y))
    print(y.shape)

    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=.2)

    arbol = DecisionTree(min_samples_split=2, max_depth=6)
    arbol.fit(X_train, y_train)

    predicciones = arbol.predict(X_test)
    print("predicciones:     ", predicciones)
    print("etiquetas reales: ", y_test)

    for i in range(0, predicciones.size - 1):
        print(i, y_test[i] == predicciones[i])


    print("--Arbol--")
    arbol.imprimir_arbol()

if __name__ == "__main__":
    main()