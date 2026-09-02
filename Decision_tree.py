import numpy as np
import pandas as pd

def csv_prueba():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    columnas = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'especie']
    df = pd.read_csv(url, names=columnas)
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

def csv_multiclase():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
    df = pd.read_csv(url, header=None)

    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values

    return X, y

def csv_cancer_mama():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    df = pd.read_csv(url, header=None)
    df = df.drop(0, axis=1)
    df[1] = df[1].map({'M':1, 'B':0})

    y = df.iloc[:,0].values
    X = df.iloc[:,1:].values

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

def evaluacion(y_real, y_pred):
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for real, pred in zip(y_real, y_pred):
        if real == 1 and pred == 1:
            TP += 1
        elif real == 1 and pred == 0:
            FN += 1
        elif real == 0 and pred == 0:
            TN += 1
        elif real == 0 and pred == 1:
            FP += 1
    print("\n")
    print("--Matriz de Confusion--")
    print(f"           Pred: 1    Pred: 0")
    print(f"Real: 1  |   {TP}   |   {FN}   |")
    print("---------------------------")
    print(f"Real: 0  |   {FP}   |   {TN}   |")
    print("---------------------------\n")

    accuracy = (TP + TN)/(TP + TN + FP + FN) if (TP + TN + FP + FN ) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

    print(f"Accuracy:       {accuracy:.4f}")
    print(f"Precision:      {precision:.4f}")
    print(f"Sensibilidad:   {recall:.4f}")
    print(f"Especificidad:  {specificity:.4f}\n")

def evaluacion_multiclase(y_real, y_pred):
    #print("ENTRAAAAAAAAA")
    correctos = np.sum(y_real == y_pred)
    total = len(y_real)
    accuracy = correctos/total

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Acertados: {correctos} de {total}")

    print("\n")
    print("--Matriz de Confusion--")
    matriz = pd.crosstab(y_real, y_pred, rownames=['Real'], colnames=['Prediccion'])
    print(matriz)
    print("-----------------------\n")
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
    def __init__(self, min_samples_split, max_depth, max_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None
        self.max_features = max_features

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

        if self.max_features == None:
            features_used  = np.arange(n_features)
        else:
            features_used = np.random.choice(n_features, self.max_features, replace=False)

        for feature in features_used:
            valores_unicos = np.unique(X[:, feature])
            thresholds = (valores_unicos[:-1] + valores_unicos[1:]) / 2

            for threshold in thresholds:
                mascara_izq = X[:,feature] <= threshold
                mascara_der = X[:,feature] > threshold

                y_izq = y[mascara_izq]
                y_der = y[mascara_der]

                #print(y_izq.shape)
                #print(y_der.shape)

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

                #print(f" mejor gini: {mejor_gini}")
                #print(f"mejor feature: {mejor_feature}")
                #print(f" mejor threshold: {mejor_threshold}")

        return mejor_threshold, mejor_feature, mejor_gini


    def construir_arbol(self, X, y, profundidad = 0):
        #parar
        if self.gini(y) == 0 or profundidad >= self.max_depth or len(y) < self.min_samples_split:
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
        while not nodo.es_hoja(): # type: ignore
            if row[nodo.feature] <= nodo.threshold: #type: ignore
                nodo = nodo.left #type: ignore
            else:
                nodo = nodo.right #type:ignore
        return nodo.valor #type:ignore
            
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

        print(f" feature {nodo.feature} <= threshold {nodo.threshold: .4f}")

        print(f"{espacio} |-- SI - ", end="")
        self.imprimir_arbol(nodo.left, espacio + " |   ")

        print(f"{espacio} |--- NO - ", end="")
        self.imprimir_arbol(nodo.right, espacio + "     ")

class RandomForest:
    def __init__(self, min_samples_split, max_depth, n_arboles):
       self.min_samples_split = min_samples_split
       self.max_depth = max_depth
       self.n_arboles = n_arboles
       self.bosque = []


    def bootstrap(self, X, y):
        samples_size = len(X)
        sample_X = []
        sample_y = []

        for i in range(samples_size):
            random = np.random.randint(0, samples_size)
            sample_X.append(X[random])
            sample_y.append(y[random])

        return np.array(sample_X), np.array(sample_y)


    def fit(self, X, y):
        self.bosque = []
        n_samples, n_features = X.shape
        n_subconjunto = int(np.sqrt(n_features))

        for i in range(self.n_arboles):
            sample_X, sample_y = self.bootstrap(X, y)
            arbol = DecisionTree(min_samples_split=self.min_samples_split, max_depth=self.max_depth, max_features=n_subconjunto)
            arbol.fit(sample_X, sample_y)

            self.bosque.append(arbol)
            print("Arbol ", i + 1)
            arbol.imprimir_arbol()
            print("\n")


    def predict(self, X):
        predicciones_finales = []

        for fila in X:
            votos = {}
            for arbol in self.bosque:
                prediccion = arbol.predict_row(fila)
                votos[prediccion] = votos.get(prediccion, 0) + 1

            ganador = max(votos, key=lambda k : votos[k])

            predicciones_finales.append(ganador)
        return np.array(predicciones_finales)

def main():
    X, y = csv_multiclase() 

    print("X")
    print(type(X))
    print(X.shape)

    print("y")
    print(type(y))
    print(y.shape)

    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=.3)

    print("BOSQUE - \n")

    while True:
        numero_arboles = int(input("Ingresa el numero de arboles impar: "))
        if numero_arboles%2 != 0:
            break
    
    bosque = RandomForest(2, 5, numero_arboles)
    bosque.fit(X_train, y_train)
    predicciones = bosque.predict(X_test)

    print("\n Predicciones del bosque:  ", predicciones)
    print("Valores reales:            ", y_test)

    evaluar = evaluacion_multiclase(y_test, predicciones)


    print("ARBOL - \n")

    #Arbol
    arbol = DecisionTree(2,5)
    arbol.fit(X_train, y_train)
    prediccion_arbol = arbol.predict(X_test)

    arbol.imprimir_arbol()

    print("\n Predicciones de un solo arbol:  ", prediccion_arbol)
    print("Valores reales:            ", y_test)

    evaluar_arbol = evaluacion_multiclase(y_test, prediccion_arbol)



if __name__ == "__main__":
    main()