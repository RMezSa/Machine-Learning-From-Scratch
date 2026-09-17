import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#Iris dataset
def csv_prueba():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    columnas = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'especie']
    df = pd.read_csv(url, names=columnas)
    df = df[df['especie'].isin(['Iris-versicolor', 'Iris-virginica'])]
    # Mapeamos a 0 Negativo y 1 Positivo
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

#Dataset de cancer de mama
def csv_cancer_mama():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    df = pd.read_csv(url, header=None)
    df = df.drop(0, axis=1)
    df[1] = df[1].map({'M':1, 'B':0})

    y = df.iloc[:,0].values
    X = df.iloc[:,1:].values

    return X, y


# Dataset de vinos, 3 clases
def csv_multiclase():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
    df = pd.read_csv(url, header=None)

    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values

    return X, y


#Dataset de magic telescope, 19000 filas
def csv_pesado():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data"
    df = pd.read_csv(url, header=None)

    df[10] = df[10].map({'g': 1, 'h': 0})

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    return X, y

#Función de split para dataset, por default 20% para test
def split_dataset(X, y, test_size=.2, random_seed=42):
    
    np.random.seed(random_seed)
    idx = np.random.permutation(len(X))
    corte = int(len(X) * (1 - test_size))
    train_idx, test_idx = idx[:corte], idx[corte:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

#Evaluación multiclase, matriz de confusión, métricas de accuracy
def evaluacion_multiclase(y_real, y_pred, titulo="Resultados"):
    correctos = np.sum(y_real == y_pred)
    total = len(y_real)
    accuracy = correctos / total

    print(f"\n-- {titulo} --")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Acertados: {correctos} de {total}")

    #Matriz de confusion
    print("\n--Matriz de Confusion--")
    matriz = pd.crosstab(pd.Series(y_real, name='Real'), pd.Series(y_pred, name='Prediccion'))
    print(matriz)
    print("-----------------------\n")

    return accuracy


def main():
    #X, y = csv_prueba()
    #X, y = csv_cancer_mama()
    #X, y = csv_multiclase()        
    X, y = csv_pesado()
    
    print("X")
    print(type(X))
    print(X.shape)

    print("y")
    print(type(y))
    print(y.shape)

    #Split de datos de entrenamiento y de test
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.3, random_seed=42)

    while True:
        numero_arboles = int(input("Ingresa el numero de arboles impar: "))
        if numero_arboles % 2 != 0:
            break

    print("-- BOSQUE (scikit-learn) -- \n")

    print("CROSS VALIDATION --  \n")

    #5 splits para el cross validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    accuracy_k_fold = []
    fold = 1

    #Cross validation 
    for train_idx, test_idx in kf.split(X_train):
        X_ctrain, X_ctest = X_train[train_idx], X_train[test_idx]
        y_ctrain, y_ctest = y_train[train_idx], y_train[test_idx]

        #creación de RandomForestClassifier para cross validation 
        bosque_cv = RandomForestClassifier(
            n_estimators=numero_arboles,
            min_samples_split=2,
            max_depth=5,
            max_features='sqrt',
            criterion='gini',
            bootstrap=True,
            random_state=42
        )
        #entrenamiento del bosque
        bosque_cv.fit(X_ctrain, y_ctrain)

        #predicciones
        prediccion_cv = bosque_cv.predict(X_ctest)

        # Encontrar accuracy
        accuracy = accuracy_score(y_ctest, prediccion_cv)
        accuracy_k_fold.append(accuracy)
        print(f"fold {fold}: accuracy: {accuracy}")
        fold += 1

    #promedio de accuracy de cross validation
    promedio_cv = np.mean(accuracy_k_fold)
    print(f"Cross validation mean accuracy: {promedio_cv: .4f} \n")

    print("ENTRENAMIENTO--")

    #Entrenamiento del random forest
    bosque = RandomForestClassifier(
        n_estimators=numero_arboles,
        min_samples_split=2,
        max_depth=5,
        max_features='sqrt',
        criterion='gini',
        bootstrap=True,
        random_state=42
    )
    #Entrenamiento del random forest
    bosque.fit(X_train, y_train)

    #Predicciones del random forest
    predicciones = bosque.predict(X_test)

    print("\n Predicciones del bosque:  ", predicciones)
    print("Valores reales:            ", y_test)

    #Evaliación
    accuracy_sklearn = evaluacion_multiclase(y_test, predicciones, titulo="Random Forest (sklearn)")

    print("\n--Classification report (sklearn)--")
    print(classification_report(y_test, predicciones))

    #metrica para comparación
    accuracy_from_scratch = None
    if accuracy_from_scratch is not None:
        print("\n-- Comparacion --")
        print(f"Accuracy from scratch : {accuracy_from_scratch:.4f}")
        print(f"Accuracy sklearn       : {accuracy_sklearn:.4f}")
        diferencia = accuracy_sklearn - accuracy_from_scratch
        print(f"Diferencia (sklearn - from scratch): {diferencia:+.4f}")


if __name__ == "__main__":
    main()