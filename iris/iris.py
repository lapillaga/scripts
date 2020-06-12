import pandas as pd # Leer datos y los data frames
import numpy as np # Matrices
import matplotlib.pyplot as plot

iris = pd.read_csv('IRIS.csv')
print(iris.head())
print(iris.info()) # ver la info
print(iris.describe()) # ver la info
print(iris.groupby('species').size()) # agrupar por especies


sepalo = iris[iris.species == 'Iris-setosa'].plot(
    kind='scatter',
    x='sepal_length',
    y='sepal_width',
    color='blue',
    label='Setosa'
)  # Sepalo Grafico

iris[iris.species == 'Iris-versicolor'].plot(
    kind='scatter',
    x='sepal_length',
    y='sepal_width',
    color='green',
    label='Versicolor',
    ax=sepalo
)

iris[iris.species == 'Iris-virginica'].plot(
    kind='scatter',
    x='sepal_length',
    y='sepal_width',
    color='red',
    label='Virginica',
    ax=sepalo
)

sepalo.set_xlabel('Sepalo - Longuitud')
sepalo.set_ylabel('Sepalo - Ancho')
sepalo.set_title('Sepalo - Longitud vs Ancho')

# Petalo Grafico
petalo = iris[iris.species == 'Iris-setosa'].plot(
    kind='scatter',
    x='petal_length',
    y='petal_width',
    color='blue',
    label='Setosa'
)

iris[iris.species == 'Iris-versicolor'].plot(
    kind='scatter',
    x='petal_length',
    y='petal_width',
    color='green',
    label='Versicolor',
    ax=petalo
)

iris[iris.species == 'Iris-virginica'].plot(
    kind='scatter',
    x='petal_length',
    y='petal_width',
    color='red',
    label='Virginica',
    ax=petalo
)

petalo.set_xlabel('Petalo - Longuitud')
petalo.set_ylabel('Petalo - Ancho')
petalo.set_title('Petalo - Longitud vs Ancho')
plot.show()

# Dividir los datos de entrenamiento y prueba
from sklearn.model_selection import train_test_split

attributes = np.array(iris.drop(['species'], 1))
labels = np.array(iris['species'])

attributes_train, attributes_test, labels_train, labels_test = train_test_split(
    attributes,
    labels,
    test_size=0.2
)
print('Datos de entrenamiento: {}'.format(attributes_train.shape[0]))
print('Datos de prueba: {}'.format(attributes_test.shape[0]))

# Modelar con Logistic regresion
from sklearn.linear_model import LogisticRegression

algoritmo = LogisticRegression()
algoritmo.fit(attributes_train, labels_train)

labels_predict = algoritmo.predict(attributes_test)
print('Presición regresión logística: {}'.format(algoritmo.score(attributes_train, labels_train)))

# Modelar con Support vector machine
from sklearn.svm import SVC
svc = SVC()
svc.fit(attributes_train, labels_train)
labels2_predict = svc.predict(attributes_test)
print('Presición Support vector machine: {}'.format(svc.score(attributes_train, labels_train)))

# Modelar con Decission tree clasifier
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
tree.fit(attributes_train, labels_train)
labels3_predict = tree.predict(attributes_test)
print('Presición arbol desicion: {}'.format(tree.score(attributes_train, labels_train)))

# Modelar con MLP
from sklearn.neural_network import MLPClassifier
iterations = 1000
hidden_layers = [10, 10, 10]

mlp = MLPClassifier(hidden_layer_sizes=hidden_layers, max_iter=iterations)
mlp.fit(attributes_train, labels_train)
labels4_predict = mlp.predict(attributes_test)
print('Presición Red Neuronal: {}'.format(mlp.score(attributes_train, labels_train)))
