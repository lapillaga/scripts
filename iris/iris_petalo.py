import pandas as pd
import numpy as np
import matplotlib.pyplot as plot

iris = pd.read_csv('IRIS.csv')

# Separar datos
from sklearn.model_selection import train_test_split
attributes = np.array(iris.drop(['species', 'sepal_length', 'sepal_width'], 1))
labels = np.array(iris['species'])

attributes_train, attributes_test, labels_train, labels_test = train_test_split(
    attributes,
    labels,
    test_size=0.2
)
# print(attributes)
# print('Datos de entrenamiento: {}'.format(attributes_train.shape[0]))
# print('Datos de prueba: {}'.format(attributes_test.shape[0]))

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