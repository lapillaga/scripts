import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

# Leer archivo csv
quien_df = pd.read_csv("quienesquien.csv")

# Remover tabla No ya que collab genera indices
uien_df = quien_df.drop('No.', axis=1)
print(quien_df.head())

# Imprime la información del dataset
print(quien_df.info())
print('=======================================================================')
print(quien_df.describe())
print('=======================================================================')
print(quien_df.Personaje)

# Gráfica pastel para mostrar hombres y mujeres
labels = 'Hombres', 'Mujeres'
sizes = [len(quien_df[quien_df['Sexo'] == "Hombre"]), len(quien_df[quien_df['Sexo'] == "Mujer"])]
explode = (0, 0) 

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal') 

plt.show()

# Reemplazo de datos en el dataframe
numeric_quien_df = quien_csv.replace(['SI', 'NO'], [1, 2])
numeric_quien_df = numeric_quien_df.replace(['SÍ', 'No'], [1, 2])
numeric_quien_df = numeric_quien_df.replace('Si')

X = np.array(numeric_quien_df.drop(['Sexo', 'Personaje'], 1))
Y = np.array(numeric_quien_df['Sexo'])

print(X)
print(Y)

#  Creación del modelo
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=2)
X_train.shape[0]

# Regresión Logística
algoritmo = LogisticRegression()
algoritmo.fit(X_train, Y_train)
Y_pred = algoritmo.predict(X_test)
print('Presición Regresión Logística: {}'.format(algoritmo.score(X_train, Y_train)))
print(Y_pred)

# Prediccióñ con Support veector machine
algoritmo = SVC()
algoritmo.fit(X_train, Y_train)
Y_pred = algoritmo.predict(X_test)
print('Presición Vector de Soporte: {}'.format(algoritmo.score(X_train, Y_train)))
print(Y_pred)

#  Predicción con Decision tree clasifier
algoritmo = DecisionTreeClassifier()
algoritmo.fit(X_train, Y_train)
Y_pred = algoritmo.predict(X_test)
print('Presición Árbol de decisión: {}'.format(algoritmo.score(X_train, Y_train)))
print(Y_pred)

# Predicción con MLP
iteraciones = 1000
capas_ocultas = [10,10,10]

algoritmo = MLPClassifier(hidden_layer_sizes=capas_ocultas, max_iter=iteraciones)
algoritmo.fit(X_train, Y_train)
Y_pred = algoritmo.predict(X_test)
print('Presición Red Neuronal: {}'.format(algoritmo.score(X_train, Y_train)))
print(Y_pred)