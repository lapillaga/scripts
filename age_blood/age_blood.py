import pandas as pd
import numpy as np
import matplotlib.pyplot as plot

blood_data = pd.read_csv('age-blood.csv', sep=';')
blood_data = blood_data.drop('Index', axis=1)

# Graphic
blood_fig = blood_data.plot(
    kind='scatter',
    x='Age',
    y='Systolic Blood Pressure',
    color='blue',
    label='Blood Age'
)
blood_fig.set_xlabel('Edad')
blood_fig.set_ylabel('Presión sistolica')
blood_fig.set_title('Edad vs Presión sistolica')
plot.show()

# Test and Train Data
from sklearn.model_selection import train_test_split

attributes = np.array(blood_data.drop(['Age'], 1))
label = np.array(blood_data['Age'])
print(attributes)
print(label)
