import pandas as pd
import numpy as np

people = pd.read_csv('quienesquien.csv')
people = people.drop('No.', axis=1)
print(people)