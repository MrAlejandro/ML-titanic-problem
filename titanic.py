import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('data.csv')

#X = dataset.drop('Survived', axis=1)
y = dataset.iloc[:, 1:2]

# splitting to train and test datasets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset, y, test_size = 1/3, random_state = 0)


# TODO: don't forget to drop the Survived column when start training
