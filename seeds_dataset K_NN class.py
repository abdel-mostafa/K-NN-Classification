import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv('................................\seeds_dataset.csv')
data.replace('?', -99999, inplace=True)

x = data.iloc[:, :7]
y = data.iloc[:, 7]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

estimator = KNeighborsClassifier(n_neighbors=3)
estimator.fit(x_train, y_train)

mean_accuracy = estimator.score(x_test, y_test)
print(mean_accuracy)
