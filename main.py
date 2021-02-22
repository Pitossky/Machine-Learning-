import numpy as np
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split
import pandas as pd

#READ DATA
data = pd.read_csv("breast-cancer-wisconsin.txt")

#REPLACE MISSING DATA
data.replace('?', -99999, inplace=True)

data.drop(['id'], 1, inplace=True)

x_label = np.array(data.drop(['class'], 1))
y_label = np.array(data['class'])

#SPLITTING DATA
x_label_train, x_label_test, y_label_train, y_label_test = train_test_split(x_label,y_label,test_size=0.2)

#DEFINE CLASSIFIER
classifier = neighbors.KNeighborsClassifier()
classifier.fit(x_label_train, y_label_train)

#TESTING THE CLASSIFIER
accuracy = classifier.score(x_label_test, y_label_test)
print(accuracy)

example_measure = np.array([[4,2,1,1,1,2,3,2,1], [4,2,1,2,2,2,3,2,1]])
example_measure = example_measure.reshape(len(example_measure), -1)
prediction = classifier.predict(example_measure)
print(prediction)


