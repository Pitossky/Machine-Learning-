import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as plt
import pickle
from matplotlib import style


data = pd.read_csv("student-mat.csv", sep=';')
data = data[["G1", "G2", "G3", "studytime", "absences", "failures"]]

predict = "G3"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])
xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

'''
best = 0
for z in range(10):
    xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    linear = linear_model.LinearRegression()
    linear.fit(xTrain, yTrain)
    accuracy = linear.score(xTest, yTest)
    print(round(accuracy*100, 3))

    if accuracy > best:
        best = accuracy
        with open("studentgrade.pickle", "wb") as f:
            pickle.dump(linear, f)'''

pickleRead = open("studentgrade.pickle", "rb")
linear = pickle.load(pickleRead)

print("Coefficients: ", linear.coef_)
print("Intercept: ", linear.intercept_)

predictions = linear.predict(xTest)
for x in range(len(predictions)):
    print(predictions[x], xTest[x], yTest[x])

p = 'failures'
style.use("ggplot")
plt.scatter(data[p], data["G3"])
plt.xlabel(p)
plt.ylabel("Final Grade")
plt.show()









