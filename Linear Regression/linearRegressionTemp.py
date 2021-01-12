import sklearn
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn import model_selection
from sklearn import linear_model

x = list(range(0, 10))
y = [1.8*F + 32 for F in x]
#y = [1.8*F + 32 + random.randint(-3,3) for F in x]
print(f'X: {x}')
print(f'Y: {y}')


plt.plot(x,y,'-*r')
plt.show()

x = np.array(x).reshape(-1,1)
y = np.array(y).reshape(-1,1)

xTrain, xTest, yTrain, yTest = model_selection.train_test_split(x,y,test_size=0.2)
model = linear_model.LinearRegression()
model.fit(xTrain, yTrain)

accuracy = model.score(xTest, yTest)
print(f'Accuracy: {round(accuracy*100,2)}')
print(f'Coefficients: {model.coef_}')
print(f'Intercept: {model.intercept_}')






