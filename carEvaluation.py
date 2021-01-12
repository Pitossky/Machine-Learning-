import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.data")

labelOne = preprocessing.LabelEncoder()
buying = labelOne.fit_transform(list(data["buying"]))
maint = labelOne.fit_transform(list(data["maint"]))
door = labelOne.fit_transform(list(data["door"]))
persons = labelOne.fit_transform(list(data["persons"]))
lug_boot = labelOne.fit_transform(list(data["lug_boot"]))
safety = labelOne.fit_transform(list(data["safety"]))
cls = labelOne.fit_transform(list(data["class"]))

predict = "class"
x = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

model = KNeighborsClassifier(n_neighbors=9)
model.fit(xTrain, yTrain)
accuracy = model.score(xTest, yTest)
print(round(accuracy*100, 3))

predicted = model.predict(xTest)
names = ["unacc", "acc", "good", "very good"]

for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]],",", "Data: ", xTest[x],",", "Actual: ", names[yTest[x]])
    neighbourDistance = model.kneighbors([xTest[x]], 9, True)
    print("Distance to Neighbour: ", neighbourDistance)



