import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics

cancer = datasets.load_breast_cancer()

'''
print(cancer.feature_names)
print(cancer.target_names)'''

x = cancer.data
y = cancer.target

xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

classification = svm.SVC(kernel="linear", C=2)
classification.fit(xTrain, yTrain)

Prediction = classification.predict(xTest)

Accuracy = metrics.accuracy_score(yTest, Prediction)
print(round(Accuracy*100, 3))

