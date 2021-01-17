import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

(trainImages, trainLabels), (testImages, testLabels) = data.load_data()
fashionItems = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
                'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
trainImages = trainImages/255.0
testImages = testImages/255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
    ])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(trainImages, trainLabels, epochs=5)

#testLoss, testAcc = model.evaluate(testImages, testLabels)

prediction = model.predict(testImages)

for i in range(10):
    plt.grid(False)
    plt.imshow(testImages[i], cmap=plt.cm.binary)
    plt.xlabel('Actual: ' + fashionItems[testLabels[i]])
    plt.title('Prediction: ' + fashionItems[np.argmax(prediction[i])])
    plt.show()





