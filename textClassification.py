import tensorflow as tf
from tensorflow import keras
import numpy as np

data = keras.datasets.imdb
(trainData, trainLabels), (testData, testLabels) = data.load_data(num_words=88000)

wordIndex = data.get_word_index()
wordIndex = {k:(v+3) for k, v in wordIndex.items()}
wordIndex['<PAD>'] = 0
wordIndex['<START>'] = 0
wordIndex['<UNK>'] = 2
wordIndex['<UNUSED>'] = 3

reverseWordIndex = dict([(value, key) for (key, value) in wordIndex.items()])

trainData = keras.preprocessing.sequence.pad_sequences(trainData, value=wordIndex['<PAD>'], padding='post', maxlen=250)
testData = keras.preprocessing.sequence.pad_sequences(testData, value=wordIndex['<PAD>'], padding='post', maxlen=250)

def decode_review(text):
    return "".join([reverseWordIndex.get(i, '?') for i in text])

#CREATING THE MODEL

'''
model = keras.Sequential()
model.add(keras.layers.Embedding(88000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

xValue = trainData[:10000]
xTrain = trainData[10000:]

yValue = trainLabels[:10000]
yTrain = trainLabels[10000:]

fitModel = model.fit(xTrain, yTrain, epochs=40, batch_size=512, validation_data=(xValue, yValue), verbose=1)

results = model.evaluate(testData, testLabels)
print(results)

model.save('model.h5')'''


model = keras.models.load_model('model.h5')

def review_encode(s):
    encoded = [1]

    for word in s:
        if word.lower() in wordIndex:
            encoded.append(wordIndex[word.lower()])
        else:
            encoded.append(2)
    return encoded

with open('Array.rtf', encoding='utf-8') as f:
    for line in f.readlines():
        nline = line.replace(',', '').replace('.', '').replace('(', '').replace(')', '').replace(':', '').replace("\"", '').strip().split()
        encode = review_encode(nline)
        encode = trainData = keras.preprocessing.sequence.pad_sequences([encode], value=wordIndex['<PAD>'], padding='post', maxlen=250)
        predict = model.predict(encode)
        print(line)
        print(encode)
        print(predict[0])

'''
testReview = testData[0]
prediction = model.predict(testReview)
print('Review: ')
print(decode_review(testReview))
print('Prediction: ' + str(prediction[0]))
print('Actual: ' + str(testLabels[0]))
print(results)'''





