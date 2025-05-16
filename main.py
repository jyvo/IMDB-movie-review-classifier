from sklearn.datasets import load_files

from tensorflow.keras.layers import TextVectorization, Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.models import Sequential

# load dataset
trainData = load_files('data/processed/train', categories=['pos', 'neg'],
                        shuffle=True,
                        load_content=True,
                        encoding='UTF-8',
                        random_state=42)
testData = load_files('data/processed/test',
                       categories=['pos', 'neg'],
                        shuffle=True,
                        load_content=True,
                        encoding='UTF-8',
                        random_state=42)

# split data into training and testing sets.
xTrain, yTrain, xTest, yTest = trainData.data, trainData.target, testData.data, testData.target


# preprocess the text data (tokenization, padding)
tokens = 10000
v = TextVectorization(max_tokens=tokens,
                      output_mode='int',
                      output_sequence_length=500)
v.adapt(xTrain)

xTrainTokens = v(xTrain)
xTestTokens = v(xTest)


# build RNN model
model = Sequential()
model.add(Embedding(input_dim=tokens, output_dim=128, input_length=50))
model.add(Bidirectional(LSTM(64)))
model.add(Dense(1, activation='sigmoid'))

# train and compile on training set
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(xTrainTokens, yTrain, batch_size=64, epochs=5, validation_data=(xTestTokens, yTest))

# evaluate the model and its performance
model.evaluate(xTestTokens, yTest)
model.summary()

# make sample text predictions
sampleTexts = ["This movie was absolutely fantastic!",
               "I hated every minute of it.",
               "An emotional rollercoaster! The performances were heartfelt and the cinematography was stunning.",
               "I found the plot to be predictable and the characters lacked depth. Not a memorable film.",
               "It was an average experience—some scenes were entertaining, but the pacing was inconsistent.",
               "Brilliant direction and a captivating storyline! Easily one of the most compelling films this year.",
               "The soundtrack was incredible, but the acting left much to be desired.",
               "A beautifully shot movie with a script that kept me on the edge of my seat!",
               "An instant classic! The chemistry between the leads was palpable and the ending was perfect.",
               "The humor felt forced and the dialogue was awkward at times. Disappointing overall."]

for text in sampleTexts:
    textTokenized = v([text])
    predictions = model.predict(textTokenized)
    print(text)
    predLabel = predictions[0][0]
    print(f"Sample prediction {sampleTexts.index(text)+1}: {predLabel:.4f} ({'neg' if predLabel < 0.5 else 'pos'})")