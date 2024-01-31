import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

movieDataFile = "rotten_tomatoes_movies.csv"

movieData = pd.read_csv(movieDataFile, header=0)
movieData = movieData.drop(
    columns=['rotten_tomatoes_link', 'movie_info', 'critics_consensus', 'tomatometer_status', 'audience_status',
             'tomatometer_fresh_critics_count', 'tomatometer_rotten_critics_count', 'audience_rating'])
movieData.dropna(inplace=True)
movieData['genres'].str.split(pat=",")
movieData['directors'].str.split(pat=",")
movieData['authors'].str.split(pat=",")
movieData['actors'].str.split(pat=",")
movieDetailsTrain = movieData.iloc[:13000]
movieDetailsTest = movieData.iloc[13000:]
movieDetailsPredict = movieDetailsTest.iloc[:10]
criticScoresTrain = movieDetailsTrain.pop('tomatometer_rating')
criticScoresTest = movieDetailsTest.pop('tomatometer_rating')
criticScoresPredict = movieDetailsPredict.pop('tomatometer_rating')
inputs = {}
for name, column in movieDetailsTrain.items():
    dataType = column.dtype
    if dataType == object:
        dataType = tf.string
    else:
        dataType = tf.float32
    inputs[name] = keras.Input(shape=(1,), name=name, dtype=dataType)
numericInputs = {name: i for name, i in inputs.items()
                 if i.dtype == tf.float32}
if len(numericInputs) > 0:
    concatLayer = keras.layers.Concatenate()(list(numericInputs.values()))
    normLayer = keras.layers.Normalization()
    normLayer.adapt(np.array(movieData[numericInputs.keys()]))
    allNumericInputs = normLayer(concatLayer)
    preprocessedInputs = [allNumericInputs]
else:
    preprocessedInputs = []
for name, i in inputs.items():
    if i.dtype == tf.float32:
        continue
    lookupStrings = keras.layers.StringLookup(vocabulary=np.unique(movieData[name]))
    categoryEncoding = keras.layers.CategoryEncoding(num_tokens=lookupStrings.vocabulary_size())
    stringInputs = lookupStrings(i)
    stringInputs = categoryEncoding(stringInputs)
    preprocessedInputs.append(stringInputs)
concatPreprocessedInputs = keras.layers.Concatenate()(preprocessedInputs)
preprocessingModel = keras.Model(inputs, concatPreprocessedInputs)
movieDetailsTrainDict = {name: np.array(value) for name, value in movieDetailsTrain.items()}
movieDetailsTestDict = {name: np.array(value) for name, value in movieDetailsTest.items()}
movieDetailsPredictDict = {name: np.array(value) for name, value in movieDetailsPredict.items()}
model = keras.Sequential([
    keras.layers.Dense(1000, activation="relu", kernel_regularizer="l1_l2"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(800, activation="relu"),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(500, activation="relu", kernel_regularizer="l1_l2"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(500, activation="relu"),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(300, activation="relu", kernel_regularizer="l1_l2"),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(10, activation="relu"),
    keras.layers.Dense(1, activation="linear")
])
preprocessedInputs = preprocessingModel(inputs)
res = model(preprocessedInputs)
movieModel = keras.Model(inputs, res)
movieModel.compile(loss="mean_absolute_error", optimizer="adam",
                   metrics=["mae"])
movieModel.fit(x=movieDetailsTrainDict, y=criticScoresTrain, epochs=50)
movieModel.save("Project4MovieModel")
# movieModel = keras.models.load_model("Project4MovieModel")
testResults = movieModel.evaluate(x=movieDetailsTestDict, y=criticScoresTest)
print(testResults)
print(movieModel.predict(movieDetailsPredictDict))
print(criticScoresPredict)
