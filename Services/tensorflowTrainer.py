import tensorflow as tf
from tensorflow import feature_column
import pandas as pd
from sklearn.model_selection import train_test_split

def findTestTrainData(df: pd.DataFrame, testPart: int, predictionColumn: str) -> dict:
    resultCol = df.pop(predictionColumn)
    X_train, X_test, y_train, y_test = train_test_split(df, resultCol, test_size=testPart, shuffle=True, random_state=1)
    rtnData = {
        'X_train': X_train, 
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
    return rtnData

def buildFeatureColumns(data: dict, categoricalColumns: list, numericColumns: list) -> list:
    X_train = data['X_train']
    feature_columns = []
    for feature_name in categoricalColumns:
        vocabulary = X_train[feature_name].unique()
        feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))
    for feature_name in numericColumns:
        feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))
    return feature_columns

def make_input_function(data: dict, mode: str):
    features = data[f'X_{mode}']
    labels = data[f'y_{mode}']
    def input_function(training:bool=True, batch_size:int=256):
        pass
    return input_function
