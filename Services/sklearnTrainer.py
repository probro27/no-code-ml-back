from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def findTestTrainData(df: pd.DataFrame, trainPart: int, testPart: int, predictionColumn: str) -> dict:
    actualPredictionColumn = df.pop(predictionColumn)
    X_train, X_test, y_train, y_test = train_test_split(df, actualPredictionColumn, test_size=testPart, shuffle=True, random_state=1)
    rtnData = {
        'X_train': X_train, 
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
    return rtnData

def buildTrainModel(modelType: str, modelName: str, data: dict, args: dict, scoring: str = 'accuracy'):
    package = f'sklearn.{modelType}'
    name = modelName
    modelImport = getattr(__import__(package, fromlist=[name]), name)
    model = modelImport()
    model.set_params(**args)
    X_train: pd.DataFrame = data['X_train']
    textBindings = []
    for column in X_train:
        if X_train[column].unique().shape[0] > 50 and isinstance(X_train[column][0] , str):
            textBindings.append(column)
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    for column in X_train:
        if column in textBindings:
            tfidf = TfidfVectorizer(stop_words='english')
            train_x_vector = tfidf.fit_transform(X_train[column])
            pd.DataFrame.sparse.from_spmatrix(train_x_vector, index=X_train.index, columns=tfidf.get_feature_names_out())
            test_x_vector = tfidf.transform(X_test[column])
            print(type(test_x_vector))
            X_train[column] = train_x_vector.toarray()
            X_test[column] = test_x_vector.toarray()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    scorePackage = 'sklearn.metrics'
    components = scoring.split('_')
    if components[-1] == 'score' or components[-1] == 'loss' or components[-1] == 'error' or components[-1] == 'matrix': 
        scorerName = scoring
    else:
        scorerName = f'{scoring}_score'
    scorerImport = getattr(__import__(scorePackage, fromlist=[scorerName]), scorerName)
    score = scorerImport(y_test, predictions)
    rtnDict = {
        'predictions': predictions.tolist(),
        'actual': y_test.tolist(), 
        'scoring': score
    }
    return rtnDict
