from sklearn.model_selection import train_test_split
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

def buildTrainModel(type: str, modelName: str, data: dict, args: dict, scoring: str = 'accuracy'):
    package = f'sklearn.{type}'
    name = modelName
    modelImport = getattr(__import__(package, fromlist=[name]), name)
    model = modelImport()
    model.set_params(**args)
    X_train = data['X_train']
    y_train = data['y_train']
    model.fit(X_train, y_train)
    X_test = data['X_test']
    y_test = data['y_test']
    predictions = model.predict(X_test)
    scorePackage = 'sklearn.metrics'
    components = scoring.split('_')
    if components[-1] == 'score' or components[-1] == 'loss' or components[-1] == 'error': 
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
