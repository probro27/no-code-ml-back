from cgi import test
import sklearn
from sklearn.model_selection import train_test_split
import pandas as pd

def findTestTrainData(df: pd.DataFrame, trainPart: int, testPart: int) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(df, train_size=trainPart, test_size=testPart, shuffle=True)
    rtnData = {
        'X_train': X_train, 
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
    return rtnData

def buildModel(modelName: str, data: dict, args: dict):
    
    pass
