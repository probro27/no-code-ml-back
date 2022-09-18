from flask import Flask, request, jsonify
from Services.sklearnTrainer import findTestTrainData, buildTrainModel
from flask_cors import CORS
import pandas as pd

app = Flask(__name__)
CORS(app, resources={r"*": {"origins": "*"}})

@app.route('/')
def basicRoute():
    return {'hello': 'world'}

def loadDataUrl(url: str) -> pd.DataFrame:
    components = url.split('.')
    if components[-1] == "csv":
        df = pd.read_csv(url, na_values=['?'])
    else:
       df = pd.read_csv(url)
    return df

@app.route('/sklearn', methods=['POST'])
def sklearnScript():
    data = dict(request.json)
    trainPart: float = float(data['trainPart'])
    testPart: float = float(data['testPart'])
    arguments: dict = data['args']
    dataset = data['dataset']
    modelType: str = data['modelType']
    modelName: str = data['modelName']
    scoring: str = data['scoring']
    predictionColumn: str = data['predictColumn']
    df: pd.DataFrame = loadDataUrl(dataset)
    trainTestData: dict = findTestTrainData(df=df, trainPart=trainPart, testPart=testPart, predictionColumn=predictionColumn)
    results = buildTrainModel(type=modelType, modelName=modelName, data=trainTestData, args=arguments, scoring=scoring)
    return jsonify(results)

@app.route('/tensorflow', methods=['POST'])
def tensorflowScript():
    
    pass

def main():
    app.run()

if __name__=='__main__':
    main()
