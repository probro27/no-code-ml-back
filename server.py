from flask import Flask
import flask_cors
from Models.sklearn import SkLearnModel
import pandas as pd

app = Flask(__name__)
flask_cors.CORS(app)

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
def sklearnScript(data: SkLearnModel, dataset: str):
    df = loadDataUrl(dataset)
    return

def main():
    app.run()

if __name__=='__main__':
    main()
