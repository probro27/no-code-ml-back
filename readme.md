# No Code ML Model

Want to build a no code Machine Learning platform. 

## Basic Documentation

Two ways to build model - 

#### Scikit-Learn 

- Specify the Model
- Crawl and store the arguments that the model requires. 
- Use Sklearn API to make calls. 

Sample doc

```json
{
    "training_set": {
        "url" // could be a json (pandas)
    }, 
    "testing_set": {
        "url" // could be a json (pandas)
    },
    "estimator": "LinearRegression", // or something else
    "args": {
        ... // depends on the estimator selected
    }
}
```

#### Tensorflow

- Specify the model
- Ask for Neural Network - Dense or CNN
- Visualize layers for user to add. 

