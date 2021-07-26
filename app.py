import flask
from flask import Flask, request
import numpy as np
import joblib

'''
loading predicted models
'''
# loading iris dataset
iris_model_file_path = 'models/iris_model.pkl'
iris_model_file = joblib.load(iris_model_file_path)

# loading fruit dataset
fruit_model_file_path = 'models/fruit_model.pkl'
fruit_model_file = joblib.load(fruit_model_file_path)

# loading Titanic dataset
titanic_model_file_path = 'models/titanic_model.pkl'
titanic_model_file = joblib.load(titanic_model_file_path)

app = Flask(__name__)


@app.route('/')
def index():
    title = 'Home'
    return flask.render_template('index.html', title=title)


@app.route('/iris')
def iris_dataset():
    title = 'Iris Dataset'
    return flask.render_template('iris.html', title=title)


@app.route('/iris-predict', methods=['POST'])
def predict_iris():
    title = 'Iris Predicted - Output'
    dataset_title = 'Iris Dataset Output'
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = np.array(list(map(float, to_predict_list))).reshape(1, -1)
        print(to_predict_list)
        prediction = iris_model_file.predict(to_predict_list)
        finalAns = prediction[0]
        return flask.render_template('result.html', prediction=finalAns, title=title, dataset_title=dataset_title)


@app.route('/fruit')
def fruit_dataset():
    title = 'Fruit Dataset'
    return flask.render_template('fruit.html', title=title)


@app.route('/fruit-predict', methods=['POST'])
def predict_fruit():
    title = 'Fruit Predicted - Output'
    dataset_title = 'Fruit Dataset Output'
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = np.array(list(map(float, to_predict_list))).reshape(1, -1)
        print(to_predict_list)
        prediction = fruit_model_file.predict(to_predict_list)
        finalAns = prediction[0]
        return flask.render_template('result.html', prediction=finalAns, title=title, dataset_title=dataset_title)


@app.route('/titanic')
def titanic_dataset():
    title = 'Titanic Dataset'
    return flask.render_template('titanic.html', title=title)


@app.route('/titanic-predict', methods=['POST'])
def predict_titanic():
    title = 'Titanic Predicted - Output'
    dataset_title = 'Titanic Dataset Output'
    if request.method == 'POST':
        PClass = request.form['PClass']
        if PClass == 'Upper':
            PClass = 1
        elif PClass == 'Middle':
            PClass = 2
        else:
            PClass = 3
        Age = float(request.form['Age'])
        Fare = float(request.form['Fare'])
        gender = request.form['Gender']
        if gender == 'Male':
            gender = 1
        else:
            gender = 0
        prediction = titanic_model_file.predict([[PClass, Age, Fare, gender]])
        finalAns = prediction[0]
        if finalAns:
            finalAns = 'Survived'
        else:
            finalAns = 'Not Survived'
        return flask.render_template('result.html', title=title, dataset_title=dataset_title, prediction=finalAns)


if __name__ == '__main__':
    app.run(debug=True)
