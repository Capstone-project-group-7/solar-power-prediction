from flask import Flask
from flask import request
from flask_cors import CORS
from numpy import array
import pickle

app = Flask(__name__)

# enable cors
CORS(app)


# load the models from disk
radiation_model_filename = './ml_models/radiation_model.sav'
test_model_filename = './ml_models/test_model.sav'

# load models using pickle
loaded_model_radiation = pickle.load(open(radiation_model_filename, 'rb'))
loaded_model_test = pickle.load(open(test_model_filename, 'rb'))


@app.route("/")
def home_view():
    return {
        "api_status": "Up and running"
    }


# Radiation model api endpoint http://127.0.0.1:8000/api/model/radiation
@app.post('/api/model/radiation')
def radiation():
    first_interval = request.form.get('first_interval')
    second_interval = request.form.get('sec_interval')
    third_interval = request.form.get('third_interval')

    capacity_of_one_panel = request.form.get('capacity_of_one_panel')
    number_of_solar_panels = request.form.get('number_of_solar_panels')
    area_of_panel = request.form.get('area_of_panel')

    yield_of_one_panel = (capacity_of_one_panel/(area_of_panel * 10))

    actual_solar_radiation = array([float(first_interval), float(second_interval), float(third_interval)])
    
    actual_solar_radiation = actual_solar_radiation.reshape((1, 3, 1))
    
    predictions = loaded_model_radiation.predict(actual_solar_radiation)

    predictions = predictions.tolist()

    energy = (predictions * capacity_of_one_panel * number_of_solar_panels * area_of_panel * 0.75 * yield_of_one_panel)/(1000) 

    # response
    return {

        "Power energy": energy.tolist()
    }









    # Radiation model api endpoint http://127.0.0.1:8000/api/model/test
@app.post('/api/model/test')
def radiation():
    first_interval = request.form.get('first_interval')
    second_interval = request.form.get('sec_interval')
    third_interval = request.form.get('third_interval')

    actual_solar_radiation = array(
        [float(first_interval), float(second_interval), float(third_interval)])
    actual_solar_radiation = actual_solar_radiation.reshape((1, 3, 1))
    predictions = loaded_model_radiation.predict(actual_solar_radiation)

    # response
    return {
        "predictions": predictions.tolist()
    }
