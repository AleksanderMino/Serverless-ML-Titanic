import gradio as gr
import numpy as np
from PIL import Image
import requests

import hopsworks
import joblib

project = hopsworks.login()
fs = project.get_feature_store()


mr = project.get_model_registry()
model = mr.get_model("titanic_modal", version=2)
model_dir = model.download()
model = joblib.load(model_dir + "/titanic_model.pkl")


def passenger(age, embarked, sex, fare, pclass):

    import random
    
    input_list = []

    passenger_id = round(random.uniform(892,2000))

    if age == '1-7':
        age_encoded = 0
    elif age == '8-15':
        age_encoded = 1
    elif age == '16-25':
        age_encoded = 2
    elif age == '26-30':
        age_encoded = 3
    elif age == '31-35':
        age_encoded = 4
    elif age == '36-50':
        age_encoded = 5
    else:
        age_encoded = 6
    
    if embarked == 'Cherbourgh':
        embarked_encoded = 0
    elif embarked == 'Queenstone':
        embarked_encoded = 1
    else:
        embarked_encoded = 2
    
    if sex == 'Female':
        sex_encoded = 1
    else:
        sex_encoded = 0
    
    if fare == '<= 12.5':
        fare_encoded = 0
    elif fare == '13-25':
        fare_encoded = 1
    elif fare == '26-50':
        fare_encoded = 2
    elif fare == '51-75':
        fare_encoded = 3
    elif fare == '76-100':
        fare_encoded = 4
    elif fare == '101-150':
        fare_encoded = 5
    else:
        fare_encoded = 6
    
    if pclass == '1':
        pclass_int = 1
    elif pclass == '2':
        pclass_int = 2
    else:
        pclass_int = 3

    input_list.append(passenger_id)
    input_list.append(pclass_int)
    input_list.append(sex_encoded)
    input_list.append(embarked_encoded)
    input_list.append(age_encoded)
    input_list.append(fare_encoded)



    
    """ input_list.append(sepal_length)
    input_list.append(sepal_width)
    input_list.append(petal_length)
    input_list.append(petal_width) """

    # 'res' is a list of predictions returned as the label.
    res = model.predict(np.asarray(input_list).reshape(1, -1))
    print(res)
    # We add '[0]' to the result of the transformed 'res', because 'res' is a list, and we only want
    # the first element.
    if res[0] == 0:
        passenger_url = "https://raw.githubusercontent.com/AleksanderMino/Serverless-ML-Titanic/main/Dicaprio_fate.jpeg"
    else:
        passenger_url = "https://raw.githubusercontent.com/AleksanderMino/Serverless-ML-Titanic/main/survived.jpeg"
    img = Image.open(requests.get(passenger_url, stream=True).raw)
    return img

demo = gr.Interface(
    fn=passenger,
    title="Titanic Passenger Fate Predictive Analytics",
    description="Experiment with sepal/petal lengths/widths to predict which flower it is.",
    allow_flagging="never",
    inputs=[
        gr.inputs.Dropdown(['1-7', '8-15', '16-25', '26-30','31-35','36-50','> 50'], label="Age"),
        gr.inputs.Dropdown(['Cherbourgh','Queenstone','Southampton'], label="City of embarkation"),
        gr.inputs.Dropdown(['Male', 'Female'], label="Sex"),
        gr.inputs.Dropdown(['<= 12.5', '13-25','26-50','51-75', '76-100','101-150','> 150'], label="Fare"),
        gr.inputs.Dropdown(['1','2','3'],label='Passenger Class')],
    outputs=gr.Image(type="pil"))

demo.launch(share=True)

