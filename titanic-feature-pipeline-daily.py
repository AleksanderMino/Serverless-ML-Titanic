import os 
import modal

BACKFILL = False
LOCAL = True

if LOCAL == False:
    stub = modal.Stub()
    image = modal.Image.debian_slim().apt_install(["libgomp1"]).pip_install(["hopsworks==3.0.4", "seaborn", "joblib", "scikit-learn"])

    @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
    def f():
        g()

def generate_passenger(survived, age, sex, embarked, fare, pclass):
    import pandas as pd
    import random

    df = pd.DataFrame({ "age_encoded": [age],
    "sex_encoded": sex,
    "embarked_encoded": embarked,
    "fare_encoded": fare,
    "pclass": pclass
})

    df['survived'] = survived
    df['passenger_id'] = round(random.uniform(891,1000))
    return df

def get_random_passenger():

    import pandas as pd 
    import random

    pick_random = random.uniform(0,2)
    if pick_random >= 1:
        passenger_df = generate_passenger(1, float(round(random.uniform(0,3))), round(random.uniform(0,1)), round(random.uniform(1,2)), float(round(random.uniform(3,6))), round(random.uniform(2,3)))
        print("Survived")
    else:
        passenger_df = generate_passenger(0, float(round(random.uniform(3,6))), round(random.uniform(0,1)), round(random.uniform(0,1)), float(round(random.uniform(0,3))), round(random.uniform(1,2)))
        print("Did not survive")
    
    return passenger_df


# def generate_passenger():
def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()

    if BACKFILL == True:
        titanic_df = pd.read_csv("titanic_dataset.csv")
    else:
        titanic_df = get_random_passenger()
    
    titanic_fg = fs.get_or_create_feature_group(
        name="titanic_modal_2",
        version=1,
        primary_key=["passnger_id"], 
        description="Titanic dataset")
    titanic_fg.insert(titanic_df, write_options={"wait_for_job" : False})

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()
