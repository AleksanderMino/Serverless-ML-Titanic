import os 
import modal

LOCAL = True

if LOCAL == False:
    stub = modal.Stub()
    image = modal.Image.debian_slim().pip_install(['hopsworks == 3.0.4', 'joblib','seaborn','sklearn','dataframe-image'])
    
    @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name('HOPSWORKS_API_KEY'))
    def f():
        g()

def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()
    titanic_df = pd.read_csv('titanic_dataset.csv')
    titanic_fg = fs.get_or_create_feature_group(
        name='titanic_modal_2',
        version=1,
        primary_key=['passenger_id'],
        description='Titanic dataset'
    )
    titanic_fg.insert(titanic_df, write_options={"wait_for_job" : False})

if __name__ == '__main__':
    if LOCAL == True:
        g()
    else:
        with stub.run():
            f()
