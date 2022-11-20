import os
import modal

LOCAL=True

if LOCAL == False:
   stub = modal.Stub()
   hopsworks_image = modal.Image.debian_slim().pip_install(["hopsworks==3.0.4","joblib","seaborn","sklearn","dataframe-image"])
   @stub.function(image=hopsworks_image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       g()

def g():
    import pandas as pd
    import hopsworks
    import joblib
    import datetime
    from PIL import Image
    from datetime import datetime
    import dataframe_image as dfi
    from sklearn.metrics import confusion_matrix
    from matplotlib import pyplot
    import seaborn as sns
    import requests

    project = hopsworks.login()
    fs = project.get_feature_store()

    mr = project.get_model_registry()
    model = mr.get_model("titanic_modal", version=2)
    model_dir = model.download()
    model = joblib.load(model_dir + "/titanic_model.pkl")

    feature_view = fs.get_feature_view(name="titanic_modal_2", version=1)
    batch_data = feature_view.get_batch_data()

    y_pred = model.predict(batch_data)
    # print(y_pred)
    passenger = y_pred[y_pred.size-1]
    print('Predicted fate in binary: ',passenger)
    if passenger == 1:
        print("Fate of the passenger predicted: Survived!!" )
        passenger_icon = "https://raw.githubusercontent.com/AleksanderMino/Serverless-ML-Titanic/main/survived.jpeg"

    else:
        print("Fate of the passenger predicted: Did not survive")
        passenger_icon = "https://raw.githubusercontent.com/AleksanderMino/Serverless-ML-Titanic/main/Dicaprio_fate.jpeg"
    img = Image.open(requests.get(passenger_icon,stream=True).raw)
    img.save("./latest_passenger.png")
    dataset_api = project.get_dataset_api()
    dataset_api.upload("./latest_passenger.png", "Resources/images", overwrite=True)

    titanic_fg = fs.get_feature_group(name="titanic_modal_2", version=1)
    df = titanic_fg.read()
    print(df["survived"])
    label_binary = df.iloc[-1]["survived"]
    print('actual fate in binary ',label_binary)
    if label_binary==0.0:
        label = "Not Survived"
    else:
        label = "Survived"

    if label == 'Survived':
        print("Actual fate of the passenger Survived!!" )
        passenger_icon = "https://raw.githubusercontent.com/AleksanderMino/Serverless-ML-Titanic/main/survived.jpeg"

    else:
        print("Actual fate of the passenger: Did not survive")
        passenger_icon = "https://raw.githubusercontent.com/AleksanderMino/Serverless-ML-Titanic/main/Dicaprio_fate.jpeg"
    img = Image.open(requests.get(passenger_icon, stream=True).raw)
    img.save("./actual_fate.png")
    dataset_api.upload("./actual_fate.png", "Resources/images", overwrite=True)

    monitor_fg = fs.get_or_create_feature_group(name="titanic_predictions",
                                                version=1,
                                                primary_key=["datetime"],
                                                description="Titanic Prediction/Outcome Monitoring"
                                                )

    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    data = {
        'prediction' : [passenger],
        'label' : [label_binary],
        'datetime': [now] 
    }

    monitor_df = pd.DataFrame(data)
    monitor_fg.insert(monitor_df, write_options={"wait_for_job" : False})

    history_df = monitor_fg.read()
    history_df = pd.concat([history_df, monitor_df])

    df_recent = history_df.tail(5)
    dfi.export(df_recent, './df_recent.png', table_conversion = 'matplotlib')
    dataset_api.upload("./df_recent.png", "Resources/images", overwrite=True)
    
    predictions = history_df[['prediction']]
    labels = history_df[['label']]

    print("Number of different fate predictions to date: " + str(predictions.value_counts().count()))
    if predictions.value_counts().count() == 2:
        results = confusion_matrix(labels, predictions)

        df_cm = pd.DataFrame(results, ['True Fate: Survived', 'True Fate: Not Survived'],
                             ['Pred Fate: Survived', 'Pred Fate: Not Survived'])
        cm = sns.heatmap(df_cm, annot=True)
        fig = cm.get_figure()
        fig.savefig("./confusion_matrix.png")
        dataset_api.upload("./confusion_matrix.png", "Resources/images", overwrite=True)
    else:
        print("You need 2 different passenger's fate predictions to create the confusion matrix.")
        print("Run the batch inference pipeline more times until you get 2 different passenger's fate predictions")

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()

