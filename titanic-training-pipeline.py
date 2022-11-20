import os
import modal
import numpy as np

LOCAL = True

if LOCAL == False:
    stub = modal.Stub()
    image = modal.Image.debian_slim().apt_install(["libgomp1"]).pip_install(["hopsworks==3.0.4", "seaborn", "joblib", "scikit-learn"])

    @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
    def f():
        g()

def g():

    import hopsworks
    import pandas as pd
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier

    from sklearn.svm import LinearSVC
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    import seaborn as sns
    from matplotlib import pyplot
    from hsml.schema import Schema
    from hsml.model_schema import ModelSchema
    import joblib
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    import time
    print('what is it')
    project = hopsworks.login()
    fs = project.get_feature_store()

    try:
        feature_view = fs.get_feature_view(name="titanic_modal_2", version=1)
    except:
        titanic_fg = fs.get_feature_group(name="titanic_modal_2", version=1)
        query = titanic_fg.select_all()
        feature_view = fs.create_feature_view(name="titanic_modal_2",
                                          version=1,
                                          description="Read from Titanic dataset",
                                          labels=["survived"],
                                          query=query)
 
    X_train, X_test, y_train, y_test = feature_view.train_test_split(0.2)
    print(y_train)
    print(X_train)
    #model = LinearSVC(C=0.0001)
    model_parameters = { 
    'n_estimators': [100,200,500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [6,8,10],
    'criterion' :['gini', 'entropy'],
    'min_samples_split': [2, 4, 6]
    
    }

    model=RandomForestClassifier(min_samples_split=6,max_depth=6)

  
    #model = KNeighborsClassifier(n_neighbors=2)
    model.fit(X_train,y_train.values.ravel())

    #res = model.predict(np.asarray(input_list).reshape(1, -1))
    #print('Prediction is: ',res[0])
    #start_time = time.time()

    #rand_search= RandomizedSearchCV(model, model_parameters, cv=5)
    #rand_search.fit(X_train, y_train.values.ravel())
    #print(rand_search.best_params_)
    #print("best accuracy :",rand_search.best_score_)

    #end_time = time.time()
   # print("Total execution time: {} seconds".format(end_time - start_time))

   

    y_pred = model.predict(X_test)
    metrics = classification_report(y_test, y_pred, output_dict=True)
    results = confusion_matrix(y_test,y_pred)
    print('Results: ',results)
    print("score on test: " + str(model.score(X_test, y_test)))
    print("score on train: "+ str(model.score(X_train, y_train)))

    mr = project.get_model_registry()
    model_dir="titanic_model"
    if os.path.isdir(model_dir) == False:
        os.mkdir(model_dir)
    
    joblib.dump(model, model_dir + "/titanic_model.pkl")

    input_schema = Schema(X_train)
    output_schema = Schema(y_train)
    model_schema = ModelSchema(input_schema, output_schema)

    titanic_model = mr.python.create_model(
        name="titanic_modal",
        metrics={"accuracy" : metrics['accuracy']},
        model_schema=model_schema,
        description="Titanic Survivor Predictor"
    )

    titanic_model.save(model_dir)

if __name__ == "__main__":
    if LOCAL == True:
        g()
else:
    with stub.run():
        f()