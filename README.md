# Serverless-ML-Titanic
A Server-less Machine Learning system for the titanic dataset

The purpose of this project was to create pipeline programs for the different stages needed
and create two spaces in hugging-face. In one of the the user can choose different values for the features and get as a
result the prediction of the model based on these values. In the secong hugging-face space is displayed the lastest 
prediction of the last row in the feature store and the actual fate of the passenger. In addition, the last 5 predictions are displayed in a table format.
Finally all the predictions and the actual fates are displayed in a 2-d confusion matrix.

The first step of this project was to manipulate the data of the titanic datasets. This is done in the dataset_changes script. The resulted dataset is saved as titanic_dataset.csv locally.

Then, I started working on the feature pipeline function which reads the dataset and then creates (if it does not exist already) a feature group in hopsworks with the name titanic_modal_2 and it has as a primary key the passenger id.
This function is not really necessary because the same thing is also done in the function titanic-feature-pipeline-daily.py.

In titanic-training-pipeline function, a feature view is created from the feature group that was created.
The feature view is splitted, 80 % as training data and 20 % as testing data and then is fitted into a Random Forest Classifier model. The model is then saved in hopsworks.

In titanic-feature-pipeline-daily function a new row is created for the feature group where the I chose randomly to create a passenger that survived or a passenger that did not survived. Finally, The new row is inserted to the feature group. 

In the titanic-batch-inference-pipeline function, I get the rows from the feature view and use the model from hopsworks to make the predictions. Then, only the last predicted value is checked and based on that value I save the corresponding image in hopsworks, with the name latest_passenger.png. The next step is to check from the feature group the value of the survived label from the last insertion. As previously based on the value of the label we save a image with the name actual_fate.png.
Furthemore, a new feature group is created with the name titanic_predictions where the predicted values and the actual fate are stored with datetime as the primary key.
I read the last 5 predictions and save the table as an image with the name df_recent.png.
Finally, If there are 2 different values for the label prediction the confusion matrix is saved as an image with the name confusion_matrix.
