## Distracted Driver Detection

### Project Definition
The aim of this project is given the dataset of driver images, each taken in a car with a driver doing something in the car (texting, eating, talking on the phone, makeup, reaching behind, etc), predict the likelihood of what the driver is doing in each picture. The 10 classes to predict are:
* c0: safe driving
* c1: texting - right
* c2: talking on the phone - right
* c3: texting - left
* c4: talking on the phone - left
* c5: operating the radio
* c6: drinking
* c7: reaching behind
* c8: hair and makeup
* c9: talking to passenger

### Dataset
The dataset used for this problem is the State Farm Distracted Driver dataset available on Kaggle.The dataset contains 102,150 JPG images divided into training and test data. Each image is sized at 640 x 480 pixels. The total dataset file size is 4GB.
Source: https://www.kaggle.com/c/state-farm-distracted-driver-detection/data

### Implementation
The Distracted Driver model was trained using Google Cloud Platform (GCP) AutoML Vision API. The image data set was loaded into a storage bucket and this was linked to AutoML vision to train the image classification model. After the model was trained, several metrics were listed to illustrate model performance. The model scored a Precision of 100% and Recall of 99.73%. A confusion matrix was also presented for further model performance analysis. The model was then deployed on the Cloud Platform to be accessed for predictions. 

The major components of the system were the image data set, trained AutoML model and the API which was served with the Flask microframework. For predictions on an image, the image file is parsed to the model which outputs a json file with the class prediction and prediction score (between 0 and 1 for the probability of the class prediction determined). 

A simple webpage was designed to access this application. A user can upload any JPG image of a driver on this web page to obtain a prediction on what activity the driver is performing. The results were extracted from the json file containing predictions and parsed to this web page using the POST method from Flask to display the model predictions. This web page is currently deployed to GCP App Engine.

All image data and model API is stored in a GCP storage bucket. Future images parsed to the model will also be retained in the storage bucket to add to existing data set for model retraining as needed.

### Link
The App Engine webpage can be accessed at "https://stable-hybrid-249623.appspot.com" <<currently disabled>>
