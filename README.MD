# Image Classifier Using Tensorflow <br>
This project is broken out into two objectives, the first is creating a jupyter notebook that creates a Tensorflow Keras model and a python application that runs on the command line. The project classifies flower images among 102 classes of different flowers and provides probabilities for each assignment.
 <br> <br>

## Motivation <br>
To create a real world application of deep neural networks. In this project I trained a image classifier to recognize 102 different flowers and then created an application to envision a potential app that would be used to classify flowers with a phone app.
  <br><br>


## Dataset <br>
The dataset used is [Oxford Flowers 102](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet).
<br><br>


## Files <br>
1. Image_Classifier.ipynb - Notebook used to create the Tensorflow Keras model.
2. predict.py - Python application that predicts image label using Keras model from the command line.
3. my_model.h5 - Best performing keras model created by the Image_Classifier notebook.
4. util.py - Helper functions used for predict.py
<br> <br>


## Objectives <br>
For the jupyter notebook, I used a pre-trained model MobileNet from Tensorflow Hub. I added keras layers and trained
the model. I split the data into training, validation and testing sets and organized with a pipeline in order to prepare it for model ingestion. When it finished training I exported the best model my_model.h5.

For the predict.py file, the code runs from the command line in the following manner utilizing my_model.h5.

![Basic Usage](https://github.com/jatp24/Image_Classification_TensorFlow/blob/master/readme_images/basic_usage.PNG)

* image_path - path of image to evaluate.
* model_path - path of tensorflow keras model.
* --top_k - top k highest probabilities.
* --category_names - json mapping file path for class ids to class names.

<br>

#### Example of output:<br>

![Predict.py output](https://github.com/jatp24/Image_Classification_TensorFlow/blob/master/readme_images/predict_print.PNG)


<br><br>
## Requirements: <br>
The following are requirements to run predict.py.

 ![Requirements](https://github.com/jatp24/Image_Classification_TensorFlow/blob/master/readme_images/requirements_image.PNG)
