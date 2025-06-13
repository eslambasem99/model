from __future__ import division, print_function
# coding=utf-8

import os
import numpy as np

# Keras
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.utils import load_img, img_to_array 

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'breast_cancer_model.h5'

# Load your trained model
model = load_model(MODEL_PATH)
# print('Model loaded. Start serving...')

print('Model loaded. Check http://127.0.0.1:5001/')

#function for processing the input image abd prediction
def model_predict(img_path, model):

    # Preprocessing the image
    img = load_img(img_path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    y = model.predict(x)

    return np.argmax(y)


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        

        # Process your result for human
        dic= {0:"Normal", 1:"Malignant",2:"Bengin"}
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        pred_class = dic[preds]   # ImageNet Decode
        # return the result
        return pred_class
    return None


if __name__ == '__main__':
    app.run(debug=True, port=5001)

