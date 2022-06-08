from flask import Flask,render_template, request
import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    #Load and save the image to be able to use it..
    img = request.files["img"]
    imgName = img.filename
    imgpath = "./static/styles/Images/I" + img.filename
    img.save(imgpath)
    model = tf.keras.models.load_model('../mnist_clothes/reduced_model_with_drop')
    img = load_img(imgpath, color_mode = "grayscale", target_size=(28, 28),interpolation='nearest')
    # img.show()
    img = img_to_array(img)
    img = img.reshape(1, 28, 28, 1)
    img = img.astype('float32')
    img = 255-img
    predict = model.predict([img])
    labels = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
    }  
    
    answer = labels[np.argmax(predict)]

    return render_template('index.html', finalAnswer = answer, imgPath = imgpath , imgName = imgName)

    
if __name__ == '__main__':
    app.run(port=3000, debug = True)