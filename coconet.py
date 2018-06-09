# Math
import numpy as np
from math import atan2

# Machine Learning
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

# Image Processing
from skimage.measure import compare_psnr, compare_ssim, compare_mse
import cv2

# Others
import os

def generate_placeholder_matrix(picture_sizex, picture_sizey):
    # Generate placeholder matrix with given dimensions
    X = []
    for x_it in range(0, picture_sizex):
        for y_it in range(0, picture_sizey):
            x0 = x_it + 0.5
            y0 = y_it + 0.5
            x = (x0 - picture_sizex / 2)
            y = (y0 - picture_sizey / 2)
            X.append((x0, y0, picture_sizex - x0, picture_sizey - y0, (x**2+y**2)**(1/2), atan2(y0, x0)))
    return np.asarray(X)

def generate_value_matrix(img, picture_sizex, picture_sizey):
    # Generate value matrix from image
    Y = []
    for x_iterator in range(0, picture_sizex):
        for y_iterator in range(0, picture_sizey):
            Y.append(np.multiply(1/255, img[x_iterator][y_iterator]))
    return np.asarray(Y)

def generate_model_dense(width_list):
    # Generate dense sequential model with fixed input and output and hidden layer widths from width_list
    model = Sequential()
    model.add(Dense(width_list[0], input_dim=6, activation = 'tanh', init = 'random_uniform'))
    for i in range(1, len(width_list)):
        model.add(Dense(width_list[i], activation = 'tanh'))
    model.add(Dense(3, activation = 'sigmoid'))
    model.compile(loss = 'mean_squared_error', optimizer = keras.optimizers.Adam(lr=0.001), metrics = ['accuracy'])
    model.save_weights('initial_weights.h5')
    return model

def load_image(address):
    # Load image as np.array and extract filename
    filename = os.path.basename(address)
    img = cv2.imread(address)
    return img, filename

def compare_images(img1, img2):
    # Compute PSNR, SSIM and MSE for 2 images
    psnr = compare_psnr(img1, img2)
    ssim = compare_ssim(img1, img2, multichannel=True)
    mse = compare_mse(img1, img2)
    return psnr, ssim, mse

def predict(model, X, picture_sizex, picture_sizey):
    # Predict
    prediction = model.predict(X)
    prediction = np.multiply(255, prediction)
    prediction = prediction.reshape(picture_sizex, picture_sizey, 3)
    return prediction.astype('uint8')
