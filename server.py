"""
This program will create and/or train a model that will be trained to predict the following classes:
 - Dino (from the fiction movie 'Jurassic Park'
 - Lugia (from the anime 'Pokémon'
 - Taz (from the cartoon 'Looney Tunes'
"""

#import the required libraries
import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import matplotlib.pyplot as plt


#general declarations
TRAIN_DIR = r'C:\Users\kevinornales\Desktop\Machine Learning\Kinpo\Toys\Train'
TEST_DIR = r'C:\Users\kevinornales\Desktop\Machine Learning\Kinpo\Toys\Test'
IMG_SIZE = 100
LR = 1e-3
MODEL_NAME = 'Toys.model'.format(LR, '2conv-basic') 


#define a function to convert the training images into numerical data
def label_img(img):
    
    word_label = img.split('.')[-3]
    
    # value of array for a Dino                           
    if word_label == 'Dino': return [1,0,0]
    # value of array for a Lugia                           
    elif word_label == 'Lugia': return [0,1,0]
    # value of array for a Taz
    elif word_label == 'Taz': return [0,0,1]


#define a function to create the training dataset array
def create_train_data():
    
    training_data = []
    
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR,img)
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img),np.array(label)])
        
    shuffle(training_data)
    np.save('toys_train_data.npy', training_data)
    
    return training_data


#define a function to create the model 
def create_model():
    
    tf.reset_default_graph()
    
    convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

    convnet = conv_2d(convnet, 32, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 128, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 32, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)
    
    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, 3, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir='log')
    return model


#define a function to train the model
def trainModel(model, train_data):
    
    train = train_data[:345]
    test = train_data[-45:]

    X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    Y = [i[1] for i in train]

    test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    test_y = [i[1] for i in test]

    model.fit({'input': X}, {'targets': Y}, n_epoch=40, validation_set=({'input': test_x}, {'targets': test_y}), snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
    model.save(MODEL_NAME)



def main():
    #load or create the train data
    if os.path.exists('train_data.npy'):
        train_data = np.load('train_data.npy')
        print('Train Data Loaded!!')        
    else:
        train_data = create_train_data()        

    #load or create the model
    model = create_model()
    if os.path.exists('{}.meta'.format(MODEL_NAME)):
        model.load(MODEL_NAME)
        print('Model Loaded!')

    #train the model
    trainModel(model, train_data)


if __name__ == "__main__":
    main()	


