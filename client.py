'''
This script will load images of toys from path and predict whether the images falls under a certain class:
 - Dino (from the fiction movie 'Jurassic Park'
 - Lugia (from the anime 'Pokémon'
 - Taz (from the cartoon 'Looney Tunes')

This script will create the output file of the predictions with labels and test output accuracy: output-file.csv
'''

#import the required libraries
import cv2
import numpy as np
import os
import random
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
MODEL_NAME = 'Toys-{}-{}.model'.format(LR, '2conv-basic') 


#define a function to create the test dataset array
def process_test_data():
    testing_data = []
    
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR,img)
        img_name = (img.split('.')[0],img.split('.')[1])
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img), img_name])
        
    shuffle(testing_data)
    np.save('toy_test_data.npy', testing_data)
    return testing_data


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


#define a function to classify the test images 	
def testModel(model, test_data):
    
    # Dino:  [1, 0, 0]
    # Lugia: [0, 1, 0]
    # Taz:   [0, 0, 1]

    #create Figure 1 of results
    fig1 = plt.figure()
	
    for num, data in enumerate(test_data[:12]):
        img_name = data[1]
        img_data = data[0]
        y = fig1.add_subplot(3,4,num+1)
        orig = img_data
        data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
        model_out = model.predict([data])[0]
        if np.argmax(model_out) == 0:
            str_label = 'Dino'
        elif np.argmax(model_out) == 1:
            str_label = 'Lugia'
        elif np.argmax(model_out) == 2:
            str_label = 'Taz'
        else:   str_label = 'Unknown'          
        
        y.imshow(orig)
        plt.title(str_label)
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)
                
    #create Figure 2 of results 
    fig2 = plt.figure()
    for num, data in enumerate(test_data[12:24]):
        img_name = data[1]
        img_data = data[0]
        y = fig2.add_subplot(3,4,num+1)
        orig = img_data
        data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
        model_out = model.predict([data])[0]
        if np.argmax(model_out) == 0:
            str_label = 'Dino'
        elif np.argmax(model_out) == 1:
            str_label = 'Lugia'
        elif np.argmax(model_out) == 2:
            str_label = 'Taz'
        else:   str_label = 'Unknown'          

        y.imshow(orig)
        plt.title(str_label)
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)


    #save the results on a spreadsheet
    with open('output-file.csv', 'w') as f:
        f.write('id,Dino,Lugia,Taz\n')
        
    with open('output-file.csv', 'a') as f:
        correct_predictions = 0
        total_predictions = 0
        
        for data in tqdm(test_data):
            img_name = data[1]
            img_data = data[0]
            orig = img_data
            data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
            model_out = model.predict([data])
            if np.argmax(model_out) == 0:
                str_label = 'Dino'
            elif np.argmax(model_out) == 1:
                str_label = 'Lugia'
            elif np.argmax(model_out) == 2:
                str_label = 'Taz'
            else:   str_label = 'Unknown'       
            f.write('{},{},{},{}\n'.format(img_name[0] + img_name[1], model_out[0][0], model_out[0][1], model_out[0][2]))
            if img_name[0] == str_label:
                correct_predictions += 1
            total_predictions += 1
                    
        if total_predictions == 0:
            accuracy = 0
        else:
            accuracy = (correct_predictions / total_predictions) * 100
            
        f.write('Test Accuracy : {} %\n'.format(accuracy))

    #show the plotted Figure 1 and Figure 2
    plt.show()        


def main():
    #load test dataset
        test_data = process_test_data()
        model = create_model()
	
        if os.path.exists('{}.meta'.format(MODEL_NAME)):
                model.load(MODEL_NAME)
                print('Model Loaded!')
               
    #classify the images based on the model
        testModel(model, test_data)


if __name__ == "__main__":
	main()	


