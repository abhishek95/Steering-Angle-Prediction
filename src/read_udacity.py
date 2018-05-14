# This code is to get data from the dataset and put it in order

import numpy as np
import matplotlib.image as mpimg

import os
import csv

read_test_data()

def read_training_data():
    # Directories for csv file and images
    directory = "../datasets/steering angle/udacity/Ch2_002/center/"
    dir_csv = "../datasets/steering angle/udacity/Ch2_002/"
    # Create dictionary for filename and its steering angle
    dictt = {}
    f = open(dir_csv + 'interpolated.csv', 'rb')
    reader = csv.reader(f)
    for row in reader:
        string = str(row[5])
        if ('center' in string):
            string = string[7:]
            dictt[string] = row[6]
    f.close()
    
    # Get number of files in directory and shape of sample image
    path, dirs, files = os.walk(directory).next()
    #img = mpimg.imread(directory + "1479424215880976321.jpg")
    # Create placeholders for X and Y
    #X = np.zeros((len(files), img.shape[0], img.shape[1], img.shape[2]))
    X = []
    Y = []
    # Iterate and form dataset
    i = 0
    counter=0
    for filename in os.listdir(directory):   
        if (filename != '.DS_Store'):
            counter+=1                            ########uncomment the counter for full dataset#####
            if (counter==100):                      #################################################
                break
            image = mpimg.imread(directory + filename)
            #print ('image',filename,'read')
            angle = dictt[filename]
            #print ('angle',filename,'found')
            angle = float(angle)
                
            X.append(image)
            Y.append(angle)
            i+=1
        
    #X = np.fromiter(X.itervalues(),dtype = float,count = len(X))
    X = np.array(X)
    Y = np.array(Y)
    # Save the files
    np.save("X_center", X)
    np.save("Y_center", Y)
    
    
def read_test_data():
     # Directories for csv file and images
    
    directory = "../datasets/steering angle/udacity/Ch2_001/center/"
    dir_csv = "../datasets/steering angle/udacity/Ch2_001/"
    
    
    # Create dictionary for filename and its steering angle
    
    dictt = {}
    
    f = open(dir_csv + 'CH2_final_evaluation.csv', 'rb')
    reader = csv.reader(f)
    
    for row in reader:
        if ('frame_id' in row[0]):
            continue
        string = str(row[0])
        dictt[string+'.jpg'] = row[1]
    f.close()
    
    # Get number of files in directory and shape of sample image
    path, dirs, files = os.walk(directory).next()
    #img = mpimg.imread(directory + "1479424215880976321.jpg")
    # Create placeholders for X and Y
    
    #X = np.zeros((len(files), img.shape[0], img.shape[1], img.shape[2]))
    X = []
    Y = []
    
    
    
    # Iterate and form dataset
    
    i = 0
    counter=0
    for filename in os.listdir(directory):   
        if (filename != '.DS_Store'):
            counter+=1                            ########uncomment the counter for full dataset#####
            if (counter==100):                      #################################################
                break
            image = mpimg.imread(directory + filename)
            #print ('image',filename,'read')
            angle = dictt[filename]
            #print ('angle',filename,'found')
            angle = float(angle)
                
            X.append(image)
            Y.append(angle)
            i+=1
        
    #X = np.fromiter(X.itervalues(),dtype = float,count = len(X))
    X = np.array(X)
    Y = np.array(Y)
    
    # Save the files
    
    np.save("X_test", X)
    np.save("Y_test", Y)
    return



    
    
