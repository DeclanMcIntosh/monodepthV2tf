import keras
import os
import random
import numpy as np
'''

Declan McIntosh Robert Lee Data Generator for MonodepthV2 Keras/Tf implementation

'''

def get_input(path,imageSize):
    img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (imageSize,imageSize))/255.
    return img 

def get_output(path, imageSize,annotationFiles):
    labels = cv2.resize(cv2.imread(annotationFiles+path, cv2.IMREAD_GRAYSCALE), (imageSize,imageSize))/255.
    return labels

def preprocess_input(image, randomVals):
    '''
    preforms data augmentations listed below each with chance == 50%
    - Horiontal flip
    - Random brightness +- 0.2
    - Random contrast +- 0.2
    - Random saturation +- 0.2
    - Hue Jitter +- 0.1

    all random values provided in range (0,1)
    '''
    if randomVals[0] > 0.5:
        # flip image horizontally
        image = np.flip(image, 1)
    if randomVals[1] > 0.5:
        # increase/ decrease contrast
        image = np.clip(image * (0.8 + randomVals[2]/2.5), a_max=255)
    # Convert image to HSV for some transformations
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if randomVals[3] > 0.5:
        # change brightness of image
        hsv_image[:,:,2] += (((randomVals[4]/2.5 )- 0.2 ) * 255.) 
        hsv_image[:,:,2] = np.clip(hsv_image[:,:,2], a_min = 0, a_max = 255) 
    if randomVals[5] > 0.5:
        # change staturation
        hsv_image[:,:,1] += (((randomVals[6]/2.5 )- 0.2 ) * 255.)
        hsv_image[:,:,1] = np.clip(hsv_image[:,:,1], a_min = 0, a_max = 255) 
    if randomVals[7] > 0.5:
        # change Hue
        hsv_image[:,:,0] += (((randomVals[8]/2.5 )- 0.2 ) * 179.)
        hsv_image[:,:,0] = np.clip(hsv_image[:,:,0], a_min = 0, a_max = 179) 
    # Convert image back from HSV
    image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return image

def preprocess_output(anno, randomVals):
    if randomVals[0] > 0.5:
        # flip image horizontally
        image = np.flip(image, 1)
    return anno

def image_generator(files, annotationFiles, batch_size = 64, seed=100, imageSize=256):
    '''
    Keras data generator function which continuously yeilds new input data, 
    order of data is randomized after each pass over entire dataset.
    '''
    random.seed(seed)
    filesSub = np.array(random.shuffle(os.listdir(files),random=random.random()))
    counter = 0
    while True:
        x = 0
        batch_paths = []
        while x < batch_size:
            if filesSub[counter] != 'desktop.ini':
                batch_paths.append(filesSub[counter])
                x += 1   
            counter = (counter + 1) 
            if counter % (filesSub.shape[0]) == 0:
                filesSub = np.array(random.shuffle(os.listdir(files),random=random.random()))
        batch_input  = []
        batch_output = [] 
        
        for input_path in batch_paths:
            input_img = get_input(files + input_path, imageSize)
            output = get_output(input_path, annotationFiles=annotationFiles, imageSize=imageSize)
            randomVals = []
            for x in range(0,9):
                randomVals.append(random.random())
            input_img = preprocess_input(image=input_img, randomVals=randomVals)
            output = preprocess_output(anno=output, randomVals=randomVals)
            batch_input += [ input_img ]
            batch_output += [ output ]

        batch_x = np.array( batch_input )
        batch_y = np.array( batch_output )
    
        yield( batch_x, batch_y )