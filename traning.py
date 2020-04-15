import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import tensorflow as tf
import keras 
import cv2
import numpy as np
import keras.backend as K
from keras.optimizers import Adam

from modelDef import create_monoDepth_Model
from lossFunctions import monoDepthV2Loss
from dataGen import depthDataGenerator

batchSize = 1

training_generator = depthDataGenerator('../validate/left/','../validate/right/',batch_size=batchSize, shuffle=False)
testing_generator  = depthDataGenerator('../test/left/', '      ../test/right/', batch_size=batchSize, shuffle=False)

loss = monoDepthV2Loss(0.5,0.5,640,192,batchSize).applyLoss

model = create_monoDepth_Model(input_shape=(640,192,3), encoder_type=18)
model.compile(optimizer=Adam(lr=1),loss=loss)
#model.compile(optimizer='adam',loss={'OutputConvBlock' : loss, 'upSampleSclae1Out': loss,\
#            'upSampleSclae2Out': loss, 'upSampleSclae3Out': loss})

model.fit_generator(training_generator, epochs = 100)

