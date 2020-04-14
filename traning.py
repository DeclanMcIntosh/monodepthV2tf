import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import tensorflow as tf
import keras 
import cv2
import numpy as np
import keras.backend as K

from modelDef import create_monoDepth_Model
from lossFunctions import monoDepthV2Loss
from dataGen import depthDataGenerator

training_generator = depthDataGenerator('../validate/left/','../validate/right/',batch_size=8)
testing_generator  = depthDataGenerator('../validate/left/', '../validate/right/',batch_size=8)

loss = monoDepthV2Loss(0.5,0.5,640,192).applyLoss

model = create_monoDepth_Model(input_shape=(640,192,3), encoder_type=18)
model.compile(optimizer='adam',loss=loss)
#model.compile(optimizer='adam',loss={'OutputConvBlock' : loss, 'upSampleSclae1Out': loss,\
#            'upSampleSclae2Out': loss, 'upSampleSclae3Out': loss})

model.fit_generator(training_generator)

