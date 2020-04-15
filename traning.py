import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import tensorflow as tf
import keras 
import cv2
import numpy as np
from math import cos, pi
import keras.backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard,LearningRateScheduler

from modelDef import create_monoDepth_Model
from lossFunctions import monoDepthV2Loss
from dataGen import depthDataGenerator

# define these
batchSize = 1
trainingRunDate = '2020_4_15'
Notes = 'Full_data_no_smoothness_custom_mu_lower learning'

# build data generators
train_generator = depthDataGenerator('../train/left/','../train/right/',batch_size=batchSize, shuffle=True, max_img_time_diff=700 )
val_generator  = depthDataGenerator('../val/left/', '../val/right/', batch_size=batchSize, shuffle=False, agumentations=False, max_img_time_diff=700)

# build loss
loss = monoDepthV2Loss(0.5,0.5,640,192,batchSize).applyLoss

# build model
model = create_monoDepth_Model(input_shape=(640,192,3), encoder_type=18)
model.compile(optimizer=Adam(lr=1e-3),loss=loss)

# lets define some callbacks
if not os.path.exists('models/' + Notes + '_' + trainingRunDate + '_batchsize_' + str(batchSize) + '/'):
    os.mkdir('models/' + Notes + '_' + trainingRunDate + '_batchsize_' + str(batchSize) + '/')
mc = ModelCheckpoint('models/' + Notes + '_' + trainingRunDate + '/_weights_epoch{epoch:02d}_val_loss_{val_loss:.4f}_train_loss_{loss:.4f}.hdf5', monitor='val_loss')
mc1 = ModelCheckpoint('models/' + Notes + '_' + trainingRunDate + '/_weights_epoch{epoch:02d}_val_loss_{val_loss:.4f}_train_loss_{loss:.4f}.hdf5', monitor='loss')
rl = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1)
tb = TensorBoard(log_dir='logs/' + Notes + '_' + trainingRunDate + '_batchsize_' + str(batchSize), update_freq=1000)

# Schedule Learning rate Callback
def lr_schedule(epoch):
    if epoch < 1:
        return 0.0 
    if epoch < 10:
        return ((epoch/10)**2) * 1e-4
    else:
        return cos((((epoch-10)%30)/30)*(pi/2)) * 1e-4

lr = LearningRateScheduler(schedule=lr_schedule,
                                                    verbose=1)


model.fit_generator(train_generator, epochs = 100, validation_data=val_generator, callbacks=[mc,mc1,rl,tb])

