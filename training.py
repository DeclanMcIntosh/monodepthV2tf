import os
import tensorflow as tf
import keras 
import cv2
import numpy as np
from math import cos, pi
import keras.backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard,LearningRateScheduler
from keras.utils import multi_gpu_model

from modelDef import create_monoDepth_Model
from lossFunctions import monoDepthV2Loss
from dataGen import depthDataGenerator

# define these
batchSize = 12
trainingRunDate = '2020_4_26'
Notes = 'Full_data_no_mu_on_left_right_only_full_loss_smoothness_0_3_disparity_no_scalling_res_18_no_SSI_nosmoothness_just_reprojections_'

# build data generators
train_generator = depthDataGenerator('../train/left/','../train/right/', batch_size=batchSize, shuffle=True, max_img_time_diff=700 )
val_generator  = depthDataGenerator('../val/left/', '../val/right/', batch_size=batchSize, shuffle=False, agumentations=False, max_img_time_diff=700)

# build loss
lossClasss = monoDepthV2Loss(0.001,0.85,640,192,batchSize)
loss = lossClasss.applyLoss 
repoLoss = lossClasss.fullReprojection
smoothLoss = lossClasss.fullSmoothnessLoss

# build model
model = create_monoDepth_Model(input_shape=(640,192,3), encoder_type=18)
model.compile(optimizer=Adam(lr=1e-3),loss=loss, metrics=[repoLoss,smoothLoss])

# lets define some callbacks
if not os.path.exists('models/' + Notes + '_' + trainingRunDate + '_batchsize_' + str(batchSize) + '/'):
    os.makedirs('models/' + Notes + '_' + trainingRunDate + '_batchsize_' + str(batchSize) + '/')
mc = ModelCheckpoint('models/' + Notes + '_' + trainingRunDate +  '_batchsize_' + str(batchSize) + '/_weights_epoch{epoch:02d}_val_loss_{val_loss:.4f}_train_loss_{loss:.4f}.hdf5', monitor='val_loss')
mc1 = ModelCheckpoint('models/' + Notes + '_' + trainingRunDate +  '_batchsize_' + str(batchSize) + '/_weights_epoch{epoch:02d}_val_loss_{val_loss:.4f}_train_loss_{loss:.4f}.hdf5', monitor='loss')
rl = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1) # not used
tb = TensorBoard(log_dir='logs/' + Notes + '_' + trainingRunDate + '_batchsize_' + str(batchSize), update_freq=250)

# Schedule Learning rate Callback
def lr_schedule(epoch):
    if epoch < 15:
        return 1e-3 
    else:
        return 1e-4

lr = LearningRateScheduler(schedule=lr_schedule,verbose=1)

model.fit_generator(train_generator, epochs = 20, validation_data=val_generator, callbacks=[mc,mc1,lr,tb], initial_epoch=0)

