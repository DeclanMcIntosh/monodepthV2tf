import os
import tensorflow as tf
import keras 
import cv2
import random
import numpy as np
from math import cos, pi
import keras.backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard, LearningRateScheduler
from keras.utils import multi_gpu_model

from modelDef import create_monoDepth_Model
from lossFunctions import monoDepthV2Loss
from dataGen import depthDataGenerator
from evaluationMetrics import generateARDmetric, absoluteRelativeSqrd

# flags
visualize = False

# define these
batchSize = 1

# build loss
lossClasss = monoDepthV2Loss(0.001,0.85,640,192,batchSize)
loss = lossClasss.applyLoss 
lossL1 = lossClasss.applyLossL1 
repoLossL1 = lossClasss.fullReprojectionL1 
repoLoss = lossClasss.fullReprojection
smoothLoss = lossClasss.fullSmoothnessLoss

# build model
model = create_monoDepth_Model(input_shape=(640,192,3), encoder_type=18)

model.compile(optimizer=Adam(lr=1e-3),loss=loss, metrics=[repoLoss, smoothLoss, repoLossL1, lossL1])

def evaluateModel(model,batchSize, visualize):
    val_generator  = depthDataGenerator('../test/left/', '../test/right/', batch_size=batchSize, shuffle=False, agumentations=False, max_img_time_diff=700)
    scores = model.evaluate_generator(val_generator, verbose=1)
    print("Total Loss, Reprojection Loss, Smoothness Loss, L1 Reprojection Loss, L1 Total Loss")
    print(scores)
    ARD = 0
    count = 0 
    ABS = 0 
    SQR = 0 
    # Random Qualitative Evaluation
    imageName = random.choice(os.listdir('../test/left/'))
    rawImage = cv2.resize(cv2.imread('../test/left/' + imageName), (640,192))
    inputImg  = np.transpose(cv2.resize(cv2.imread('../test/left/' + imageName), (640,192)).astype('float32'),      axes=[1,0,2])
    output = model.predict(np.expand_dims(inputImg,0))# * 640 * 0.3
    def displayOutput(output, scale):
        outputTransformed = np.transpose(  output[0,:,:,scale],    axes=[1,0])
        outputTransformed = outputTransformed - np.mean(outputTransformed)
        outputTransformed = np.clip(outputTransformed, (np.mean(outputTransformed) - np.std(outputTransformed)), (np.mean(outputTransformed) + np.std(outputTransformed)))
        outputTransformed = outputTransformed - np.min(outputTransformed)
        outputTransformed = np.clip(outputTransformed / np.max(outputTransformed) * 255, 0, 255).astype('uint8')
        return outputTransformed

    outputDisplayScale0 = displayOutput(output, 0)
    outputDisplayScale1 = displayOutput(output, 1)
    outputDisplayScale2 = displayOutput(output, 2)
    outputDisplayScale3 = displayOutput(output, 3)

    if visualize:
        cv2.imshow("Input Image", rawImage)
        cv2.imshow("Display Disparity No Downscale",  outputDisplayScale0)
        cv2.imshow("Display Disparity 1/2 Downscale", outputDisplayScale1)
        cv2.imshow("Display Disparity 1/4 Downscale", outputDisplayScale2)
        cv2.imshow("Display Disparity 1/8 Downscale", outputDisplayScale3)

        #cv2.imwrite("../Images/InputImages.png",  rawImage )
        #cv2.imwrite("../Images/Display Disparity No Downscale.png",  outputDisplayScale0 )
        #cv2.imwrite("../Images/Display Disparity 1_2 Downscale.png", outputDisplayScale1 )
        #cv2.imwrite("../Images/Display Disparity 1_4 Downscale.png", outputDisplayScale2 )
        #cv2.imwrite("../Images/Display Disparity 1_8 Downscale.png", outputDisplayScale3 )
        cv2.waitKey(-1)


    # actual Evaluation
    imgs = os.listdir('../test/left/')
    print("")
    for filename in os.listdir('../test/left/'):
        inputImg    = np.transpose(cv2.resize(cv2.imread('../test/left/' + filename), (640,192)).astype('float32'),      axes=[1,0,2])
        inputShape = cv2.imread('../test/disp/' + filename[:-4] + '.png', cv2.IMREAD_ANYDEPTH).shape[1]
        debug = cv2.imread('../test/disp/' + filename[:-4] + '.png', cv2.IMREAD_ANYDEPTH)
        groundTruth = np.transpose((cv2.resize(cv2.imread('../test/disp/' + filename[:-4] + '.png', cv2.IMREAD_ANYDEPTH), (640,192), interpolation=cv2.INTER_NEAREST).astype('float32')/256.) * (640/inputShape) , axes=[1,0])
        output = model.predict(np.expand_dims(inputImg,0)) * 640 * 0.3
        ARD += generateARDmetric(output[0,:,:,0], groundTruth)
        ABS_, SQR_ = absoluteRelativeSqrd(output[0,:,:,0], groundTruth)
        ABS += ABS_
        SQR += SQR_
        count += 1
        print(count, " of ", len(imgs) , end='\r')
    print("Mean ARD: ", ARD / count)
    print("Mean SQR: ", SQR / count)
    print("")


print("Testing model trained on full loss with SSIM")
model.load_weights("models/Full_data_no_mu_with_SSIM_on_left_right_only_full_loss_smoothness_0_3_disparity_scalling_res_18_bugfix_2__2020_4_19_batchsize_12/_weights_epoch20_val_loss_1.7095_train_loss_1.7080.hdf5")
evaluateModel(model,batchSize, visualize)

print("Testing model trained on just L1 distance Loss")
model.load_weights("models/Full_data_no_mu_with_SSIM_on_left_right_only_full_loss_smoothness_0_3_disparity_scalling_res_18_no_SSI__2020_4_19_batchsize_12/_weights_epoch20_val_loss_0.0662_train_loss_0.0556.hdf5")
evaluateModel(model,batchSize, visualize)




