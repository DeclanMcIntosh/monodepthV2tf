'''

TODO:
    - all these at multiple scales of output of the network which are up-sampled 
        - Temporal min reporojection loss.
            - mask out pxiles which are different between samples
        - calculate per-pixle smoothness loss 

    
''' 

import keras
import keras.backend as K
import tensorflow as tf
import config


def findGradients(y_predicted, leftImgPyramid):
    '''
    parameters:
        y_predicted -  array of 4 scales of estimation with (Network dispartiy map of shape (height, width, 1))

        leftImgPyramid - up sampled pyramid of the image for the different scales
    '''
    # addapted from https://github.com/mtngld/monodepth-1/blob/1f1fc80ac0dc727f3de561ead89e6792aea5e178/monodepth_model.py#L109 
    def gradient_x(img):
        gx = img[:,:,:-1,:] - img[:,:,1:,:]
        return gx
    def gradient_y(img):
        gy = img[:,:-1,:,:] - img[:,1:,:,:]
        return gy

    image_gradients_x = [gradient_x(img) for img in leftImgPyramid]
    image_gradients_y = [gradient_y(img) for img in leftImgPyramid]
    
    dispGradientX = [gradient_x(d) for d in y_predicted]
    dispGradientY = [gradient_y(d) for d in y_predicted]
 
    weightX = [K.exp(-K.mean(K.abs(g), 3, keep_dims=True)) for g in image_gradients_x]
    weightY = [K.exp(-K.mean(K.abs(g), 3, keep_dims=True)) for g in image_gradients_y]

    smoothness_x = [dispGradientX[i] * weightX[i] for i in range(4)]
    smoothness_y = [dispGradientY[i] * weightY[i] for i in range(4)]
    return smoothness_x + smoothness_y

def smoothnessLoss(y_predicted, leftImage):
    '''
    parameters:
        y_predicted -  array of 4 scales of estimation with (Network dispartiy map of shape (height, width, 1))

        leftImage - Left Image
    '''

    # y_predicted: (1,881,400,1)
    # leftImage: (1, 881, 400, 3)

    # Create pyramid
    GausBlurKernel = K.expand_dims(K.expand_dims(K.constant([[1/16,1/8,1/16],[1/8,1/4,1/8],[1/16,1/8,1/16]]), -1),-1)
    # TF kernel shape: (rows, cols, input_depth, depth)
    # If this is converted to TH, TH kernel shape: (depth, input_depth, rows, cols)
    # GausBlurKernel (3,3)


    leftImgPyramid = K.expand_dims(K.mean(leftImage,axis=-1),-1) # scale colours to greyscale for gradients
    # leftImgPyramid (1, 881, 400, 1)

    #leftImgPyramid_1_down = K.conv2d(leftImgPyramid, GausBlurKernel, strides=tuple((1,1)), dilation_rate=tuple((1,1)))
    leftImgPyramid_1_down = K.conv2d(leftImgPyramid, GausBlurKernel)
    # Given an input tensor of shape [batch, in_height, in_width, in_channels] and 
    # a filter / kernel tensor of shape [filter_height, filter_width, in_channels, out_channels], this op performs the following: 
    #   1. Flattens the filter to a 2-D matrix with shape [filter_height * filter_width * in_channels, output_channels].
    #   2. Extracts image patches from the input tensor to form a virtual tensor of shape [batch, out_height, out_width, filter_height * filter_width * in_channels].
    #   3. For each patch, right-multiplies the filter matrix and the image patch vector.

    leftImgPyramid_2_down = K.conv2d(leftImgPyramid_1_down, GausBlurKernel)
    leftImgPyramid_3_down = K.conv2d(leftImgPyramid_2_down , GausBlurKernel)

    leftImgPyramid = [leftImgPyramid, leftImgPyramid_1_down, leftImgPyramid_2_down, leftImgPyramid_3_down]

    return sum([K.mean(K.abs(findGradients(y_predicted, leftImgPyramid)[i])) / 2 ** i for i in range(4)]) / 4.


def photoMetric(disp, left, right, width, height, batchsize):

    # Flatten and seperate out channels
    # [batch, width, height, channel]
    disp_f =     K.flatten( K.permute_dimensions(          disp, pattern=(0,2,1,3)))
    left_f_0 =   K.flatten( K.permute_dimensions( left[:,:,:,0], pattern=(0,2,1)))
    right_f_0 =  K.flatten( K.permute_dimensions(right[:,:,:,0], pattern=(0,2,1)))
    left_f_1 =   K.flatten( K.permute_dimensions( left[:,:,:,1], pattern=(0,2,1)))
    right_f_1 =  K.flatten( K.permute_dimensions(right[:,:,:,1], pattern=(0,2,1)))
    left_f_2 =   K.flatten( K.permute_dimensions( left[:,:,:,2], pattern=(0,2,1)))
    right_f_2 =  K.flatten( K.permute_dimensions(right[:,:,:,2], pattern=(0,2,1)))
    #print(K.eval(disp_f))
    #print(K.eval(left_f_0))
    # find the self-referantiatl indicies in the tensor
    indicies = K.arange(0,K.shape(disp_f)[0], dtype='float32')


    #print("indicies", K.eval(indicies))
    # offset the indicies by the disparities to make the reprojection referances for the left image

    right_referances = K.clip(indicies + (disp_f * -1 * width), 0, width*height*batchsize)

    # OK TO THIS POINT NO GRADS GET LOST
    intReferances = K.cast(tf.floor(right_referances), 'int32')

    #print("intReferances", K.eval(intReferances))

    # gather the values to creat the left re-projected images
    right_f_referance_to_projected_0 = K.gather(right_f_0, intReferances) # not differentiable due to cast operation
    #test2 = K.eval(right_referances)
    right_f_referance_to_projected_1 = K.gather(right_f_1, intReferances)
    right_f_referance_to_projected_2 = K.gather(right_f_2, intReferances)

    # get difference between original left and right images
    #L2Direct      = K.sqrt(  K.square(left_f_0 - right_f_0) 
    #                      +  K.square(left_f_1 - right_f_1) 
    #                      +  K.square(left_f_2 - right_f_2))
    L1Direct =  K.abs((left_f_0 - right_f_0)) \
              + K.abs((left_f_1 - right_f_1)) \
              + K.abs((left_f_2 - right_f_2))

    # develop mask for loss where the repojected loss is better than the direct comparision loss
    # minMask = K.cast(K.less(diffReproject, diffDirect), 'float32')

    #L2Reproject = K.sqrt(   K.square(left_f_0 - right_f_referance_to_projected_0) \
    #                      + K.square(left_f_1 - right_f_referance_to_projected_1) \
    #                      + K.square(left_f_2 - right_f_referance_to_projected_2) )
 
    L1Reproject =   K.abs(left_f_0 - right_f_referance_to_projected_0) \
                  + K.abs(left_f_1 - right_f_referance_to_projected_1) \
                  + K.abs(left_f_2 - right_f_referance_to_projected_2) 
    #print("L1Direct Loss ", K.eval(K.mean(L1Direct)))
    #print("L1Repoject Loss ", K.eval(K.mean(L1Reproject)))

    return L1Direct, L1Reproject * (right_referances /( right_referances + 1e-10))

    #test4 = K.eval(diffReproject)
    # develop mask for loss where the repojected loss is better than the direct comparision loss
    #minMask = K.less(L1Reproject, L1Direct)
    # apply mask
    #out = (L1Reproject/255.) * K.cast(minMask, 'float32') * (right_referances /( right_referances + 1e-10))

    # determine mean and normalize 
    #return (K.sum(out) / K.cast(tf.math.count_nonzero(out),dtype='float32'))


class monoDepthV2Loss():
    def __init__(self, mu, lambda_, width, height, batchsize):
        self.mu = mu
        self.lambda_ = lambda_
        self.width = width
        self.height = height
        self.batchsize = batchsize

    def applyLoss(self, y_true, y_pred):
        '''
        For photometric 
        get direct comparision loss left to right 
        get repojections pe values to t-1 t and t+1 from left
        K = take elementwise minimium between the reporjection losses
        mask with the minimum between the direct comparison and K 

        using mu mask as defined in paper works poorly due to our samples being so seperated in time

        '''
        # rename and split values
        # [batch, width, height, channel]
        left        = y_true[:,:,:,0:3 ]
        right_minus = y_true[:,:,:,3:6 ]
        right       = y_true[:,:,:,6:9 ]
        right_plus  = y_true[:,:,:,9:12]

        disp        = y_pred
        # up-sample disparities by a nearest interpolation scheme for comparision at highest resolution per alrogithm

        #L_s = smoothnessLoss(y_pred, left)
        #smoothnessLoss(disp,left)

        Direct, Reproject_0     = photoMetric(disp,left, right,       self.width, self.height, self.batchsize)
        Direct, Reproject_1     = photoMetric(disp,left, right_plus,  self.width, self.height, self.batchsize)
        Direct, Reproject_neg1  = photoMetric(disp,left, right_minus, self.width, self.height, self.batchsize)

        ReprojectedError = K.minimum(Reproject_0, Reproject_1)
        ReprojectedError = K.minimum(ReprojectedError, Reproject_neg1)

        mu_mask = K.cast(K.less(ReprojectedError, Direct), 'float32')

        mu_mask_custom = K.sqrt(K.mean(K.square(right_minus - right), axis=-1))
        mu_mask_custom = K.flatten(K.permute_dimensions( mu_mask_custom, pattern=(0,2,1)))
        mu_mask_custom = K.cast(K.less(ReprojectedError, mu_mask_custom ), 'float32')

        #ReprojectedError = mu_mask * ReprojectedError 
        ReprojectedError = mu_mask_custom * ReprojectedError 

        #L_p = K.mean(disp) # switching to this fixes out of bounds issue, will check to see if i can get this working 
        #return L_p* self.mu + L_s * self.lambda_
        return ReprojectedError

'''
TODO

get averaging along scales working, get final loss

test

get batch size != 1 working
'''

if __name__ == "__main__":

    leftImage  = '../validate/left/2018-07-16-15-37-46_2018-07-16-15-38-12-727.jpg'
    dispImage  = '../validate/disp/2018-07-16-15-37-46_2018-07-16-15-38-12-727.png' # actuall associated disparity
    dispImage1  = '../validate/disp/2018-07-16-15-37-46_2018-07-16-16-32-48-979.png' # bad disparity totally random
    rightImage = '../validate/right/2018-07-16-15-37-46_2018-07-16-15-38-12-727.jpg'

    # leftImage  = '../validate/left/2018-07-09-16-11-56_2018-07-09-16-11-56-502.jpg'
    # dispImage  = '../validate/disp/2018-07-09-16-11-56_2018-07-09-16-11-56-502.png' # actuall associated disparity
    # dispImage1  = '../validate/disp/2018-07-16-15-37-46_2018-07-16-16-32-48-979.png' # bad disparity totally random
    # rightImage = '../validate/right/2018-07-16-15-37-46_2018-07-16-15-38-12-727.jpg'

    import numpy as np
    import cv2

    left  = np.transpose(cv2.imread(leftImage),     axes=[1,0,2]).astype('float32')
    dispTrue  = np.transpose(cv2.imread(dispImage),     axes=[1,0,2]).astype('float32')[:,:,0] / 256.
    dispWrong = np.transpose(cv2.imread(dispImage1),    axes=[1,0,2]).astype('float32')[:,:,0] / 256.
    right = np.transpose(cv2.imread(rightImage),    axes=[1,0,2]).astype('float32')

    width = left.shape[0]
    height = left.shape[1]
    realOffset = 3

    # memes = np.arange(0,width)

    # left      = np.zeros(shape=(width,height,3)).astype('float32')
    # right     = np.zeros(shape=(width,height,3)).astype('float32')
    # dispTrue  = np.full(shape=(width,height,1),  fill_value=realOffset).astype('float32')    / width 
    # dispWrong = np.full(shape=(width,height,1),  fill_value=realOffset -1).astype('float32') / width

    # for _ in range(height):
    #     left[:,_,0] = memes
    #     left[:,_,1] = memes
    #     left[:,_,2] = memes

    #     right[0:width-realOffset,_,0] = memes[realOffset:]
    #     right[0:width-realOffset,_,1] = memes[realOffset:]
    #     right[0:width-realOffset,_,2] = memes[realOffset:]

    
    # left.reshape(1,  left.shape[0], left.shape[1], left.shape[2]  )
    # dispTrue.reshape(1,  dispTrue.shape[0], dispTrue.shape[1], 1)
    # dispWrong.reshape(1,  dispWrong.shape[0], dispWrong.shape[1], 1)
    # right.reshape(1, right.shape[0], right.shape[1], right.shape[2]  )
    
    leftImage_tensor  = tf.expand_dims(tf.convert_to_tensor(left), 0)
    rightImage_tensor = tf.expand_dims(tf.convert_to_tensor(right), 0)
    dispImage_tensor  = tf.expand_dims(tf.expand_dims(tf.convert_to_tensor(dispTrue), 0), -1)
    dispImage_tensor1 = tf.expand_dims(tf.expand_dims(tf.convert_to_tensor(dispWrong), 0), -1)

    Lp  = photoMetric(dispImage_tensor,  leftImage_tensor, rightImage_tensor, width, height, 1)
    Lp1 = photoMetric(dispImage_tensor1, leftImage_tensor, rightImage_tensor, width, height, 1)

    # print("good")
    # print(K.eval(Lp))
    # print("bad")
    # print(K.eval(Lp1))

    #disp1 = np.random.uniform(size=disp.shape).astype('float32')
#
    #left.reshape(1,  left.shape[0], left.shape[1], left.shape[2]  )
    #disp.reshape(1,  disp.shape[0], disp.shape[1], 1)
    ##dispO.reshape(1,  dispO.shape[0], dispO.shape[1], 1)
    #disp1.reshape(1,  disp1.shape[0], disp1.shape[1], 1)
    #right.reshape(1, right.shape[0], right.shape[1], right.shape[2]  )
#
    #leftImage_tensor  = tf.expand_dims(tf.convert_to_tensor(left), 0)
    #rightImage_tensor = tf.expand_dims(tf.convert_to_tensor(right), 0)
    #dispImage_tensor  = tf.expand_dims(tf.expand_dims(tf.convert_to_tensor(disp), 0), -1)
    #dispImage_tensor1 = tf.expand_dims(tf.expand_dims(tf.convert_to_tensor(disp1), 0), -1)
    ##dispImage_tensorO = tf.expand_dims(tf.expand_dims(tf.convert_to_tensor(dispO), 0), -1)
#
    #Lp  = photoMetric(dispImage_tensor,  leftImage_tensor, rightImage_tensor, left.shape[1], left.shape[2], 1)
    #Lp1 = photoMetric(dispImage_tensor1, leftImage_tensor, rightImage_tensor, left.shape[1], left.shape[2], 1)
    ##LpO = photoMetric(dispImage_tensorO, leftImage_tensor, rightImage_tensor, left.shape[1], left.shape[2], 1)
#
    #print("good")
    #print(K.eval(Lp))
    #print("random")
    #print(K.eval(Lp1))
    #print("other")
    #print(K.eval(LpO))  

    print("smoothness good test")
    smoothness = smoothnessLoss(dispImage_tensor,leftImage_tensor)
    print(K.eval(smoothness))
