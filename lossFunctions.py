'''

TODO:
    - all these at multiple scales of output of the network which are up-sampled 
        - Temporal min reporojection loss.
            - mask out pxiles which are different between samples
        - calculate per-pixle smoothness loss 

    
''' 

import keras
import keras.backend as K


def photometricLRLoss(y_predicted, y_actual):
    '''
    parameters:
        y_predicted -  array of 4 scales of estimation with (Network dispartiy map of shape (height, width, 1))

        y_actual    - Four referance images (no datagen modifications) of shape (height, width, 12)
    '''
    pass

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
    # Create pyramid
    GausBlurKernel = K.constant([[1/16,1/8,1/16],[[1/8,1/4,1/8]],[1/16,1/8,1/16]])
    leftImgPyramid        = leftImage
    leftImgPyramid_1_down = K.conv2d(leftImgPyramid, GausBlurKernel)
    leftImgPyramid_2_down = K.conv2d(leftImgPyramid_1_down, GausBlurKernel)
    leftImgPyramid_3_down = K.conv2d(leftImgPyramid_2_down , GausBlurKernel)

    leftImgPyramid = [leftImgPyramid, leftImgPyramid_1_down, leftImgPyramid_2_down, leftImgPyramid_3_down]

    return [K.mean(K.abs(findGradients(y_predicted, leftImgPyramid)[i])) / 2 ** i for i in range(4)]


def photoMetric(disp, left, right):

    # Flatten and seperate out channels
    disp_f =   K.flatten(disp)
    left_f_0 =   K.flatten( left[:,:,0])
    right_f_0 =  K.flatten(right[:,:,0])
    left_f_1 =   K.flatten( left[:,:,1])
    right_f_1 =  K.flatten(right[:,:,1])
    left_f_2 =   K.flatten( left[:,:,2])
    right_f_2 =  K.flatten(right[:,:,2])

    # find the self-referantiatl indicies in the tensor
    indicies = K.arange(0,K.shape(disp_f)[0], dtype='float64')

    # offset the indicies by the disparities to make the reprojection referances for the left image
    right_referances = K.clip(K.update_add(indicies, disp_f * -1. * K.shape(disp_f)[0]), 0, K.shape(disp_f)[0])

    # gather the values to creat the left re-projected images
    right_f_referance_to_projected_0 = K.gather(right_f_0, K.cast(right_referances, 'int64'))
    right_f_referance_to_projected_1 = K.gather(right_f_1, K.cast(right_referances, 'int64'))
    right_f_referance_to_projected_2 = K.gather(right_f_2, K.cast(right_referances, 'int64'))

    # get difference between original left and right images
    diffDirect      = K.abs(left_f_0 - right_f_0) + K.abs(left_f_1 - right_f_1) + K.abs(left_f_2 - right_f_2)/3.

    # get difference between right and left reprojected images
    diffReproject   = K.abs(left_f_0 - right_f_referance_to_projected_0) + K.abs(left_f_1 - right_f_referance_to_projected_1) + K.abs(left_f_2 - right_f_referance_to_projected_2)/3.

    # develop mask for loss where the repojected loss is better than the direct comparision loss
    minMask = K.cast(K.less(diffReproject, diffDirect), 'float64')

    # apply mask
    out = diffReproject * minMask

    # determine mean and normalize 
    return (K.sum(out) / K.sum(minMask))/255.


class monoDepthV2Loss():
    def __init__(self, mu, lambda_):
        self.mu = mu
        self.lambda_ = lambda_

    def applyLoss(self, y_true, y_pred):
        # rename and split values
        left        = y_true[0]
        right_minus = y_true[1]
        right       = y_true[2]
        right_plus  = y_true[3]

        disp