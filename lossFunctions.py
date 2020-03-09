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
