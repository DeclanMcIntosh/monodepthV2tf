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
    GausBlurKernel = K.expand_dims(K.constant([[1/16,1/8,1/16],[1/8,1/4,1/8],[1/16,1/8,1/16]]), 0)
    leftImgPyramid = K.expand_dims(K.mean(leftImage,axis=-1),-1) # convert to greyscale for gradients
    #leftImgPyramid_1_down = K.conv2d(leftImgPyramid, GausBlurKernel, strides=tuple((1,1)), dilation_rate=tuple((1,1)))
    leftImgPyramid_1_down = K.conv2d(leftImgPyramid, GausBlurKernel)
    leftImgPyramid_2_down = K.conv2d(leftImgPyramid_1_down, GausBlurKernel)
    leftImgPyramid_3_down = K.conv2d(leftImgPyramid_2_down , GausBlurKernel)

    leftImgPyramid = [leftImgPyramid, leftImgPyramid_1_down, leftImgPyramid_2_down, leftImgPyramid_3_down]

    return sum([K.mean(K.abs(findGradients(y_predicted, leftImgPyramid)[i])) / 2 ** i for i in range(4)]) / 4.


def photoMetric(disp, left, right, width, height, batchsize):

    # Flatten and seperate out channels
    # [batch, width, height, channel]
    disp_f =     K.flatten(disp)
    print(K.eval(disp_f))
    left_f_0 =   K.flatten( left[:,:,:,0]) 
    right_f_0 =  K.flatten(right[:,:,:,0])
    left_f_1 =   K.flatten( left[:,:,:,1])
    right_f_1 =  K.flatten(right[:,:,:,1])
    left_f_2 =   K.flatten( left[:,:,:,2])
    right_f_2 =  K.flatten(right[:,:,:,2])

    disp_shape = K.shape(disp)

    # find the self-referantiatl indicies in the tensor
    indicies = K.arange(0,K.shape(disp_f)[0], dtype='float32')

    # offset the indicies by the disparities to make the reprojection referances for the left image
    right_referances = K.clip(indicies + (disp_f * 1 * width), 0, width*height*batchsize)

    #right_referances = K.clip(indicies + (disp_f * -1 * K.cast(disp_shape[1], 'float32')), 0, K.eval(disp_shape[0]*disp_shape[1]))

    # OK TO THIS POINT NO GRADS GET LOST
    intReferances = K.cast(tf.floor(right_referances), 'int32')


    # gather the values to creat the left re-projected images
    right_f_referance_to_projected_0 = K.gather(right_f_0, intReferances) # not differentiable due to cast operation
    #test2 = K.eval(right_referances)
    right_f_referance_to_projected_1 = K.gather(right_f_1, intReferances)
    right_f_referance_to_projected_2 = K.gather(right_f_2, intReferances)

    # get difference between original left and right images
    diffDirect      = (K.abs(left_f_0 - right_f_0) 
                    +  K.abs(left_f_1 - right_f_1) 
                    +  K.abs(left_f_2 - right_f_2))/3.

    # develop mask for loss where the repojected loss is better than the direct comparision loss
    # minMask = K.cast(K.less(diffReproject, diffDirect), 'float32')

    diffReproject   =   ( K.abs(left_f_0 - right_f_referance_to_projected_0) \
                        + K.abs(left_f_1 - right_f_referance_to_projected_1) \
                        + K.abs(left_f_2 - right_f_referance_to_projected_2) ) /3.

    #test4 = K.eval(diffReproject)
    # develop mask for loss where the repojected loss is better than the direct comparision loss
    minMask = K.less(diffReproject, diffDirect)
    # apply mask
    out = (diffReproject/255.) * K.cast(minMask, 'float32') * (right_referances /( right_referances + 1e-10))

    # determine mean and normalize 
    return (K.sum(out) / K.cast(tf.math.count_nonzero(out),dtype='float32'))


class monoDepthV2Loss():
    def __init__(self, mu, lambda_, width, height, batchsize):
        self.mu = mu
        self.lambda_ = lambda_
        self.width = width
        self.height = height
        self.batchsize = batchsize

    def applyLoss(self, y_true, y_pred):
        # rename and split values
        # [batch, width, height, channel]
        left        = y_true[:,:,:,0:3 ]
        right_minus = y_true[:,:,:,3:6 ]
        right       = y_true[:,:,:,6:9 ]
        right_plus  = y_true[:,:,:,9:12]

        disp        = y_pred
        # up-sample disparities by a nearest interpolation scheme for comparision at highest resolution per alrogithm

        L_s = smoothnessLoss(y_pred, left)
        #smoothnessLoss(disp,left)

        L_p  = photoMetric(disp,left,right, self.width, self.height, self.batchsize)
        #L_p = K.mean(disp) # switching to this fixes out of bounds issue, will check to see if i can get this working 
        #return L_p* self.mu + L_s * self.lambda_
        return L_p

'''
TODO

get averaging along scales working, get final loss

test
'''

# import cv2
# if __name__ == "__main__":
#     print('Importing sample images...')
#     left_path = '../Photometric Loss Sample/2018-10-17-14-35-33_2018-10-17-14-36-11_359_left.jpg'
#     right_path = '../Photometric Loss Sample/2018-10-17-14-35-33_2018-10-17-14-36-11-359_right.jpg'
#     disp_path = '../Photometric Loss Sample/2018-10-17-14-35-33_2018-10-17-14-36-11-359.png'
#     image_size=(640,192)

#     leftImg = cv2.imread(left_path)
    
#     left        = cv2.resize(cv2.imread(left_path), dsize=image_size).astype('float32')
#     right       = cv2.resize(cv2.imread(right_path), dsize=image_size).astype('float32')
#     disp        = cv2.resize(cv2.imread(disp_path, cv2.IMREAD_UNCHANGED), dsize=image_size).astype('float32')

#     out = photoMetric(disp, left, right)

#     print(out)

if __name__ == "__main__":
    leftImage  = '../validate/left/2018-07-09-16-11-56_2018-07-09-16-11-56-502.jpg'
    dispImage  = '../validate/disp/2018-07-09-16-11-56_2018-07-09-16-11-56-502.png' # actuall associated disparity
    dispImage1  = '../validate/disp/2018-07-16-15-37-46_2018-07-16-16-32-48-979.png' # bad disparity totally random
    rightImage = '../validate/right/2018-07-09-16-11-56_2018-07-09-16-11-56-502.jpg'

    import numpy as np
    import cv2

    left  = np.transpose(cv2.imread(leftImage),    axes=[1,0,2]).astype('float32')
    disp  = np.transpose(cv2.imread(dispImage),    axes=[1,0,2]).astype('float32')[:,:,0] / 256.
    dispO = np.transpose(cv2.imread(dispImage1),    axes=[1,0,2]).astype('float32')[:,:,0] / 256.
    right = np.transpose(cv2.imread(rightImage),   axes=[1,0,2]).astype('float32')

    disp1 = np.random.uniform(size=disp.shape).astype('float32')

    left.reshape(1,  left.shape[0], left.shape[1], left.shape[2]  )
    disp.reshape(1,  disp.shape[0], disp.shape[1], 1)
    dispO.reshape(1,  dispO.shape[0], dispO.shape[1], 1)
    disp1.reshape(1,  disp1.shape[0], disp1.shape[1], 1)
    right.reshape(1, right.shape[0], right.shape[1], right.shape[2]  )

    leftImage_tensor  = tf.expand_dims(tf.convert_to_tensor(left), 0)
    rightImage_tensor = tf.expand_dims(tf.convert_to_tensor(right), 0)
    dispImage_tensor  = tf.expand_dims(tf.expand_dims(tf.convert_to_tensor(disp), 0), -1)
    dispImage_tensor1 = tf.expand_dims(tf.expand_dims(tf.convert_to_tensor(disp1), 0), -1)
    dispImage_tensorO = tf.expand_dims(tf.expand_dims(tf.convert_to_tensor(dispO), 0), -1)

    Lp  = photoMetric(dispImage_tensor,  leftImage_tensor, rightImage_tensor, left.shape[0], left.shape[1], 1)
    Lp1 = photoMetric(dispImage_tensor1, leftImage_tensor, rightImage_tensor, left.shape[0], left.shape[1], 1)
    LpO = photoMetric(dispImage_tensorO, leftImage_tensor, rightImage_tensor, left.shape[0], left.shape[1], 1)
    print("good")
    print(K.eval(Lp))
    print("random")
    print(K.eval(Lp1))
    print("other")
    print(K.eval(LpO))  

    print("smootheness good test")
    smoothness = smoothnessLoss(dispImage_tensor,leftImage_tensor[0,:,:])
    print(K.eval(smoothness))


