'''

TODO:
    - all these at multiple scales of output of the network which are up-sampled 
        - Temporal min reprojection loss.
            - mask out pixels which are different between samples
        - calculate per-pixel smoothness loss 

    
''' 

import keras
import keras.backend as K
import numpy as np
import cv2

## DEBUG libraries
import time
import tensorflow as tf

'''
Out = zeros(M,N,1)
For x in range(M)
	For y in range (N)
		Out1  = average(abs(left(x,y) - right(x+floor(disp(x,y)),y))* abs(ceil(disp(x,y)) - disp(x,y))
		Out2  = average(abs(left(x,y) - right(x+ceil(disp(x,y)),y)) * abs(floor(disp(x,y)) - disp(x,y))
		out(x,y) = (out1 + out2)
Return out 

'''
# eps = 1e-6
# def calculateLoss(left, right, disp):
#     h,w,ch = left.shape

#     # Adjust disparity to width in pixels
#     disp = disp * w

#     out = np.zeros((h,w), dtype=np.float)

#     for M in range(h):
#         for N in range(w):
#             if (N+int(np.ceil(disp[M,N])) >= w):
#                 continue
#             out1 = np.mean(np.abs(left[M,N,:] - right[M, N+int(np.floor(disp[M,N])),:]) )#* np.abs(np.ceil (disp[M,N]+eps) - disp[M,N]))
#             # out2 = np.mean(np.abs(left[M,N,:] - right[M, N+int(np.ceil (disp[M,N])),:]) * np.abs(np.floor(disp[M,N]) - disp[M,N]))
#             out[M,N] = out1 #+ out2
    
#     return out



def calculateLoss(left, right, disp):
    # obtain dimensions of the disparity map (M,N)
    h,w = disp.shape

    # Flatten and seperate out channels
    disp_f =   K.flatten(disp)
    left_f_0 =   K.flatten( left[:,:,0])
    right_f_0 =  K.flatten(right[:,:,0])
    left_f_1 =   K.flatten( left[:,:,1])
    right_f_1 =  K.flatten(right[:,:,1])
    left_f_2 =   K.flatten( left[:,:,2])
    right_f_2 =  K.flatten(right[:,:,2])

    # find the self-referential indices in the tensor
    indices = K.arange(0, h*w, dtype='float64')

    # offset the indices by the disparities to make the reprojection references for the left image
    right_references = K.update_add(indices, disp_f * -1. * w)
    right_references = K.clip(right_references, 0, disp_f.shape[0])

    # gather the values to creat the left re-projected images
    right_f_referance_to_projected_0 = K.gather(right_f_0, K.cast(right_references, 'int64'))
    right_f_referance_to_projected_1 = K.gather(right_f_1, K.cast(right_references, 'int64'))
    right_f_referance_to_projected_2 = K.gather(right_f_2, K.cast(right_references, 'int64'))

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
    
    # # (w,w) size tensor, arranged with each row [0, w-1]
    # x_range = K.arange(0, stop=w, step=1)
    # xt = K.repeat_elements(x_range, rep=w, axis=1)
    # # (h,h) size tensor, arranged with each row [0, h-1]
    # yt = K.repeat(K.arange(0, stop=h, step=1), n=h)

    # xt_flat = K.reshape(xt, (1,-1))
    # yt_flat = K.reshape(yt, (1,-1))


    # with tf.Session() as sess:
    #     # print(sess.run(x_range))
    #     # print(sess.run(y_t))

    # with tf.variable_scope('transform'):
    #     # grid of (x_t, y_t, 1), eq (1) in ref [1]
    #     x_t, y_t = tf.meshgrid(tf.linspace(0.0,   _width_f - 1.0,  _width),
    #                             tf.linspace(0.0 , _height_f - 1.0 , _height))

    #     x_t_flat = tf.reshape(x_t, (1, -1))
    #     y_t_flat = tf.reshape(y_t, (1, -1))

    #     x_t_flat = tf.tile(x_t_flat, tf.stack([_num_batch, 1]))
    #     y_t_flat = tf.tile(y_t_flat, tf.stack([_num_batch, 1]))

    #     x_t_flat = tf.reshape(x_t_flat, [-1])
    #     y_t_flat = tf.reshape(y_t_flat, [-1])

    #     x_t_flat = x_t_flat + tf.reshape(x_offset, [-1]) * _width_f

    #     input_transformed = _interpolate(input_images, x_t_flat, y_t_flat)

    #     output = tf.reshape(
    #         input_transformed, tf.stack([_num_batch, _height, _width, _num_channels]))
    #     return output



def photometricLRLoss(y_predicted, y_actual):
    '''
    parameters:
        y_predicted -  array of 4 scales of estimation with (Network dispartiy map of shape (height, width, 1))

        y_actual    - Four reference images (no datagen modifications) of shape (height, width, 12)


        left (t), 3 right images (t-1, t, t+1)
        Ypred: 4 disparity scales (1/8, 1/4, 1/2, 1) these get upscaled to 1 before doing loss 

        Disparities are given in 0-1 and  (pixels  =  disparity * width) 
        Disparity (pixel) less than 1 get clipped to zero 

    '''

    left, right = np.zeros((700,400))



    # moving the loss function from GPU to CPU

    # inputs needed for the loss function
    # general algorithm view

    '''
    tensor (x,y,1) (dispartiy map 0-1) PREDICTION for t input

    list of tensors all of shape (x,y,3) (3 tensors) ACTUAL (frames t-1, t, t+1)

    make sure to do everything in the keras backend

    can only use those operations on tensors in the loss functions

    MonodepthV1 good resource for losses in TF, but there are some differences, like masking with changing pixels, and the min temporal operation for photometric loss

    https://keras.io/backend/
    '''
    pass

def findGradients(y_predicted, leftImgPyramid):
    '''
    parameters:
        y_predicted -  array of 4 scales of estimation with (Network disparity map of shape (height, width, 1))

        leftImgPyramid - up sampled pyramid of the image for the different scales
    '''
    # adapted from https://github.com/mtngld/monodepth-1/blob/1f1fc80ac0dc727f3de561ead89e6792aea5e178/monodepth_model.py#L109 
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


class Timer(object):
    """
    Class to evaluate the execution time of code

    with Timer(verbose=True) as t:
        [your code you want to check]
    print("this took {} mins".format(t.mins))
    """
    def __init__(self, verbose=False):
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        self.mins = self.secs // 60
        self.msecs = self.secs * 1000  # millisecs
        if self.verbose:
            print('elapsed time: {:.2f} s ({:.4f} ms)'.format(self.secs, self.msecs))

#
# data.py ends here


if __name__ == "__main__":
    leftPath = './Photometric Loss Sample/2018-10-17-14-35-33_2018-10-17-14-36-11_359_left.jpg'
    rightPath = './Photometric Loss Sample/2018-10-17-14-35-33_2018-10-17-14-36-11-359_right.jpg'
    dispPath = './Photometric Loss Sample/2018-10-17-14-35-33_2018-10-17-14-36-11-359.png'

    left = cv2.imread(leftPath, cv2.IMREAD_COLOR)
    right = cv2.imread(rightPath, cv2.IMREAD_COLOR)
    disp = cv2.imread(dispPath, cv2.IMREAD_UNCHANGED)

    disp = disp / np.iinfo(disp.dtype).max #normalize disparity

    K_left = K.variable(left)
    K_right = K.variable(right)
    K_disp = K.variable(disp)

    K.print_tensor(K_disp)


    # input1 = K.constant(1)
    # input2 = K.constant(2)
    # input3 = K.constant(3)

    # node1 = tf.add(input1, input2)
    # print_output = K.print_tensor(node1)
    # # output = tf.multiply(print_output, input3)


    # sess = tf.Session()
    # sess.run(print_output)
    # sess.close()
    

    with Timer(verbose=False) as t:
        out = calculateLoss(left, right, disp)
    print("Evaluation took {} msecs".format(t.msecs))
    

    # print(np.mean(out))
