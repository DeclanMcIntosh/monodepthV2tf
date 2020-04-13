import keras
import os
import random
import numpy as np
import cv2

'''
Declan McIntosh Robert Lee Data Generator for MonodepthV2 Keras/Tf implementation
'''


def preprocess_input(image, randomVals):
    '''
    preforms data augmentations listed below each with chance == 50%
    - Horiontal flip
    - Random brightness +- 0.2
    - Random contrast +- 0.2
    - Random saturation +- 0.2
    - Hue Jitter +- 0.1

    all random values provided in range (0,1)
    '''
    if randomVals[0] > 0.5:
        # flip image horizontally
        #image = np.flip(image, 1)
        None
    if randomVals[1] > 0.5:
        # increase/ decrease contrast
        image = np.uint8(np.clip(image * (0.8 + randomVals[2]/2.5), a_min=0, a_max=255))
    # Convert image to HSV for some transformations
    hsv_image = np.int32(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    if randomVals[3] > 0.5:
        # change brightness of image
        hsv_image[:,:,2] += int(((randomVals[4]/2.5 )- 0.2 ) * 255.) 
        hsv_image[:,:,2] = np.clip(hsv_image[:,:,2], a_min =0, a_max = 255) 
    if randomVals[5] > 0.5:
        # change staturation
        hsv_image[:,:,1] += int(((randomVals[6]/2.5 )- 0.2 ) * 255.)
        hsv_image[:,:,1] = np.clip(hsv_image[:,:,1], a_min = 0, a_max = 255) 
    if randomVals[7] > 0.5:
        # change Hue
        hsv_image[:,:,0] += int(((randomVals[8]/2.5 )- 0.2 ) * 179.)
        hsv_image[:,:,0] = np.clip(hsv_image[:,:,0], a_min = 0, a_max = 179) 
    # Convert image back from HSV
    image = cv2.cvtColor(np.uint8(hsv_image), cv2.COLOR_HSV2BGR)
    return image


    #for x in range(0,9):
    #    randomVals.append(random.random())
    #input_img = preprocess_input(image=input_img, randomVals=randomVals)

class depthDataGenerator(keras.utils.Sequence):
    '''Generates data for Keras'''
    '''Framework taken from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly'''
    '''Provided directories should contain the same number of files all with the same names to their pair image'''
    def __init__(self, left_dir, right_dir, batch_size = 64, image_size=(640,192), shuffle=True, max_img_time_diff=700, agumentations=True):
        self.left_dir = left_dir
        self.right_dir = right_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_img_time_diff = max_img_time_diff
        self.agumentations = agumentations
        self.inputs = []
        self.on_epoch_end()

    def __len__(self):
        '''Denotes the number of batches per epoch'''
        return int(np.floor(len(self.inputs) / self.batch_size))


    def __getitem__(self, index):
        '''Generate one batch of data'''
        outX =  np.empty((self.batch_size,  *self.image_size, 3))
        outY_0 = np.empty((self.batch_size, *self.image_size, 3))
        outY_1 = np.empty((self.batch_size, *self.image_size, 3))
        outY_2 = np.empty((self.batch_size, *self.image_size, 3))
        outY_3 = np.empty((self.batch_size, *self.image_size, 3))

        imageNames = self.inputs[index*self.batch_size:(index+1)*self.batch_size]

        for _, imageNameSet in enumerate(imageNames):
            left        = cv2.resize(cv2.imread(self.left_dir  + imageNameSet[0]), dsize=self.image_size)
            right_minus = cv2.resize(cv2.imread(self.right_dir + imageNameSet[1]), dsize=self.image_size)
            right       = cv2.resize(cv2.imread(self.right_dir + imageNameSet[2]), dsize=self.image_size)
            right_plus  = cv2.resize(cv2.imread(self.right_dir + imageNameSet[0]), dsize=self.image_size)
            
            #print(left.shape)
            #cv2.imshow('test', left)
            #cv2.waitKey(-1)

            
            if self.agumentations:
                randomVals = []
                for x in range(0,9):
                    randomVals.append(random.random())
                left_augmented = preprocess_input(image=left, randomVals=randomVals)
            else:
                left_augmented = left

            outX[_]   =  np.transpose(left_augmented,   axes=[1,0,2])
            outY_0[_] =  np.transpose(left,             axes=[1,0,2])
            outY_1[_] =  np.transpose(right_minus,      axes=[1,0,2])
            outY_2[_] =  np.transpose(right,            axes=[1,0,2])
            outY_3[_] =  np.transpose(right_plus,       axes=[1,0,2])

        return outX, [outY_0, outY_1, outY_2, outY_3]
                        

    def on_epoch_end(self):
        '''Updates indexes after each epoch'''
        '''We want only images that have corresponding right iamges and nearby right images'''
        print("")
        leftImgs = os.listdir(self.left_dir)
        leftImgs.sort()
        
        prefixes = ('.')
        for word in leftImgs[:]:
            if word.startswith(prefixes):
                leftImgs.remove(word)

        debugCount = 0
        debugBadCount = 0
        for leftImageName in leftImgs:
            index = leftImgs.index(leftImageName)
            if index != 0 and index != len(leftImgs) - 1:
                # grab nearby images
                t_minus_1_name = leftImgs[index-1]
                t_plus_1_name  = leftImgs[index+1]

                # check the iamges are close together in time
                
                left_img_1_time = (int( leftImageName[-7:-4]) + int( leftImageName[-10:-8]) * 1000 +  int( leftImageName[-13:-11]) * 1000 * 60)# % (60 *60 * 1000)
                t_minus_1_time  = (int(t_minus_1_name[-7:-4]) + int(t_minus_1_name[-10:-8]) * 1000 +  int(t_minus_1_name[-13:-11]) * 1000 * 60)# % (60 *60 * 1000)
                t_plus_1_time   = (int( t_plus_1_name[-7:-4]) + int( t_plus_1_name[-10:-8]) * 1000 +  int( t_plus_1_name[-13:-11]) * 1000 * 60)# % (60 *60 * 1000)
                
                # ensure images are within some time frame of center image

                if abs(left_img_1_time - t_minus_1_time) < self.max_img_time_diff and abs(left_img_1_time - t_plus_1_time) < self.max_img_time_diff:
                    self.inputs.append([leftImageName,t_minus_1_name,t_plus_1_name])
                    debugCount += 1
                    print("Found ", debugCount, " input image sets to use in ", self.left_dir, "  " , debugBadCount, "number of un-useable images", end='\r')
                else:
                    debugBadCount += 1
        if self.shuffle:
            random.shuffle(self.inputs)         
        print("")



if __name__ == "__main__":
    test = depthDataGenerator('../validate/left/', '../validate/right/')

    print('Data generator test success.')