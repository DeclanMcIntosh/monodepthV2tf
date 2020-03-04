import keras
from keras.layers import Conv2D, BatchNormalization, MaxPool2D, Flatten, AveragePooling2D, concatenate, Add, Activation
'''

Create a resnet encoder either type 50 layers or type 18 layers...

output 1000 feature values/classes
pad zeros of skill connection from input when channels increase...




main model does a resnet encoding then a depth decoding 



'''

def generateResNetEncoderLayers(inputLayer, resnetType=50):
    '''
    takes an input layer of type Input from keras   

    returns the output layer of a resent of the specifified type
    input layer must be of form: (batches, even#, even#, 3*numImages)
    '''
    assert resnetType in [50,18]
    if resnetType == 50:
        x = resentLayerInitialStage(inputLayer)

        x = resentLayerLaterState_50(x, 64, [64,64,256])
        x = resentLayerLaterState_50(x, 256, [64,64,256])
        x = resentLayerLaterState_50(x, 256, [64,64,256])
        
        x = resentLayerLaterState_50(x, 256, [128,128,512])
        x = resentLayerLaterState_50(x, 512, [128,128,512])
        x = resentLayerLaterState_50(x, 512, [128,128,512])
        x = resentLayerLaterState_50(x, 512, [128,128,512])

        x = resentLayerLaterState_50(x, 512, [256,256,1024])
        x = resentLayerLaterState_50(x, 1024, [256,256,1024])
        x = resentLayerLaterState_50(x, 1024, [256,256,1024])
        x = resentLayerLaterState_50(x, 1024, [256,256,1024])
        x = resentLayerLaterState_50(x, 1024, [256,256,1024])
        x = resentLayerLaterState_50(x, 1024, [256,256,1024])

        x = resentLayerLaterState_50(x, 1024, [512,512,2048])
        x = resentLayerLaterState_50(x, 2048, [512,512,2048])
        x = resentLayerLaterState_50(x, 2048, [512,512,2048])

        x = resnetOuputStage(x)
        return x 
    else: # is resnet 18
        x = resentLayerInitialStage(inputLayer)
        x = resentLayerLaterState_18(x, 64, [64,64])
        x = resentLayerLaterState_18(x, 64, [64,64])

        x = resentLayerLaterState_18(x, 64, [128,128])
        x = resentLayerLaterState_18(x, 128, [128,128])

        x = resentLayerLaterState_18(x, 128, [256,256])
        x = resentLayerLaterState_18(x, 256, [256,256])
        
        x = resentLayerLaterState_18(x, 256, [512,512])
        x = resentLayerLaterState_18(x, 512, [512,512])

        x = resnetOuputStage(x)
        return x

def resentLayerInitialStage(inputLayer):
    x = Conv2D(filters=64,kernel_size=7,strides=2,data_format='channels_last',activation='relu')(inputLayer)
    x = BatchNormalization(axis=3)(x)
    x = MaxPool2D(pool_size=(3,3),strides=2, data_format='channels_last')(x)
    return x

def resentLayerLaterState_50(inputLayer, inputChannels, channels):
    '''
    3 convolutional blocks

    1x1, channels[0], relu
    3x3, channels[0], relu
    1x1, channels[0], linear
    add input and output
    relu
    '''

    assert len(channels) == 3

    x = Conv2D(channels[0], kernel_size=1,strides=1,data_format='channels_last',activation='relu')(inputLayer)
    x = Conv2D(channels[1], kernel_size=3,strides=1,data_format='channels_last',activation='relu')(inputLayer)
    x = Conv2D(channels[2], kernel_size=1,strides=1,data_format='channels_last',activation='linear')(inputLayer)

    if inputChannels != channels[2]:
        # this should be zero padding but just simple one conv for now untill fixed
        inputLayer = Conv2D(channels[2], kernel_size=1, strides=1,data_format='channels_last',activation='linear')(inputLayer)

    x = Add()([x,inputLayer])
    x = Activation('relu')(x)
    return x

def resentLayerLaterState_18(inputLayer, inputChannels, channels):
    '''
    two convolutional blocks 
    3x3, channels[0], relu
    3x3, channels[0], linear
    add input and output
    relu
    '''

    assert len(channels) == 2

    x = Conv2D(channels[0], kernel_size=3,strides=1,data_format='channels_last',activation='relu')(inputLayer)
    x = Conv2D(channels[1], kernel_size=3,strides=1,data_format='channels_last',activation='linear')(inputLayer)

    if inputChannels != channels[1]:
        # this should be zero padding but just simple one conv for now untill fixed
        inputLayer = Conv2D(channels[1], kernel_size=1, strides=1,data_format='channels_last',activation='linear')(inputLayer)

    x = Add()([x,inputLayer])
    x = Activation('relu')(x)

    return x

def resnetOuputStage(inputLayer, pools=1000):
    # TODO not sure if should be encoded in 1k values as described or decoded from this point...
    return inputLayer