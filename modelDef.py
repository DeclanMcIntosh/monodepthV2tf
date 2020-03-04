import keras
from keras.layers import Conv2D, BatchNormalization, MaxPool2D, Flatten, AveragePooling2D, concatenate, Add, Activation, Input

from keras import Model
from keras.optimizers import Adam
'''

Create a resnet encoder either type 50 layers or type 18 layers...

output 1000 feature values/classes
pad zeros of skill connection from input when channels increase...




main model does a resnet encoding then a depth decoding 


the res-net can be tested on and trained on imagenet to verify correctness and working

ref:
https://medium.com/@14prakash/understanding-and-implementing-architectures-of-resnet-and-resnext-for-state-of-the-art-image-cf51669e1624    
https://keras.io/applications/#resnet
https://towardsdatascience.com/understanding-and-visualizing-resnets-442284831be8

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

        x = resentLayerLaterState_50(x, 64, [64,64,256],1)
        x = resentLayerLaterState_50(x, 256, [64,64,256],1)
        x = resentLayerLaterState_50(x, 256, [64,64,256],1)
        
        x = resentLayerLaterState_50(x, 256, [128,128,512],2)
        x = resentLayerLaterState_50(x, 512, [128,128,512],1)
        x = resentLayerLaterState_50(x, 512, [128,128,512],1)
        x = resentLayerLaterState_50(x, 512, [128,128,512],1)

        x = resentLayerLaterState_50(x, 512, [256,256,1024],2)
        x = resentLayerLaterState_50(x, 1024, [256,256,1024],1)
        x = resentLayerLaterState_50(x, 1024, [256,256,1024],1)
        x = resentLayerLaterState_50(x, 1024, [256,256,1024],1)
        x = resentLayerLaterState_50(x, 1024, [256,256,1024],1)
        x = resentLayerLaterState_50(x, 1024, [256,256,1024],1)

        x = resentLayerLaterState_50(x, 1024, [512,512,2048],2)
        x = resentLayerLaterState_50(x, 2048, [512,512,2048],1)
        x = resentLayerLaterState_50(x, 2048, [512,512,2048],1)

        x = resnetOuputStage(x)
        return x 
    else: # is resnet 18
        x = resentLayerInitialStage(inputLayer)
        x = resentLayerLaterState_18(x, 64, [64,64],1)
        x = resentLayerLaterState_18(x, 64, [64,64],1)

        x = resentLayerLaterState_18(x, 64, [128,128],2)
        x = resentLayerLaterState_18(x, 128, [128,128],1)

        x = resentLayerLaterState_18(x, 128, [256,256],2)
        x = resentLayerLaterState_18(x, 256, [256,256],1)
        
        x = resentLayerLaterState_18(x, 256, [512,512],2)
        x = resentLayerLaterState_18(x, 512, [512,512],1)

        x = resnetOuputStage(x)
        return x

def resentLayerInitialStage(inputLayer):
    x = Conv2D(filters=64,kernel_size=7,strides=2,data_format='channels_last',activation='relu',padding='same')(inputLayer)
    x = BatchNormalization(axis=3)(x)
    x = MaxPool2D(pool_size=(3,3),strides=2, data_format='channels_last',padding='same')(x)
    return x

def resentLayerLaterState_50(inputLayer, inputChannels, channels, poolingStride):
    '''
    3 convolutional blocks

    1x1, channels[0], relu
    3x3, channels[0], relu
    1x1, channels[0], linear
    add input and output
    relu
    '''

    assert len(channels) == 3

    x = Conv2D(channels[0], kernel_size=1,strides=poolingStride,data_format='channels_last',activation='relu',padding='same')(inputLayer)
    x = BatchNormalization(axis=3)(x)
    x = Conv2D(channels[1], kernel_size=3,strides=1,data_format='channels_last',activation='relu',padding='same')(inputLayer)
    x = BatchNormalization(axis=3)(x)
    x = Conv2D(channels[2], kernel_size=1,strides=1,data_format='channels_last',activation='linear',padding='same')(inputLayer)
    x = BatchNormalization(axis=3)(x)

    if inputChannels != channels[2]:
        # this could be zero padding but instread were doing 1x1 stride 1 convolution to make the shapes the same, both are technically from paper acceptable
        inputLayer = Conv2D(channels[2], kernel_size=1, strides=1,data_format='channels_last',activation='linear',padding='same')(inputLayer)
        if poolingStride != 1:
            inputLayer = MaxPool2D(pool_size=2)(inputLayer)

    x = Add()([x,inputLayer])
    x = Activation('relu')(x)
    return x

def resentLayerLaterState_18(inputLayer, inputChannels, channels, poolingStride):
    '''
    two convolutional blocks 
    3x3, channels[0], relu
    3x3, channels[0], linear
    add input and output
    relu
    '''

    assert len(channels) == 2

    x = Conv2D(channels[0], kernel_size=3,strides=poolingStride,data_format='channels_last',activation='relu',padding='same')(inputLayer)
    x = BatchNormalization(axis=3)(x)
    x = Conv2D(channels[1], kernel_size=3,strides=1,data_format='channels_last',activation='linear',padding='same')(x)
    x = BatchNormalization(axis=3)(x)

    if inputChannels != channels[1]:
        # this should be zero padding but just simple one conv for now untill fixed
        inputLayer = Conv2D(channels[1], kernel_size=1, strides=1,data_format='channels_last',activation='linear',padding='same')(inputLayer)
        if poolingStride != 1:
            inputLayer = MaxPool2D(pool_size=2,padding='same')(inputLayer)

    x = Add()([x,inputLayer])
    x = Activation('relu')(x)

    return x

def resnetOuputStage(inputLayer, pools=1000):
    # TODO not sure if should be encoded in 1k values as described or decoded from this point... not sure how they are doing that, 
    # likely just a flatten operation but need a second opinion
    output = AveragePooling2D(pool_size=2,strides=1)(inputLayer)
    return output


if __name__ == "__main__":
    
    
    InputLayer = Input(shape=(256,256,3))

    networkOuput = generateResNetEncoderLayers(InputLayer, resnetType=18)


    model = Model(inputs=[InputLayer], output=[networkOuput])
    model.compile(optimizer=Adam(), loss='mse')
    model.summary()