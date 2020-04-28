'''
This file contains two ealuation metrics implemented for 
using numpy and cv2, these will need to re-implemented using tensorflow for live training evaluation
'''
import numpy as np
import cv2
from math import sqrt


def generateARDmetric(estimated, ground):
    '''
    this metric generates the ARD metric for a single capture
    between the estimated and ground truth space depths
    '''
    # assert ground and estimated values shapes lineup
    assert estimated.shape ==  ground.shape

    # convert types of arrays to prevent rounding
    estimated.astype(np.float)
    ground.astype(np.float)

    # count the number of valid ground depth values as it is sparce
    N_R_k = np.count_nonzero(ground)

    output = 0 

    # get the differences
    for x in range(0,estimated.shape[0]):
        for y in range(0,estimated.shape[1]):
            if ground[x,y] != 0:
                output += abs((estimated[x,y]-ground[x,y]))/ground[x,y]

    # return value
    return output/N_R_k


def generateMRmetric(estimated, ground, theta):
    ''' 
    this generates the MR (matching rate) preformance metric
    based on the estimate, ground depth data, and the theta value for matching values which agree
    '''
    # assert ground and estimated values shapes lineup
    assert estimated.shape ==  ground.shape

    # count the number of valid ground depth values as it is sparce
    Normalizer = np.count_nonzero(ground)
    elements = np.nonzero(ground)

    agreementCount = 0.
    # define function for vectorization which finde th ADR value for each pixle depth pair
    for element in range(elements[0].shape[0]):
        x = elements[0][element]
        y = elements[1][element]
        val1 = estimated[x,y]
        val2 = ground[x,y]
        agreementCount += 1. if max([val1/val2, val2/val1]) < theta else 0.

    # determine percentage
    d_p = (agreementCount/Normalizer) * 100

    return d_p

def absoluteRelativeSqrd(estimated, ground):
    ''' 
    this generates the MR (matching rate) preformance metric
    based on the estimate, ground depth data, and the theta value for matching values which agree
    '''
    # assert ground and estimated values shapes lineup
    assert estimated.shape ==  ground.shape

    # count the number of valid ground depth values as it is sparce
    Normalizer = np.count_nonzero(ground)
    elements = np.nonzero(ground)

    agreement = 0
    agreementSqrd = 0
    # define function for vectorization which finde th ADR value for each pixle depth pair
    for element in range(elements[0].shape[0]):
        x = elements[0][element]
        y = elements[1][element]
        val1 = estimated[x,y]
        val2 = ground[x,y]
        agreement += abs(val1-val2)
        agreementSqrd += abs(val1*val1 - val2*val2) / val2

    return agreement/Normalizer, agreementSqrd/Normalizer

if __name__ == "__main__":
    depth = cv2.imread("../test/disp/2018-10-27-08-54-23_2018-10-27-08-54-27-073.png")
    depth = depth[:,:,1]
    depthNoisy = depth + (np.random.normal(size=depth.shape) * np.mean(depth) * 0.2)
    print(generateARDmetric(depth,depth))
    print(generateMRmetric(depth,depth,1.02))