"""
Script to obtain the mnist dataset
"""

###################################################

from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

###################################################


def data_gen():

    """
        Method to generate the mnist dataset, and pair images with labels.
    """

    (Xtrain,ytrain),(Xtest,ytest)=mnist.load_data()

    X=np.vstack((Xtrain,Xtest))
    y=np.hstack((ytrain,ytest))

    data={"data":X,"target":y}

    return data


if __name__=="__main__":

    data=data_gen()

    print(data["data"].shape)
    print(data["target"].shape)
