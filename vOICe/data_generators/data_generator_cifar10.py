"""
Script to generate the cifar-10 dataset.
"""

######################################################

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.utils import shuffle
import cv2

######################################################

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', default='./cifar_10/cifar-10-batches-py/', type=str, help='file path of the dataset')
parser.add_argument("--save_path",default="./",type=str,help="location to save .npz file")
args = parser.parse_args()


def unpickle(file):

    """
        Function to reverse pickle batches, in order to obtain the cifar-10 dataset.
    """

    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def data_gen(path,limit=-1):
    
    """
        Function to generate the cifar-10 dataset through unpickle.
    """
    
    file_names=["data_batch_1","data_batch_2","data_batch_3","data_batch_4","data_batch_5","test_batch"]

    X=np.zeros((1,1024))
    y=np.zeros((1,))

    X_gray=np.zeros((10000,1024))

    for i in range(0,len(file_names)):
        
        data=unpickle(os.path.join(path,file_names[i]))
        X_=data[b"data"]
        y_=data[b"labels"]

        for i in range(0,X_.shape[0]):
            temp=cv2.cvtColor(np.reshape(X_[i],(32,32,3)),cv2.COLOR_RGB2GRAY)
            X_gray[i]=np.reshape(temp,(1024,))

        X=np.vstack((X,X_gray))
        y=np.hstack((y,np.array(y_)))

    X,y=shuffle(X,y)

    data={"data":X[1:limit+1,:],"target":y[1:limit+1]}

    return data


if __name__=="__main__":
    
    limit=100
    data=data_gen(args.dataset_path,limit)

    print(data["data"].shape)
    print(data["target"].shape)



