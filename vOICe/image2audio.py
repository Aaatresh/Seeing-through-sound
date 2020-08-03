"""
Script to convert images to audio
"""


#########################################################
import data_generator_mnist as mnist
import data_generator_cifar10 as cifar10
import os
import numpy as np
import argparse
import scipy.io.wavfile as wav
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as knn
#########################################################


parser = argparse.ArgumentParser()
parser.add_argument('--datafiles', default='./trial_images/', type=str, help='file path of the dataset')
parser.add_argument("--dataset",default="mnist",type=str,help="Name of the dataset")
args = parser.parse_args()


def getImageData(path,dataset,num_of_samples):

    """
        Method to obtain image data.
    """

    if(dataset=="cifar10"):
        data=cifar10.data_gen(path,num_of_samples)
    elif(dataset=="mnist"):
        data=mnist.data_gen(num_of_samples)

    return data


def convert2audio(img):

    """
        Method to perform image to audio conversion using the vOICe algorithm.
    """

    height_img=img.shape[1]
    width_img=img.shape[0]

    fstep=((10000-100)/height_img)+1

    n=np.arange(0,0.01,(1/44100.0))
    sound=np.zeros((n.size*width_img,2))
    tone=np.zeros((n.shape))

    for j in range(0,width_img):

        total_tone=np.zeros((n.size,2))

        k=img.shape[1]

        for i in range(0,img.shape[1]):

            tone=img[i,j]*np.sin(2*np.pi*fstep*(k)*n)

            total_tone[:,0]=total_tone[:,0]+(width_img-j)*tone
            total_tone[:,1]=total_tone[:,1]+(j)*tone

            k=k-1

        sound[j*n.size:(j+1)*n.size,0]=total_tone[:,0]
        sound[j*n.size:(j+1)*n.size,1]=total_tone[:,1]

    #making it into a single array by concatenating L and R audio
    audio=np.hstack((sound[:,0],sound[:,1]))

    return audio


def image2audio(path,num_of_samples,dataset="other"):
   
   """
        Method to convert images to audio files using the vOICe algorithm, given an image path.
   """
   
    if(dataset=="cifar10" or dataset=="mnist"): 
        data_image=getImageData(path,dataset,num_of_samples)
        X_image=data_image["data"]
        y=data_image["target"]

        if(dataset=="cifar10"):
            X_image=np.reshape(X_image,(num_of_samples,32,32))

    X_audio=[]

    i=0
    while(i<X_image.shape[0]):

        X_audio.append(convert2audio(X_image[i]))
        
        i=i+1
        print("Images done: ",end=" ")
        print(i)

    X_audio=np.array(X_audio)

    data_audio={"data":X_audio,"target":y}

    print("Audio conversion for "+dataset+" complete")

    return data_audio
    

if __name__=="__main__":
    
    ## Constants 
    num_of_samples=300

    ## Converting image data to audio
    audio=image2audio("./cifar_10/cifar-10-batches-py/",num_of_samples,"cifar10")
    
    #_________________________________________________________________________________________________________________
    
    
    ## Now, this audio data can be used to classify images as well!! We obtained no change in classification performance of knn when we - 
    ## on image data and on audio(converted from image) data. 
    print("------------------------------------------------------")
    print("Dataset shape:")
    print(audio["data"].shape)
    print(audio["target"].shape)
    print("------------------------------------------------------")


    X=audio["data"]
    y=audio["target"]

    print("target variable: ",y)
    print("------------------------------------------------------")

    
    ## Zero centering and scaling
    scaler=StandardScaler().fit(X)
    Xs=scaler.fit_transform(X)
    

    ## Splitting data into training and test set
    Xtrain,Xtest,ytrain,ytest=train_test_split(Xs,y,train_size=0.7)

    ## Initializing the KNN model
    model=knn(n_neighbors=5)

    ## Training the model
    model.fit(Xtrain,ytrain)

    ## Calculating the score
    print("Accuracy: ",model.score(Xtest,ytest))
    
