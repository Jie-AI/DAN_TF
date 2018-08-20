#coding=utf-8

##测试部分的代码
import utils
import tensorflow as tf
# import ImageServer
from ImageServer import ImageServer
from models import DAN
import numpy as np
from scipy import misc
from scipy import ndimage
from pylab import *
import os
import cv2

datasetDir = "../data/"
# testSet = ImageServer.Load(datasetDir + "commonSet.npz")
testSet = ImageServer.Load(datasetDir + "w300Set.npz")

def evaluateError(landmarkGt, landmarkP):
    e = np.zeros(68)
   #  ocular_dist = np.mean(np.linalg.norm(landmarkGt[36:42] - landmarkGt[42:48], axis=1)) # 瞳孔之间的距离作为norm
    ocular_dist = np.mean(np.linalg.norm(landmarkGt[36] - landmarkGt[45])) # 眼角之间的距离作为norm
    for i in range(68):
        e[i] = np.linalg.norm(landmarkGt[i] - landmarkP[i])
    e = e / ocular_dist
    return e

def evaluateBatchError(landmarkGt, landmarkP, batch_size):
    e = np.zeros([batch_size, 68])
    for i in range(batch_size):
        e[i] = evaluateError(landmarkGt[i], landmarkP[i])
    # mean_err = e[:,:].mean(axis=1)#axis=0)
    mean_err = e[:,:].mean(axis=1)
    return mean_err

def getLabelsForDataset(imageServer):
    nSamples = imageServer.gtLandmarks.shape[0]
    nLandmarks = imageServer.gtLandmarks.shape[1]

    y = np.zeros((nSamples, nLandmarks, 2), dtype=np.float32)
    y = imageServer.gtLandmarks

    return y.reshape((nSamples, nLandmarks * 2))

nSamples = testSet.gtLandmarks.shape[0]
print('The number of samples is ', nSamples)
imageHeight = testSet.imgSize[0]
imageWidth = testSet.imgSize[1]
nChannels = testSet.imgs.shape[1]

A = testSet.A  # (2,2)
t = testSet.t  # (2)
# print(A[1].shape)
# print(t[1].shape)

Xtest = testSet.imgs
# print(Xtest)
Ytest = getLabelsForDataset(testSet)

meanImg = testSet.meanImg
stdDevImg = testSet.stdDevImg
initLandmarks = testSet.initLandmarks[0].reshape((-1))

dan = DAN(initLandmarks)

with tf.Session() as sess:
    Saver = tf.train.Saver()
    Writer = tf.summary.FileWriter("logs/", sess.graph)

    Saver.restore(sess,'./Model/Model')
    print('Pre-trained model has been loaded!')
       
    # Landmark68Test(MeanShape,ImageMean,ImageStd,sess)
    errs = []
    # Batch_size = 64

    for iter in range(1):
        num_images = len(Ytest)
        # TestErr, S2_Ret = sess.run([dan['S2_Cost'],dan['S2_Ret']],{dan['InputImage']:Xtest,dan['GroundTruth']:Ytest,\
        #     dan['S1_isTrain']:False,dan['S2_isTrain']:False})
        # S2_Ret = np.reshape(S2_Ret, [-1, 68, 2])
        TestErr, Landmark = sess.run([dan['S1_Cost'],dan['S1_Ret']],{dan['InputImage']:Xtest,dan['GroundTruth']:Ytest,\
             dan['S1_isTrain']:False,dan['S2_isTrain']:False})
        Landmark = np.reshape(Landmark, [-1, 68, 2])
        # errs.append(TestErr)
        # print('The mean error for image %d is: %f'.format(iter, TestErr))
        # print('The mean error for image ',iter,' is: ', TestErr)
        '''
        
        error_per_image = evaluateBatchError(Ytest.reshape([-1, 68, 2]), Landmark.reshape([-1, 68, 2]), num_images)
        
        sorted_error = np.sort(error_per_image, axis=0)
        
        
        plt.figure()
        plt.grid()
        plt.xlabel('Point-to-point Normalized RMS Error')
        plt.ylabel('Image Proportion')
        plt.yticks(np.arange(0, 1, 0.1))
        plt.xlim([0, 0.08])
        plt.ylim([0, 1])
        plt.plot(np.concatenate([[0], sorted_error]), np.arange(num_images+1, dtype=np.float)/num_images, color='darkorange', lw=2, label='cumulative curve')
        plt.savefig('images/plot1.png', format='png')
        plt.figure()
        plt.grid()
        plt.xlim([0, 0.35])
        plt.ylim([0, 1])
        plt.plot(np.concatenate([[0], sorted_error]), np.arange(num_images+1, dtype=np.float)/num_images, color='darkorange', lw=2, label='cumulative curve')
        plt.savefig('images/plot2.png', format='png')
        '''
        # print(evaluateBatchError(Ytest.reshape([-1, 68, 2]), Landmark.reshape([-1, 68, 2]), Batch_size))
        for i in range(num_images):
            
            A_temp = np.linalg.inv(A[i])
            t_temp = np.dot(np.reshape(-t[i],(1,2)), A_temp)
            landmark = Landmark[i]
            landmark = np.reshape(np.dot(landmark, A_temp) + t_temp, [68, 2])
            ptsFilename = testSet.filenames[i][:-3] + "pts"
            groundtruth = utils.loadFromPts(ptsFilename)
            error = evaluateError(landmark,groundtruth)
            # print(error)
            errs.append(np.mean(error))
            '''
            img = cv2.imread(testSet.filenames[i])
            
            for point in range(68):
                cv2.circle(img,( int(landmark[point][0]), int(landmark[point][1])),1,(0,255,255),3);
            cv2.imwrite(os.path.join("../data/vis",testSet.filenames[i][-14:]),img)

            '''
        
        sorted_error = np.sort(errs, axis=0)

        plt.figure()
        plt.grid()
        plt.xlabel('Point-to-point Normalized RMS Error')
        plt.ylabel('Image Proportion')
        plt.yticks(np.arange(0, 1, 0.1))
        plt.xlim([0, 0.08])
        plt.ylim([0, 1])
        plt.plot(np.concatenate([[0], sorted_error]), np.arange(num_images+1, dtype=np.float)/num_images, color='darkorange', lw=2, label='cumulative curve')
        plt.savefig('images/plot1.png', format='png')
        plt.figure()
        plt.grid()
        plt.xlim([0, 0.35])
        plt.ylim([0, 1])
        plt.plot(np.concatenate([[0], sorted_error]), np.arange(num_images+1, dtype=np.float)/num_images, color='darkorange', lw=2, label='cumulative curve')
        plt.savefig('images/plot2.png', format='png')
        print('The overall mean error is: %f' % np.mean(errs))
    # errs = np.array(errs)

    

