#coding=utf-8

##训练部分的代码

import tensorflow as tf
import numpy as np

from ImageServer import ImageServer
from models import DAN
from scipy import misc
from scipy import ndimage
from pylab import *
import os

datasetDir = "../data/"
trainSet = ImageServer.Load(datasetDir + "dataset_nimgs=62960_perturbations=[0.2, 0.2, 20, 0.25]_size=[112, 112].npz")
# validationSet = ImageServer.Load(datasetDir + "dataset_nimgs=9_perturbations=[]_size=[112, 112].npz")
validationSet = ImageServer.Load(datasetDir + "dataset_nimgs=100_perturbations=[]_size=[112, 112].npz")

testSet = ImageServer.Load(datasetDir + "commonSet.npz")


# print(trainSet.meanShape)
# imgs = validationSet.imgs # image size is (112,112,1)
# for i in range(len(imgs)):
#     # print(np.reshape(imgs[i],(112,112)).shape)
#     misc.imsave(os.path.join("../data/vis",validationSet.filenames[i][-14:]), np.reshape(imgs[i],(112,112)))

# Create a new DAN regressor.
# 两个stage训练，不宜使用estimator
# regressor = tf.estimator.Estimator(model_fn=DAN,
#                                 params={})
def evaluateError(landmarkGt, landmarkP):
    e = np.zeros(68)
    ocular_dist = np.mean(np.linalg.norm(landmarkGt[36:42] - landmarkGt[42:48], axis=1))
    for i in range(68):
        e[i] = np.linalg.norm(landmarkGt[i] - landmarkP[i])
    e = e / ocular_dist
    return e

def evaluateBatchError(landmarkGt, landmarkP, batch_size):
    e = np.zeros([batch_size, 68])
    for i in range(batch_size):
        e[i] = evaluateError(landmarkGt[i], landmarkP[i])
    mean_err = e[:,:].mean()#axis=0)
    return mean_err


def getLabelsForDataset(imageServer):
    nSamples = imageServer.gtLandmarks.shape[0]
    nLandmarks = imageServer.gtLandmarks.shape[1]

    y = np.zeros((nSamples, nLandmarks, 2), dtype=np.float32)
    y = imageServer.gtLandmarks

    return y.reshape((nSamples, nLandmarks * 2))

nSamples = trainSet.gtLandmarks.shape[0]
imageHeight = trainSet.imgSize[0]
imageWidth = trainSet.imgSize[1]
nChannels = trainSet.imgs.shape[3]


Xtest = testSet.imgs
Ytest = getLabelsForDataset(testSet)
Xtrain = trainSet.imgs
Xvalid = validationSet.imgs
# print(len(Xtrain))
# print(len(Xvalid))
# import pdb; pdb.set_trace()

Ytrain = getLabelsForDataset(trainSet)
Yvalid = getLabelsForDataset(validationSet)
testIdxsTrainSet = range(len(Xvalid))
testIdxsValidSet = range(len(Xvalid))
meanImg = trainSet.meanImg
stdDevImg = trainSet.stdDevImg
initLandmarks = trainSet.initLandmarks[0].reshape((1, 136))

dan = DAN(initLandmarks)

STAGE = 1

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

with tf.Session() as sess:
    Saver = tf.train.Saver()
    # merged = tf.summary.merge_all()
    Writer = tf.summary.FileWriter("logs/", sess.graph)
    if STAGE < 2:
        # Saver.restore(sess,'./Model/Model')
        sess.run(tf.global_variables_initializer())
    else:
        Saver.restore(sess,'./Model/Model')
        print('Pre-trained model has been loaded!')
       
    # Landmark68Test(MeanShape,ImageMean,ImageStd,sess)
    print("Starting training......")
    global_step = 0
    errors=[]
    errorsTrain=[]
    errorsTest=[]
    for epoch in range(1000):
        epoch_train_err = 0
        epoch_val_err=0
        epoch_test_err=0
        Count = 0
        Batch_size = 64
        for batch in iterate_minibatches(Xtrain, Ytrain, Batch_size, True):
            inputs, targets = batch
        # while Count * Batch_size < Xtrain.shape[0]:
            RandomIdx = np.random.choice(Xtrain.shape[0],Batch_size,False)
            if STAGE == 1 or STAGE == 0:
                BatchErr,_ = sess.run([dan['S1_Cost'],dan['S1_Optimizer']], feed_dict={dan['InputImage']:inputs,\
                    dan['GroundTruth']:targets,dan['S1_isTrain']:True,dan['S2_isTrain']:False,dan['global_step']:global_step})
                ValidErr= sess.run(dan['S1_Cost'], {dan['InputImage']:Xvalid,dan['GroundTruth']:Yvalid,\
                        dan['S1_isTrain']:False,dan['S2_isTrain']:False})
                TestErr= sess.run(dan['S1_Cost'], {dan['InputImage']:Xtest,dan['GroundTruth']:Ytest,\
                        dan['S1_isTrain']:False,dan['S2_isTrain']:False})
                file=open('../batch_losses_2.txt','a')
                file.write('%d epoch %d batch the batch_error is %f\n' % (epoch,Count,BatchErr))
                file.close()
                file=open('../val_losses_2.txt','a')
                file.write('%d epoch %d batch the val_error is %f\n' % (epoch,Count,ValidErr))
                file.close()
                file=open('../test_losses_2.txt','a')
                file.write('%d epoch %d batch the test_error is %f\n' % (epoch,Count,TestErr))
                file.close()
            else:
                sess.run(dan['S2_Optimizer'], feed_dict={dan['InputImage']:inputs,\
                    dan['GroundTruth']:targets,dan['S1_isTrain']:False,dan['S2_isTrain']:True})
            epoch_train_err += BatchErr
            epoch_val_err += ValidErr
            epoch_test_err += TestErr
            '''
            if Count % 40 == 0:
                if STAGE == 1 or STAGE == 0:
                    ValidErr, Landmark= sess.run([dan['S1_Cost'],dan['S1_Ret']], {dan['InputImage']:Xvalid,dan['GroundTruth']:Yvalid,\
                        dan['S1_isTrain']:False,dan['S2_isTrain']:False})
                    idxs=range(1000)
                    idxs = np.array_split(idxs, 10)
                    error=0
                    for i in range(len(idxs)):
                        TrainErr = sess.run(dan['S1_Cost'],{dan['InputImage']:Xtrain[idxs],\
                            dan['GroundTruth']:Ytrain[idxs],dan['S1_isTrain']:False,dan['S2_isTrain']:False})
                        error += TrainErr
                    error = error / len(idxs)
                else:
                    ValidErr ,Landmark = sess.run([dan['S2_Cost'],dan['S2_Ret']],{dan['InputImage']:Xvalid,dan['GroundTruth']:Yvalid,\
                        dan['S1_isTrain']:False,dan['S2_isTrain']:False})
                    TrainErr = sess.run(dan['S2_Cost'],{dan['InputImage']:Xtrain,\
                        dan['GroundTruth']:Ytrain,dan['S1_isTrain']:False,dan['S2_isTrain']:False})
                print('Epoch: ', epoch, ' Batch: ', Count, 'ValidErr:', ValidErr, ' TrainErr:', TrainErr)
                
            '''
            Count += 1
            global_step +=1 
        errors.append(epoch_val_err)
        errorsTrain.append(epoch_train_err)
        plot(errors)
        plot(errorsTrain)
        plot(epoch_test_err)
        # ylim(ymax=np.max( [errors[0], errorsTrain[0], errorsTest[0]]))
        ylim(ymax=0.5)
        savefig("../errors_perEpoch_2.jpg")
        clf()
        Saver.save(sess,'./checkpoints_2/Model_%d/Model' % epoch)
        print('Epoch: ', epoch, ' Train_error:', epoch_train_err /Count , ' Value_error:' , epoch_val_err/Count , 'Test_error:' , epoch_test_err/Count)