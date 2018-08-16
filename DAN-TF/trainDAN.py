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

STAGE = 2

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
        Saver.restore(sess,'./Model/Model')
        # sess.run(tf.global_variables_initializer())
    else:
        Saver.restore(sess,'./Model/Model')
        print('Pre-trained model has been loaded!')
       
    # Landmark68Test(MeanShape,ImageMean,ImageStd,sess)
    print("Starting training......")
    global_step = 0
    for epoch in range(1000):
        train_err = 0
        Count = 0
        Batch_size = 64
        for batch in iterate_minibatches(Xtrain, Ytrain, Batch_size, True):
            inputs, targets = batch
        # while Count * Batch_size < Xtrain.shape[0]:
            RandomIdx = np.random.choice(Xtrain.shape[0],Batch_size,False)
            if STAGE == 1 or STAGE == 0:
                sess.run(dan['S1_Optimizer'], feed_dict={dan['InputImage']:inputs,\
                    dan['GroundTruth']:targets,dan['S1_isTrain']:True,dan['S2_isTrain']:False,dan['global_step']:global_step})
                # writer.add_summary(merge,i)
            else:
                sess.run(dan['S2_Optimizer'], feed_dict={dan['InputImage']:inputs,\
                    dan['GroundTruth']:targets,dan['S1_isTrain']:False,dan['S2_isTrain']:True})
                
                # writer.add_summary(merge,i)
            if Count % 40 == 0:
                ValidErr = 0
                BatchErr = 0

                if STAGE == 1 or STAGE == 0:
                    ValidErr, Landmark= sess.run([dan['S1_Cost'],dan['S1_Ret']], {dan['InputImage']:Xvalid,dan['GroundTruth']:Yvalid,\
                        dan['S1_isTrain']:False,dan['S2_isTrain']:False})
                    
                    '''
                    if Count%200==0 and Count!=0:
                        Landmark= np.reshape(Landmark, [-1, 68, 2])
                        # for i in range(len(validationSet.filenames)):
                        for i in range(30):
                            # draw prediction
                            img = testSet.imgs[i]
                            img = np.reshape(img,(112,112))
                            imshow(img, cmap ='gray')
                            x=[]
                            y=[]
                            for point in range(68):
                                x.append(Landmark[i][point][0])
                                y.append(Landmark[i][point][1])
                            plot(x, y, 'r*')
                            savefig(os.path.join("../data/vis",testSet.filenames[i][-14:]))
                            # misc.imsave(os.path.join("../data/vis",testSet.filenames[i][-14:]), img)
                            clf()
                            cla()
                            close()
                            # draw groundtruth
                            gtLandmarks = testSet.gtLandmarks
                            img = testSet.imgs[i]
                            img = np.reshape(img,(112,112))
                            imshow(img, cmap ='gray')
                            x=[]
                            y=[]
                            for point in range(68):
                                x.append(gtLandmarks[i][point][0])
                                y.append(gtLandmarks[i][point][1])
                            plot(x, y, 'r*')
                            filename = testSet.filenames[i][-14:-3] +'_groundTrue.png'
                            savefig(os.path.join("../data/vis",filename))
                            # misc.imsave(os.path.join("../data/vis",testSet.filenames[i][-14:]), img)
                            clf()
                            cla()
                            close()
                    '''
                    # print(evaluateBatchError(Ytest.reshape([-1, 68, 2]), Landmark.reshape([-1, 68, 2]), Batch_size))

                    BatchErr = sess.run(dan['S1_Cost'],{dan['InputImage']:inputs,\
                        dan['GroundTruth']:targets,dan['S1_isTrain']:False,dan['S2_isTrain']:False})
                    train_err += BatchErr
                
                else:
                    ValidErr ,Landmark = sess.run([dan['S2_Cost'],dan['S2_Ret']],{dan['InputImage']:Xvalid,dan['GroundTruth']:Yvalid,\
                        dan['S1_isTrain']:False,dan['S2_isTrain']:False})
                    
                    '''
                    if epoch%5==0:
                        Landmark= np.reshape(Landmark, [-1, 68, 2])
                        for i in range(len(validationSet.filenames)):
                            # draw prediction
                            img = validationSet.imgs[i]
                            img = np.reshape(img,(112,112))
                            imshow(img, cmap ='gray')
                            x=[]
                            y=[]
                            for point in range(68):
                                x.append(Landmark[i][point][0])
                                y.append(Landmark[i][point][1])
                            plot(x, y, 'r*')
                            savefig(os.path.join("../data/vis",validationSet.filenames[i][-14:]))
                            # misc.imsave(os.path.join("../data/vis",testSet.filenames[i][-14:]), img)
                            clf()
                            cla()
                            close()
                            # draw groundtruth
                            gtLandmarks = validationSet.gtLandmarks
                            img = validationSet.imgs[i]
                            img = np.reshape(img,(112,112))
                            imshow(img, cmap ='gray')
                            x=[]
                            y=[]
                            for point in range(68):
                                x.append(gtLandmarks[i][point][0])
                                y.append(gtLandmarks[i][point][1])
                            plot(x, y, 'r*')
                            filename = validationSet.filenames[i][-14:-3] +'_groundTrue.png'
                            savefig(os.path.join("../data/vis",filename))
                            # misc.imsave(os.path.join("../data/vis",testSet.filenames[i][-14:]), img)
                            clf()
                            cla()
                            close()
                    '''
                    # print(evaluateBatchError(Ytest.reshape([-1, 68, 2]), Landmark.reshape([-1, 68, 2]), Batch_size))
                    BatchErr = sess.run(dan['S2_Cost'],{dan['InputImage']:inputs,\
                        dan['GroundTruth']:targets,dan['S1_isTrain']:False,dan['S2_isTrain']:False})
                    train_err += BatchErr
                print('Epoch: ', epoch, ' Batch: ', Count, 'ValidErr:', ValidErr, ' BatchErr:', BatchErr)
                file=open('../stage_2_train_losses.txt','a')
                file.write('%f\n' % BatchErr)
                file.close()
                file=open('../stage_2_val_losses.txt','a')
                file.write('%f\n' % ValidErr)
                file.close()
            Count += 1
            global_step +=1 
        Saver.save(sess,'./Model_Stage2/Model')
        print('Epoch: ', epoch, ' Train_error:', train_err * 40 /Count)