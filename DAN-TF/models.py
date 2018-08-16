#coding=utf-8

# 模型定义

import os
import time
import datetime

from matplotlib import pyplot as plt
import numpy as np

import tensorflow as tf

from layers import AffineTransformLayer, TransformParamsLayer, LandmarkImageLayer, LandmarkTransformLayer


IMGSIZE = 112
N_LANDMARK = 68

def NormRmse(GroudTruth, Prediction):
    Gt = tf.reshape(GroudTruth, [-1, N_LANDMARK, 2])
    Pt = tf.reshape(Prediction, [-1, N_LANDMARK, 2])
    loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.squared_difference(Gt, Pt), 2)), 1) # 
    # norm = tf.sqrt(tf.reduce_sum(((tf.reduce_mean(Gt[:, 36:42, :],1) - \
    #     tf.reduce_mean(Gt[:, 42:48, :],1))**2), 1))
    norm = tf.norm(tf.reduce_mean(Gt[:, 36:42, :],1) - tf.reduce_mean(Gt[:, 42:48, :],1), axis=1)
    # cost = tf.reduce_mean(loss / norm)

    return loss/norm  # 欧式距离/瞳孔之间距离作为loss



def DAN(MeanShapeNumpy):

    MeanShape = tf.constant(MeanShapeNumpy, dtype=tf.float32)
    InputImage = tf.placeholder(tf.float32,[None, IMGSIZE,IMGSIZE,1]) # image size is 112*112, one channel
    GroundTruth = tf.placeholder(tf.float32,[None, N_LANDMARK * 2])
    S1_isTrain = tf.placeholder(tf.bool)
    S2_isTrain = tf.placeholder(tf.bool)
    global_step = tf.placeholder(tf.int32)
    Ret_dict = {}
    Ret_dict['global_step'] = global_step
    Ret_dict['InputImage'] = InputImage
    Ret_dict['GroundTruth'] = GroundTruth
    Ret_dict['S1_isTrain'] = S1_isTrain
    Ret_dict['S2_isTrain'] = S2_isTrain
    initial_learning_rate = 0.01
    with tf.variable_scope('Stage1'):
        S1_Conv1a = tf.layers.batch_normalization(tf.layers.conv2d(InputImage,64,3,1,\
            padding='same',activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer()),training=S1_isTrain)
        S1_Conv1b = tf.layers.batch_normalization(tf.layers.conv2d(S1_Conv1a,64,3,1,\
            padding='same',activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer()),training=S1_isTrain)
        S1_Pool1 = tf.layers.max_pooling2d(S1_Conv1b,2,2,padding='same')

        S1_Conv2a = tf.layers.batch_normalization(tf.layers.conv2d(S1_Pool1,128,3,1,\
            padding='same',activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer()),training=S1_isTrain)
        S1_Conv2b = tf.layers.batch_normalization(tf.layers.conv2d(S1_Conv2a,128,3,1,\
            padding='same',activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer()),training=S1_isTrain)
        S1_Pool2 = tf.layers.max_pooling2d(S1_Conv2b,2,2,padding='same')

        S1_Conv3a = tf.layers.batch_normalization(tf.layers.conv2d(S1_Pool2,256,3,1,\
            padding='same',activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer()),training=S1_isTrain)
        S1_Conv3b = tf.layers.batch_normalization(tf.layers.conv2d(S1_Conv3a,256,3,1,\
            padding='same',activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer()),training=S1_isTrain)
        S1_Pool3 = tf.layers.max_pooling2d(S1_Conv3b,2,2,padding='same')

        S1_Conv4a = tf.layers.batch_normalization(tf.layers.conv2d(S1_Pool3,512,3,1,\
            padding='same',activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer()),training=S1_isTrain)
        S1_Conv4b = tf.layers.batch_normalization(tf.layers.conv2d(S1_Conv4a,512,3,1,\
            padding='same',activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer()),training=S1_isTrain)
        S1_Pool4 = tf.layers.max_pooling2d(S1_Conv4b,2,2,padding='same')

        S1_Pool4_Flat = tf.contrib.layers.flatten(S1_Pool4) # 保留batch_size这个维度，其他的维度拼成一个，用以输入至全连接层
        S1_DropOut = tf.layers.dropout(S1_Pool4_Flat,0.5,training=S1_isTrain) # 一种防止神经网络过拟合的手段。随机的拿掉网络中的部分神经元，从而减小对W权重的依赖，以达到减小过拟合的效果。

        # tf.layers.dense 全连接层
        S1_Fc1 = tf.layers.batch_normalization(tf.layers.dense(S1_DropOut,256,activation=tf.nn.relu,\
            kernel_initializer=tf.contrib.layers.xavier_initializer()),training=S1_isTrain,name = 'S1_Fc1')
        S1_Fc2 = tf.layers.dense(S1_Fc1,N_LANDMARK * 2)
        
        # S1_Fc2 is the increment of meanshape
        S1_Ret = S1_Fc2 + MeanShape
        S1_Cost = tf.reduce_mean(NormRmse(GroundTruth, S1_Ret))
        tf.summary.scalar('S1_Cost',S1_Cost)
        learning_rate = tf.train.exponential_decay(initial_learning_rate,global_step=global_step,decay_steps=100,decay_rate=0.95)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS,'Stage1')):
            S1_Optimizer = tf.train.AdamOptimizer(0.0001).minimize(S1_Cost,\
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"Stage1"))
        
    Ret_dict['S1_Ret'] = S1_Ret
    Ret_dict['S1_Cost'] = S1_Cost
    Ret_dict['S1_Optimizer'] = S1_Optimizer

    with tf.variable_scope('Stage2'):

        S2_AffineParam = TransformParamsLayer(S1_Ret, MeanShape)
        S2_InputImage = AffineTransformLayer(InputImage, S2_AffineParam)
        S2_InputLandmark = LandmarkTransformLayer(S1_Ret, S2_AffineParam)
        S2_InputHeatmap = LandmarkImageLayer(S2_InputLandmark)

        S2_Feature = tf.reshape(tf.layers.dense(S1_Fc1,int((IMGSIZE / 2) * (IMGSIZE / 2)),\
            activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer()),(-1,int(IMGSIZE / 2),int(IMGSIZE / 2),1))
        S2_FeatureUpScale = tf.image.resize_images(S2_Feature,(IMGSIZE,IMGSIZE),1)
        
        S2_ConcatInput = tf.layers.batch_normalization(tf.concat([S2_InputImage,S2_InputHeatmap,S2_FeatureUpScale],3),\
            training=S2_isTrain)
        # the size of S2_ConcatInput is (?,112,112,3)
        # print(S2_ConcatInput)
        S2_Conv1a = tf.layers.batch_normalization(tf.layers.conv2d(S2_ConcatInput,64,3,1,\
            padding='same',activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer()),training=S2_isTrain)
        S2_Conv1b = tf.layers.batch_normalization(tf.layers.conv2d(S2_Conv1a,64,3,1,\
            padding='same',activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer()),training=S2_isTrain)
        S2_Pool1 = tf.layers.max_pooling2d(S2_Conv1b,2,2,padding='same')

        S2_Conv2a = tf.layers.batch_normalization(tf.layers.conv2d(S2_Pool1,128,3,1,\
            padding='same',activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer()),training=S2_isTrain)
        S2_Conv2b = tf.layers.batch_normalization(tf.layers.conv2d(S2_Conv2a,128,3,1,\
            padding='same',activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer()),training=S2_isTrain)
        S2_Pool2 = tf.layers.max_pooling2d(S2_Conv2b,2,2,padding='same')

        S2_Conv3a = tf.layers.batch_normalization(tf.layers.conv2d(S2_Pool2,256,3,1,\
            padding='same',activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer()),training=S2_isTrain)
        S2_Conv3b = tf.layers.batch_normalization(tf.layers.conv2d(S2_Conv3a,256,3,1,\
            padding='same',activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer()),training=S2_isTrain)
        S2_Pool3 = tf.layers.max_pooling2d(S2_Conv3b,2,2,padding='same')

        S2_Conv4a = tf.layers.batch_normalization(tf.layers.conv2d(S2_Pool3,512,3,1,\
            padding='same',activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer()),training=S2_isTrain)
        S2_Conv4b = tf.layers.batch_normalization(tf.layers.conv2d(S2_Conv4a,512,3,1,\
            padding='same',activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer()),training=S2_isTrain)
        S2_Pool4 = tf.layers.max_pooling2d(S2_Conv4b,2,2,padding='same')

        S2_Pool4_Flat = tf.contrib.layers.flatten(S2_Pool4)
        S2_DropOut = tf.layers.dropout(S2_Pool4_Flat,0.5,training=S2_isTrain)

        S2_Fc1 = tf.layers.batch_normalization(tf.layers.dense(S2_DropOut,256,\
            activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer()),training=S2_isTrain)
        S2_Fc2 = tf.layers.dense(S2_Fc1,N_LANDMARK * 2)

        S2_Ret = LandmarkTransformLayer(S2_Fc2 + S2_InputLandmark,S2_AffineParam, Inverse=True) # 转置
        S2_Cost = tf.reduce_mean(NormRmse(GroundTruth,S2_Ret))
        tf.summary.scalar('S2_Cost',S2_Cost)

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS,'Stage2')):
            S2_Optimizer = tf.train.AdamOptimizer(0.0001).minimize(S2_Cost,\
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"Stage2"))

    Ret_dict['S2_Ret'] = S2_Ret
    Ret_dict['S2_Cost'] = S2_Cost
    Ret_dict['S2_Optimizer'] = S2_Optimizer

    Ret_dict['S2_InputImage'] = S2_InputImage
    Ret_dict['S2_InputLandmark'] = S2_InputLandmark
    Ret_dict['S2_InputHeatmap'] = S2_InputHeatmap
    Ret_dict['S2_FeatureUpScale'] = S2_FeatureUpScale
    
    return Ret_dict