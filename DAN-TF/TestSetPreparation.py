#coding=utf-8

from ImageServer import ImageServer
import numpy as np


# commonSetImageDirs = ["../data/images/lfpw/testset/", "../data/images/helen/testset/"]
commonSetImageDirs = ["../data/images/lfpw/testset/", "../data/images/helen/testset/"]
commonSetBoundingBoxFiles = ["../data/boxesLFPWTest.pkl", "../data/boxesHelenTest.pkl"]

challengingSetImageDirs = ["../data/images/ibug/"]
challengingSetBoundingBoxFiles = ["../data/boxesIBUG.pkl"]

w300SetImageDirs = ["../data/images/300w/01_Indoor/", "../data/images/300w/02_Outdoor/"]
w300SetBoundingBoxFiles = ["../data/boxes300WIndoor.pkl", "../data/boxes300WOutdoor.pkl"]

datasetDir = "../data/"
trainSet = ImageServer.Load(datasetDir + "dataset_nimgs=62960_perturbations=[0.2, 0.2, 20, 0.25]_size=[112, 112].npz")
meanShape = np.load("../data/meanFaceShape.npz")["meanShape"]
 
# print(meanShape.shape)  (68,2)
'''
commonSet = ImageServer(initialization='box')
commonSet.PrepareData(commonSetImageDirs, commonSetBoundingBoxFiles, meanShape, 0, 1000, False)
commonSet.LoadImages()
commonSet.CropResizeRotateAll()
commonSet.imgs = commonSet.imgs.astype(np.float32)
commonSet.NormalizeImages(trainSet) #去均值，除以标准差
# commonSet.NormalizeImages()
commonSet.Save(datasetDir, "commonSet.npz")


challengingSet = ImageServer(initialization='box')
challengingSet.PrepareData(challengingSetImageDirs, challengingSetBoundingBoxFiles, meanShape, 0, 1000, False)
challengingSet.LoadImages()
challengingSet.CropResizeRotateAll()
challengingSet.imgs = challengingSet.imgs.astype(np.float32)
challengingSet.NormalizeImages() #去均值，除以标准差
challengingSet.Save(datasetDir, "challengingSet.npz")
'''
w300Set = ImageServer(initialization='box')
w300Set.PrepareData(w300SetImageDirs, w300SetBoundingBoxFiles, meanShape, 0, 1000, False)
w300Set.LoadImages()
w300Set.CropResizeRotateAll()
w300Set.imgs = w300Set.imgs.astype(np.float32)
w300Set.NormalizeImages(trainSet) #去均值，除以标准差
w300Set.Save(datasetDir, "w300Set.npz")
