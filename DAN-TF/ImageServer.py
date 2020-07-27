#coding=utf-8
from scipy import ndimage
import numpy as np
import utils
import pickle
import glob
from os import path


class ImageServer(object):
    def __init__(self, imgSize=[112, 112], frameFraction=0.25, initialization='box', color=False):
        self.origLandmarks = []
        self.filenames = []
        self.mirrors = []
        self.meanShape = np.array([])
        self.A = []
        self.t = []
        self.meanImg = np.array([])
        self.stdDevImg = np.array([])

        self.perturbations = []

        self.imgSize = imgSize
        self.frameFraction = frameFraction
        self.initialization = initialization
        self.color = color;

        self.boundingBoxes = []

    # @staticmethod
    def Load(filename):
        imageServer = ImageServer()
        arrays = np.load(filename)
        imageServer.__dict__.update(arrays)

        if (len(imageServer.imgs.shape) == 3):
            imageServer.imgs = imageServer.imgs[:, :, :, np.newaxis]

        return imageServer

    def Save(self, datasetDir, filename=None):
        if filename is None:
            filename = "dataset_nimgs={0}_perturbations={1}_size={2}".format(len(self.imgs), list(self.perturbations), self.imgSize)
            if self.color:
                filename += "_color={0}".format(self.color)
            filename += ".npz"

        arrays = {key:value for key, value in self.__dict__.items() if not key.startswith('__') and not callable(key)}
        np.savez(datasetDir + filename, **arrays)

    def PrepareData(self, imageDirs, boundingBoxFiles, meanShape, startIdx, nImgs, mirrorFlag):
        filenames = []#此list和filenamesInDir完全一样，可以去除
        landmarks = []#此list用于存储特征点坐标
        boundingBoxes = []#此list用于存储bbx
        # import pdb; pdb.set_trace()
        '''
        # skip the missing pic
        missing = []
        missingfile = open('missing.txt')
        lines = missingfile.readlines()
        for line in lines:
            if line[-1] == "\n":
                missing.append(line[:-1])
            else:
                missing.append(line)
        # print(len(missing))
        # print(missing)
        '''
        for i in range(len(imageDirs)):
            filenamesInDir = glob.glob(imageDirs[i] + "*.jpg")
            filenamesInDir += glob.glob(imageDirs[i] + "*.png")
            # import pdb; pdb.set_trace()
            if boundingBoxFiles is not None:
                # boundingBoxDict = pickle.load(open(boundingBoxFiles[i], 'rb'),encoding='bytes')
                boundingBoxDict = pickle.load(open(boundingBoxFiles[i], 'rb'),encoding='latin1')
                # print(boundingBoxDict)

            for j in range(len(filenamesInDir)):
                # print(filenamesInDir[j])
                '''
                index = filenamesInDir[j].rfind("/")
                picname = filenamesInDir[j][index+1:]
                if picname in missing:
                    # print(picname)
                    continue
                '''
                filenames.append(filenamesInDir[j])
                
                ptsFilename = filenamesInDir[j][:-3] + "pts"
                landmarks.append(utils.loadFromPts(ptsFilename))

                if boundingBoxFiles is not None:
                    basename = path.basename(filenamesInDir[j])
                    if boundingBoxDict.get(basename).all()==None:
                        print(basename)
                        continue
                    boundingBoxes.append(boundingBoxDict[basename])
                

        # nImgs = len(filenames) # 在原作者代码中，nImgs=2，所以只能用两张图片，翻转+pertubation=40张图片进行训练。
        # print(nImgs)
        filenames = filenames[startIdx : startIdx + nImgs]
        landmarks = landmarks[startIdx : startIdx + nImgs]
        boundingBoxes = boundingBoxes[startIdx : startIdx + nImgs]

        mirrorList = [False for i in range(nImgs)]
        if mirrorFlag:     
            mirrorList = mirrorList + [True for i in range(nImgs)]
            filenames = np.concatenate((filenames, filenames))

            landmarks = np.vstack((landmarks, landmarks)) # 合并
            boundingBoxes = np.vstack((boundingBoxes, boundingBoxes))       

        self.origLandmarks = landmarks
        self.filenames = filenames
        self.mirrors = mirrorList
        self.meanShape = meanShape
        self.boundingBoxes = boundingBoxes

    def LoadImages(self):
        self.imgs = []
        self.initLandmarks = []
        self.gtLandmarks = []

        for i in range(len(self.filenames)):
           #这段代码写得不好，self.color并没有实际意
            if i%100==0:
                print(i)
            img = ndimage.imread(self.filenames[i])
            if self.color:
            	
                if len(img.shape) == 2:
                    img = np.dstack((img, img, img))
            else:
            	# img = ndimage.imread(self.filenames[i], mode='L')
                if len(img.shape) > 2:
                    img = np.mean(img, axis=2)
            img = img.astype(np.uint8)

            if self.mirrors[i]:
                self.origLandmarks[i] = utils.mirrorShape(self.origLandmarks[i], img.shape)
                img = np.fliplr(img)

            if not self.color:
            #     img = np.transpose(img, (2, 0, 1))
            # else:
                img = img[np.newaxis]#img从shape(H,W)变成shape(1,H,W)
                # img = np.transpose(img, (1, 2, 0))

            groundTruth = self.origLandmarks[i]

            if self.initialization == 'rect':
                bestFit = utils.bestFitRect(groundTruth, self.meanShape)#仅仅把meanshape适应进入由landmark确定的框中
            elif self.initialization == 'similarity':
                bestFit = utils.bestFit(groundTruth, self.meanShape)#找到meanShape到gt的最优变换，并变换之
            elif self.initialization == 'box':
                bestFit = utils.bestFitRect(groundTruth, self.meanShape, box=self.boundingBoxes[i])#仅仅把meanshape适应进入由检测到的bbx确定的框中

            self.imgs.append(img)
            self.initLandmarks.append(bestFit)
            self.gtLandmarks.append(groundTruth)

        self.initLandmarks = np.array(self.initLandmarks)
        self.gtLandmarks = np.array(self.gtLandmarks)    

    def GeneratePerturbations(self, nPerturbations, perturbations):
        self.perturbations = perturbations
        meanShapeSize = max(self.meanShape.max(axis=0) - self.meanShape.min(axis=0))
        destShapeSize = min(self.imgSize) * (1 - 2 * self.frameFraction)
        scaledMeanShape = self.meanShape * destShapeSize / meanShapeSize

        newImgs = []  
        newGtLandmarks = []
        newInitLandmarks = []           

        translationMultX, translationMultY, rotationStdDev, scaleStdDev = perturbations

        rotationStdDevRad = rotationStdDev * np.pi / 180         
        translationStdDevX = translationMultX * (scaledMeanShape[:, 0].max() - scaledMeanShape[:, 0].min())
        translationStdDevY = translationMultY * (scaledMeanShape[:, 1].max() - scaledMeanShape[:, 1].min())
        print("Creating perturbations of " + str(self.gtLandmarks.shape[0]) + " shapes")

        for i in range(self.initLandmarks.shape[0]):
            print(i)
            for j in range(nPerturbations):
                tempInit = self.initLandmarks[i].copy()

                angle = np.random.normal(0, rotationStdDevRad)
                offset = [np.random.normal(0, translationStdDevX), np.random.normal(0, translationStdDevY)]
                scaling = np.random.normal(1, scaleStdDev)

                R = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])
            
                tempInit = tempInit + offset
                tempInit = (tempInit - tempInit.mean(axis=0)) * scaling + tempInit.mean(axis=0)            
                tempInit = np.dot(R, (tempInit - tempInit.mean(axis=0)).T).T + tempInit.mean(axis=0)

                tempImg, tempInit, tempGroundTruth = self.CropResizeRotate(self.imgs[i], tempInit, self.gtLandmarks[i])#位移0.2，旋转20度，放缩+-0.25              

                newImgs.append(tempImg.transpose((1,2,0)))
                newInitLandmarks.append(tempInit)
                newGtLandmarks.append(tempGroundTruth)

        self.imgs = np.array(newImgs)
        self.initLandmarks = np.array(newInitLandmarks)
        self.gtLandmarks = np.array(newGtLandmarks)

    def CropResizeRotateAll(self):
        self.A=[]
        self.t=[]
        newImgs = []  
        newGtLandmarks = []
        newInitLandmarks = []   
        # initLandmarks 根据bounding box 缩放的landmarks
        for i in range(self.initLandmarks.shape[0]):
            # print(self.initLandmarks[i])
            tempImg, tempInit, tempGroundTruth = self.CropResizeRotate(self.imgs[i], self.initLandmarks[i], self.gtLandmarks[i])

            newImgs.append(tempImg.transpose((1,2,0))) # 将图片进行变换
            newInitLandmarks.append(tempInit)
            newGtLandmarks.append(tempGroundTruth)

        self.imgs = np.array(newImgs)
        self.initLandmarks = np.array(newInitLandmarks)
        self.gtLandmarks = np.array(newGtLandmarks)  

    def NormalizeImages(self, imageServer=None):
        self.imgs = self.imgs.astype(np.float32)

        if imageServer is None:
            self.meanImg = np.mean(self.imgs, axis=0)
        else:
            self.meanImg = imageServer.meanImg

        self.imgs = self.imgs - self.meanImg
        
        if imageServer is None:
            self.stdDevImg = np.std(self.imgs, axis=0)
        else:
            self.stdDevImg = imageServer.stdDevImg
        
        self.imgs = self.imgs / self.stdDevImg # zero-mean normalization  减去平均数再除以标准差 使经过处理的数据符合标准正态分布

        from matplotlib import pyplot as plt  

        meanImg = self.meanImg - self.meanImg.min()
        meanImg = 255 * meanImg / meanImg.max()  
        meanImg = meanImg.astype(np.uint8)   
        if self.color:
            # plt.imshow(np.transpose(meanImg, (1, 2, 0)))
            plt.imshow(meanImg)
        else:
            print(meanImg)
            plt.imshow(meanImg[:,:,0], cmap=plt.cm.gray)
        plt.savefig("../meanImg.jpg")
        plt.clf()

        stdDevImg = self.stdDevImg - self.stdDevImg.min()
        stdDevImg = 255 * stdDevImg / stdDevImg.max()  
        stdDevImg = stdDevImg.astype(np.uint8)   
        if self.color:
            # plt.imshow(np.transpose(stdDevImg, (1, 2, 0)))
            plt.imshow(stdDevImg)
        else:
            plt.imshow(stdDevImg[:,:,0], cmap=plt.cm.gray)
        plt.savefig("../stdDevImg.jpg")
        plt.clf()

    def CropResizeRotate(self, img, initShape, groundTruth):
		# meanShape 平均人脸，未做任何变换的
        meanShapeSize = max(self.meanShape.max(axis=0) - self.meanShape.min(axis=0)) # 每一列取出最大的值和最小的值相减 ，得到最大的长和宽，并取最大作为meanShapeSize
        destShapeSize = min(self.imgSize) * (1 - 2 * self.frameFraction) # what is frameFraction

        scaledMeanShape = self.meanShape * destShapeSize / meanShapeSize # 将meanShape缩至（112*112）* 0.5

        destShape = scaledMeanShape.copy() - scaledMeanShape.mean(axis=0) # 将meanShape放在坐标轴原点
        offset = np.array(self.imgSize[::-1]) / 2 # 抵消
        destShape += offset
        # 将meanShape映射至112*112图像中
        # initShape是投射至boundingbox的中心后的
        A, t = utils.bestFit(destShape, initShape, True) # 得到能够将initShape缩放至destShape的比例
        self.A.append(A)
        self.t.append(t)
        A2 = np.linalg.inv(A) # 矩阵求逆
        t2 = np.dot(-t, A2) 

        outImg = np.zeros((img.shape[0], self.imgSize[0], self.imgSize[1]), dtype=img.dtype)
        for i in range(img.shape[0]):
            outImg[i] = ndimage.interpolation.affine_transform(img[i], A2, t2[[1, 0]], output_shape=self.imgSize)

        initShape = np.dot(initShape, A) + t # 仿射变化

        groundTruth = np.dot(groundTruth, A) + t
        return outImg, initShape, groundTruth


