# Deep-Alignment-Network_tensorflow
A re-implementation of Deep-Alignment-Network-tensorflow

## Requirements

Tensorflow 1.3.0 (My version)

Python 3.6

Other commonly used libs for image processing

Dataset formatted like 300-W and Menpo

## Usage

### Data preparation

`cd DAN_TF`

`python training\testSetPreparation.py` 
`This command is used to get the trainSet/testSet, and will generate .npz files in the data directory`
`The trainSet contains 60960 pictures. The raw data is 3048 pictures, and after mirroring and perturbation there are 60960 training samples`
`The validationSet contains 100 pictures. And the testSets can divide into three parts, 300W-publicTest, 300W-privateTest and IbugTest`

### Training or testing

`python train\testDAN.py`
`This command is used to train/test the model, and will generate training model in the Model directory`
`The model is supposed to train 1000 epoches each stage. I only trained the first stage for about 350 epoches, and the mean_error with the inter-ocular normalization reaches 0.035690 at 300W-public testSet and 0.051233 at 300W-private testSet`

### Training hyperparameters setting
`batch_size:64`
`base learning rate:0.001`
`learning scheduler:None`
`kernel normalization:xavier_initializer`

## Pre-trained model

Model_138_0.001: Model trained after 138 epoches with learning rate 0.001
Model_79_0.0001: Model trained after 79 epoches with learning rate 0.0001

Remember to set the `STAGE` variable and modify the `data path` in trainDAN.py for different stages of the model