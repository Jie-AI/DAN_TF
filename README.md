# Deep-Alignment-Network-tensorflow
A re-implementation of Deep-Alignment-Network using TensorFlow

## Requirements

Tensorflow 1.2.0 (My version)

Python 3.6

Other commonly used libs for image processing

Dataset formatted like 300-W and Menpo

## Usage

### Data preparation

`cd DAN_TF`

`mkdir Model`


`python training\testSetPreparation.py`

### Training or testing

`python train\testDAN.py`

Remember to set the `STAGE` variable and modify the `data path` in trainDAN.py
