# Image-captioning
## Description

This project implements an image captioning model using PyTorch. The model consists of two main components: an encoder and a decoder. The encoder is a convolutional neural network (CNN) that takes an image as input and produces a fixed-length vector representation of the image. The decoder is a recurrent neural network (RNN) that takes the image representation and generates a sequence of words, one at a time, to form the final caption.



## Installation

Clone the repository

`git clone https://github.com/mohamedelabbas1996/Image-captioning`

`cd Image-captioning`

Install the dependencies:

`pip install -r requirements.txt`

Download the [Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k) dataset and copy the dataset to the project directory

## Usage
```
usage: main.py [-h] [--image_file IMAGE_FILE] [--checkpoint CHECKPOINT] [--n-epochs N_EPOCHS] [--learning-rate LEARNING_RATE] [--batch-size BATCH_SIZE] mode

positional arguments:
  mode                  operation mode can be test or train

optional arguments:
  -h, --help            show this help message and exit
  --image_file IMAGE_FILE
                        path to image file to generate caption for
  --checkpoint CHECKPOINT
                        path for model checkpoint
  --n-epochs N_EPOCHS   number of epochs for training
  --learning-rate LEARNING_RATE
                        optimizer learning rate
  --batch-size BATCH_SIZE
                        Training batch size```
