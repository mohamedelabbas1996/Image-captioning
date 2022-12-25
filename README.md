# Image-captioning
## Description

This project implements an image captioning model using PyTorch, a deep learning framework. The model consists of two main components: an encoder and a decoder. The encoder is a convolutional neural network (CNN) that takes an image as input and produces a fixed-length vector representation of the image. The decoder is a recurrent neural network (RNN) that takes the image representation and generates a sequence of words, one at a time, to form the final caption.



## Installation

Clone the repository

`git clone https://github.com/mohamedelabbas1996/Image-captioning`

`cd Image-captioning`

Install the dependencies:

`pip install -r requirements.txt`

## Usage
### Preprocessing

Before you can train the model, you need to preprocess the dataset. Depending on the dataset you are using, you may need to resize the images and tokenize the captions into a vocabulary. Follow the instructions in the preprocess.py file to preprocess the dataset.

### Training

To train the model, run the following command:

`python train.py`

This will start the training process, which may take several hours or days depending on the size of the dataset and the complexity of the model.

### Evaluation

To evaluate the model's performance on the validation set, run the following command:

`python eval.py`

This will generate the evaluation metrics, such as the BLEU score, for the model.
Generation

To generate captions for new images, run the following command:

`python generate.py --image path/to/image.jpg`

This will pass the image through the encoder and use the decoder to generate the caption.
