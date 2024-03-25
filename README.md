The primary objective of this lab is to familiarize oneself with the PyTorch library and to construct various neural architectures such as Convolutional Neural Networks (CNN), Region-based Convolutional Neural Networks (RCNN), Fully Convolutional Neural Networks (FCNN), and Vision Transformers (ViT) for computer vision tasks.
Part 1: CNN Classifier
DataSet

MNIST Dataset: Download here
Tasks

    CNN Architecture Establishment: Construct a CNN architecture using the PyTorch library to classify the MNIST Dataset. This involves defining layers (Convolutional, pooling, fully connected layers), specifying hyperparameters (kernels, padding, stride, optimizers, regularization, etc.), and executing the model in GPU mode.

    Faster R-CNN Implementation: Repeat the process of establishing a model, this time using Faster R-CNN.


    Part 2: Vision Transformer (VIT)
Overview

Vision Transformers (ViT) have emerged as dominant models in the field of Computer Vision since their introduction by Dosovitskiy et. al. in 2020. They have achieved state-of-the-art performance in tasks like image classification and others.
Tasks

    ViT Model Establishment: Follow the tutorial here to construct a Vision Transformer (ViT) model architecture from scratch. Perform a classification task on the MNIST Dataset using this ViT model.

    Result Interpretation and Comparison: Analyze and interpret the results obtained from the ViT model and compare them with the results obtained from the CNN and Faster R-CNN models in the first part.

Note: Ensure all dependencies are installed properly before running the code.

   pip install torch torchvision
