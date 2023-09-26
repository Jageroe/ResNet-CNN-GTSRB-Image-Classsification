# ResNet CNN - GTSRB - Image Classification

This repository contains a multi-class image classification project. The main objective is to categorize various types of traffic signs using the ResNet CNN architecture and achieve the highest possible accuracy.


## Data Source

The dataset for this project is from Kaggle's [GTSRB - German Traffic Sign Recognition Benchmark](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign?select=Train.csv) competition. Due to its size, I have excluded it from this repository.

It containts 43 distinct classes. With approximately 39,000 training image and 13,000 test images. 

## Project Structure

The repository is organized as follows:

- `ResNet50_Image_Classification.ipynb`: Jupyter notebook where I initialize the data pipeline and the models, train them, and then create some visualizations of the predictions.

- `data_pipeline.py`: Custom dataset class for handling the traffic signs data

- `prediction.py`: Functions to make predictions utilizing my model.

- `training.py`: Functions for the training process.

- `custom_resnet.py`: A custom implementation of a ResNet50 architecture. In the notebook, I eventually used PyTorch's built-in implementation because it's easier to use with pre-trained weights. However, I've kept this custom implementation for educational purposes.

## Results

After training ResNet models, the best model achieved an accuracy of 98% on the test images.