# What this Repo is about

Semantic Segmentation is done by assigning each pixel in the image a class. This is a key problem in th field of computer vision. In this repo, we would look at using Cascaded FCN to do this:

## How to set up

## what you need

Ananconda (download from [here](https://anaconda.org/anaconda/python))

## steps

1. Clone the Repo
2. open a terminal (bash)
3. type conda env create -f=environment.yml

## To Launch Jupter notebook

- type jupyter notebook to launch jupyter notebook

## Test data

TODO

## Data cleaning

1. `rename-files.py` to clean file names
2. `augmentor.py` to augment images for training
3. `boundingbox.py` to create cropped images for training and testing
4. `find . -name '*.DS_Store' -type f -delete` to remove .DS_Store files generated my macOS

## What is Cascaded FCN

TODO

## Training

1. `train.py`

## Reference

[1] Automatic Liver and Tumor Segmentation of CT and MRI Volumes using Cascaded Fully Convolutional Neural Networks [white paper here](https://arxiv.org/abs/1702.05970)
