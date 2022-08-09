# CoViT: Real-time phylogenetics for the SARS-CoV-2 pandemic using Vision Transformers

## Introduction
CoViT is a program that rapidly predicts the type of variant of a given SARS-CoV-2 assembled genome sequence.
It is most usefull in tracking and control of viral pandemics like the Covid-19 pandemic.

CoViT is based on vision transformer, a neural network primarily developed for image classification.
We use the principles of vision transformer to inferring the phylogeny of an assembled SARS-CoV-2 genome.
We apply a pre-processing step that employs MinHash, whose role is extracting the informative sub-sequences from the genome,
which are further fed into a modified vision transformer for classification.

## Technologies
Ubuntu 20.04.4 LTS
Python 3.8.10
Numpy 1.22.3
Pandas 1.4.2
TensorFlow 2.9.1

## Launch
The application is compatible (at the moment) only with operating systems belonging to the linux family.
To run the application, you MUST have the following:
1. Python 3.8.10
2. Numpy 1.22.3
3. Pandas 1.4.2
4. TensorFlow 2.9.1
*It is most preferable to run the application on a NVIDIA GPU!!!, it is not necessary but it will speed up the computations noticeably*
