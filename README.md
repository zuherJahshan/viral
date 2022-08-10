# CoViT: Real-time phylogenetics for the SARS-CoV-2 pandemic using Vision Transformers

## Table of contents
* [Introduction](#introduction)
* [Technologies](#technologies)
* [Launch](#launch)
* [Prediction-usecase](#prediction-usecase)

## Introduction
CoViT is a program that rapidly predicts the type of variant of a given SARS-CoV-2 assembled genome sequence.
It is most usefull in tracking and control of viral pandemics like the Covid-19 pandemic.

CoViT is based on vision transformer, a neural network primarily developed for image classification.
We use the principles of vision transformer to infer the phylogeny of an assembled SARS-CoV-2 genome.
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

It is most preferable to run the application on a NVIDIA GPU!!!, it is not necessary but it will speed up the computations noticeably

To launch the application, first you need to run the bash script "setup.sh" that is found in the project's root directory

    ./setup.sh

## Prediction-usecase
First, go to the src directory and launch python

    cd src/
    python3.8

Then in Python, first import the following packages

    from DataCollector import DataCollectorv2
    from covit import CovitProject
    
Then, create the data collector, which is a class that manages local raw data and downloads newly remote data.
But most importantly, it is used by the main class the CovitProject.

    dc = DataCollectorv2()
    
Then, load an existing CovitProject. Up for now,
we created 4 different projects which is differentiated by their names:
1. 107Lins
2. 189Lins
3. 269Lins
4. 375Lins

Lets say, for an example that we load the 107Lins project

    covit = CovitProject(project_name="107Lins",
                         data_collector=dc)
                     

Then we need to load an existing prediction model to the project. To see which models exist just call:

    covit.listNNModels()
    
Assume we take the last model in the list

    model_name = covit.listNNModels()[-1]
    
And then we load it to RAM

    covit.loadNNModel(name=model_name)
    
To make a prediction on an assembled genomes, you MUST have them held in some directory.
We provide an examplary directory that holds assembled genomes in FASTA format that is located
in the relative path "../ExampleAccs" (i.e., relative to the src directory).

    path_to_fasta_dir = "../ExampleAccs"
    
Then to make the prediction, just call:
    
    covit.predict(model_name=model_name,
                  path_to_fasta_dir=path_to_fasta_dir)
                  
The predictions to all assembled genomes will be written in a csv file named "results.csv"
which will be available in the directory that contains the assembled genomes (i.e., "../accs").

This running example supplied as a standalone script:

    from DataCollector import DataCollectorv2
    from covit import CovitProject
    dc = DataCollectorv2()
    covit = CovitProject(project_name="107Lins",
                         data_collector=dc)
    model_name = covit.listNNModels()[-1]
    covit.loadNNModel(name=model_name)
    path_to_fasta_dir = "../ExampleAccs"
    covit.predict(model_name=model_name,
                  path_to_fasta_dir=path_to_fasta_dir)
