import tensorflow as tf
import pandas as pd
import numpy as np

from DataCollector import DataCollector
from Genome import Genome
from Model import CoViTModel

class HyperParameters(object):
    def __init__(self):
        self.encoder_repeats: int = 8   # Number of times the encoder block is repeated
        self.d_out: int = 2             # Number of classes from which we should classify
        self.d_model: int               # the dimensionality of the feature vectors also equals fragment_length


class CoViT(object):
    """
    This class is the human interface class.
    It will manage the main functionalities:
    1. train
    2. fit
    3. loadModel
    4. setHP
    """
    def __init__(self):
        # Set the DataCollector
        self.data_collector = DataCollector()

        # Set the Accessions dataframe which holds all metadata
        self.acc_df = self.data_collector.getAccDF()

        # Set model


