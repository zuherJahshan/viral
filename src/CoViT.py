import tensorflow as tf
import pandas as pd
import numpy as np
from typing import List, Dict

from DataCollector import DataCollector
from Genome import Genome, FileNotFound, base_count
from Model import CoViTModel

Accession = str
Lineage = str
LineageAccessionsMap = Dict[Accession, Dict[str, List[Accession]]]

class HyperParameters(object):
    def __init__(self):
        """
        These hyperparameter controls the neural network model. The default definition of those parameters are taken
        from the paper Attention is all you need.
        """
        self._encoder_repeats: int = 8   # Number of times the encoder block is repeated
        self._d_out: int = 2             # Number of classes from which we should classify
        self._d_model: int = 256         # The dimensionality of the feature vectors also equals fragment_length
        self._d_val: int = 64            # The dimensionality of the value representation of a fragment (for self attention)
        self._d_key: int = 64            # The dimensionality of the key representation of a fragment (for self attention)
        self._heads: int = 8             # Number of heads used in the self attention layer.
        self._d_ff: int = 2048           # Feed forward hidden layer inner number of units.
        self._dropout_rate: float = 0.1  # Dropout rate for all sub-layers in the model

        self._kmer_size: int = 30        # Anchor kmer size. This hyperparameter controls the type of genome encoding
        self._n: int = 256               # Compression factor, controls the information extracted from genome to the encoding

    @property
    def encoder_repeats(self):
        return self._encoder_repeats

    @encoder_repeats.setter
    def encoder_repeats(self,
                        encoder_repeats: int):
        self._encoder_repeats = encoder_repeats

    @property
    def d_out(self):
        return self._encoder_repeats

    @d_out.setter
    def d_out(self,
              d_out: int):
        self._d_out = d_out

    @property
    def d_model(self):
        return self._d_model

    @d_model.setter
    def d_model(self,
                d_model):
        self._d_model = d_model

    @property
    def d_val(self):
        return self._d_val

    @d_val.setter
    def d_val(self,
              d_val):
        self._d_val = d_val

    @property
    def d_key(self):
        return self._d_key

    @d_key.setter
    def d_key(self,
              d_key):
        self._d_key = d_key

    @property
    def heads(self):
        return self._heads

    @heads.setter
    def heads(self,
              heads):
        self._heads = heads

    @property
    def d_ff(self):
        return self._d_ff

    @d_ff.setter
    def d_ff(self,
             d_ff):
        self._d_ff = d_ff

    @property
    def dropout_rate(self):
        return self._dropout_rate

    @dropout_rate.setter
    def dropout_rate(self,
                     dropout_rate):
        self._dropout_rate = dropout_rate

    @property
    def kmer_size(self):
        return self._kmer_size

    @kmer_size.setter
    def kmer_size(self,
                  kmer_size):
        self._kmer_size = kmer_size

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self,
          n):
        self._n = n


class CoViT(object):
    """
    This class is the human interface class.
    It will manage the main functionalities:
    1. train
    2. fit
    3. loadModel
    4. setHP
    """
    def __init__(self,
                 lineages: List[str],
                 dataset_size: str,
                 shuffle_buffer_size: int = 1000):
        # Random state
        self.rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(133))) # 133 is an arbitrary number, can be changed

        # Set the DataCollector
        self.data_collector: DataCollector = DataCollector()

        # Set the Accessions dataframe which holds all metadata
        self.acc_df: pd.DataFrame = self.data_collector.getAccDF()

        # Set hyper parameters
        self.HP: HyperParameters = HyperParameters()
        self.HP.d_out = len(lineages)   # Number of virus classes

        # Set model
        self.nn_model: tf.keras.Model = CoViTModel(N=self.HP.encoder_repeats,
                                                   d_out=self.HP.d_out,
                                                   d_model=self.HP.d_model,
                                                   d_val=self.HP.d_val,
                                                   d_key=self.HP.d_key,
                                                   d_ff=self.HP.d_ff,
                                                   heads=self.HP.heads,
                                                   dropout_rate=self.HP.dropout_rate)

        # Set Datasets
        self.lineage_accessions_map: LineageAccessionsMap = self._getLineageAccessionsMap(lineages,
                                                                                          dataset_size)
        self.train_set: tf.data.Dataset =\
            tf.data.Dataset.from_generator(generator=lambda: self._dataGenerator('train'),
                                           output_signature=tf.TensorSpec(shape=(None,
                                                                                 self.HP.d_model,
                                                                                 base_count),
                                                                          dtype=tf.float32))
        self.train_set = self.train_set.shuffle(shuffle_buffer_size).repeat().prefetch(1)

        self.valid_set: tf.data.Dataset =\
            tf.data.Dataset.from_generator(generator=lambda: self._dataGenerator('validation'),
                                           output_signature=tf.TensorSpec(shape=(None,
                                                                                 self.HP.d_model,
                                                                                 base_count),
                                                                          dtype=tf.float32))
        self.valid_set = self.valid_set.shuffle(shuffle_buffer_size).repeat().prefetch(1)

    def _dataGenerator(self,
                       dataset_type: str) -> tf.Tensor:
        assert dataset_type == 'train' or dataset_type == 'validation',\
            "dataset_type argument must be one of the two: {'train', 'validation'}"
        lineages_cnt = len(self.lineage_accessions_map)
        lineages_done_scanning = 0
        while lineages_done_scanning < lineages_cnt:
            index = 0
            for lineage in self.lineage_accessions_map:
                if index == len(self.lineage_accessions_map[lineage][dataset_type]):
                    lineages_done_scanning += 1
                    continue
                elif index > len(self.lineage_accessions_map[lineage][dataset_type]):
                    continue
                acc = self.lineage_accessions_map[lineage][dataset_type][index]
                try:
                    genome = Genome(accession_id=acc,
                                    data_collector=self.data_collector)
                    genome_tensor = genome.getFeatureTensor(kmer_size=self.HP.kmer_size,
                                                            fragment_size=self.HP.d_model,
                                                            n=self.HP.n)
                except FileNotFound:
                    continue
                yield genome_tensor
            index += 1

    def _getLineageAccessionsMap(self,
                                 lineages: List[Lineage],
                                 dataset_size: str) -> LineageAccessionsMap:
        lineage_accessions_map: LineageAccessionsMap = {}
        for lineage in lineages:
            df = self.acc_df[self.acc_df['lineage'] == lineage]
            df.dropna(subset=['acc', 'lineage'],
                      inplace=True)    # TODO? change in the future

            # Manipulating the accessions list
            train_set_size = self._getDatasetSize(dataset_size)['train']
            validation_set_size = self._getDatasetSize(dataset_size)['validation']
            accessions = self.rs.permutation(df['acc'].values) # TODO: should check that indexing is not out of bounds
            train_valid_accs = {'train': accessions[:train_set_size],
                                'validation': accessions[train_set_size:validation_set_size]}
            lineage_accessions_map.update({lineage: train_valid_accs})
        return lineage_accessions_map


    def _getDatasetSize(self,
                        dataset_size: str):
        if dataset_size == 'tiny':
            return {'train': 160,
                    'validation': 40}       # 80-20
        elif dataset_size == 'small':
            return {'train': 1600,
                    'validation': 400}      # 80-20
        elif dataset_size == 'medium':
            return {'train': 16000,
                    'validation': 4000}     # 80-20
        elif dataset_size == 'large':
            return {'train': 180000,
                    'validation': 20000}    # 90-10
        elif dataset_size == 'huge':
            return {'train': 1900000,
                    'validation': 100000}   # 95-5
        else:
            assert False,\
                "Dataset size parameter should be one of the following: 'tiny', 'small', 'medium', 'large', 'huge'"

    def train(self):
        return

    def evaluate(self):
        """
        Just evaluate the results on the training set, on the validation set.
        """
        return

    def loadModel(self,
                  saved_model_path: str):
        """
        given a saved model, just load it in. For a reference see how it is done in the TestModel file
        """
        return

    def _updateNNModel(self):
        """
        There is some hyper parameters that if changed, the model should be forsaken and a new one
        """
        return

    def saveNNModel(self,
                    model_name):
        """
        Saves the model, maybe a model name should not be given,
        else only saves the model according to HP and date.
        Models:
        --> 8-256-64-64-8-2048-30-256 (Hyper parameters)
            --> set { lineage_set <--> lineage_set_id }
            --> lineage1
                --> date1
                --> date2
                --> date3
            --> lineage2
                --> date1
                --> date2
        """
        return

    def setHP(self,
              encoder_repeats: int = None,
              d_out: int = None,
              d_model: int = None,
              d_val: int = None,
              d_key: int = None,
              heads: int = None,
              d_ff: int = None,
              kmer_size: int = None,
              n: int = None,
              transfer_model: bool = True):

        to_be_updated: bool = False
        if encoder_repeats is not None:
            self.HP.encoder_repeats = encoder_repeats
            to_be_updated = True

        if d_out is not None:
            self.HP.d_out = d_out
            to_be_updated = True

        if d_model is not None:
            self.HP.d_model = d_model
            to_be_updated = True

        if d_val is not None:
            self.HP.d_val = d_val
            to_be_updated = True

        if d_key is not None:
            self.HP.d_key = d_key
            to_be_updated = True

        if heads is not None:
            self.HP.heads = heads
            to_be_updated = True

        if d_ff is not None:
            self.HP.d_ff = d_ff
            to_be_updated = True

        if kmer_size is not None:
            self.HP.kmer_size = kmer_size

        if n is not None:
            self.HP.n = n

        if to_be_updated or not transfer_model:
            self._updateNNModel()
        return



covit = CoViT(["B.1.1.7"], "tiny")
