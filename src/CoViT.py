import keras.models
import pandas as pd
import matplotlib.pyplot as plt
from Types import *
from math import ceil

from DataCollector import DataCollector
from Genome import Genome, AccessionNotFoundLocally, base_count
from Model import CoViTModel, custom_objects

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

class HyperParameters(object):
    def __init__(self):
        """
        These hyperparameter controls the neural network model. The default definition of those parameters are taken
        from the paper Attention is all you need.
        """
        self._encoder_repeats: int = 4   # Number of times the encoder block is repeated
        self._d_out: int = 2             # Number of classes from which we should classify
        self._d_model: int = 256         # The dimensionality of the feature vectors also equals fragment_length
        self._d_val: int = 64            # The dimensionality of the value representation of a fragment (for self attention)
        self._d_key: int = 64            # The dimensionality of the key representation of a fragment (for self attention)
        self._heads: int = 12             # Number of heads used in the self attention layer.
        self._d_ff: int = 2048           # Feed forward hidden layer inner number of units.
        self._dropout_rate: float = 0.1  # Dropout rate for all sub-layers in the model

        self._kmer_size: int = 30        # Anchor kmer size. This hyperparameter controls the type of genome encoding
        self._n: int = 196               # Compression factor, controls the information extracted from genome to the encoding

        self._batch_size = 32
        self._epochs = 20

    @property
    def encoder_repeats(self):
        return self._encoder_repeats

    @encoder_repeats.setter
    def encoder_repeats(self,
                        encoder_repeats: int):
        self._encoder_repeats = encoder_repeats

    @property
    def d_out(self):
        return self._d_out

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

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self,
                   batch_size):
        self._batch_size = batch_size

    @property
    def epochs(self):
        return self._epochs

    @epochs.setter
    def epochs(self,
               epochs):
        self._epochs = epochs

class CoViTDataset(object):
    """
    The Dataset is initialized by specifying a list of "of interest" lineages to classify and the relevant size of
    datasets.
    The dataset will hold a list of all possible lineages and a map lineage |-> accessions list (this list will be shuffled).
    The tf.data.Dataset will be built with from_generator function, where the sample generator, will generate samples
    according to iterations on lineages of those which are found in the lineage_accessions map.

    """
    def __init__(self,
                 lineages: List[Lineage],
                 data_collector: DataCollector,
                 HP: HyperParameters):
        """
        Builds a tf.data.Dataset object according to the
        """

        # Data collector
        self.data_collector = data_collector

        # For all lineage that could be found in the dataframe from the data_collector, we get
        self.lineage_accessions_map: LineageAccessionsMap = {}
        self._buildLineageAccessionMap(lineages=lineages)

        # Not existing accessions
        self.not_existing_accs: List[Accession] = []

        self.activated_lineages: List = lineages

        # Build lineage label map, those are not corresponding to all lineages, rather they are those lineages given by the user.
        self.lineage_label_map: LineageLabelMap = {}
        self._buildLineageLabelMap(lineages=lineages)

        self.HP = HP

    def _buildLineageAccessionMap(self,
                                  lineages):
        for lineage in lineages:
            accs = self.data_collector.getLocalAccessions(lineage)
            accs_size = len(accs)
            devider = int(accs_size * 0.8)
            self.lineage_accessions_map.update({lineage: (accs[:devider], accs[devider:])})
            print("Done with lineage {} train set size is {} valid set size is {}".format(lineage, devider, accs_size-devider))
        return


    def _handleNotExistingAcc(self,
                              acc_id: Accession):
        self.not_existing_accs.append(acc_id)
        if len(self.not_existing_accs) >= 10:    # Should change to a variable
            self.data_collector.getSeqsByAcc(self.not_existing_accs)
            self.not_existing_accs = []

    def _genomeGenerator(self,
                         size: int,
                         dataset_type: str = 'train'):
        current_size = 0    # Already sampled examples
        lineages_done = 0   # Count of all lineages that generated all possible examples
        lineages_cnt = len(self.activated_lineages)
        index = 0   # all lineages iteration sample index
        if dataset_type == 'train':
            dataset = 0
        else:
            dataset = 1
        while lineages_done < lineages_cnt and current_size < size:
            for lineage in self.activated_lineages:
                if index == len(self.lineage_accessions_map[lineage][dataset]): # if there are no more examples from lineage.
                    lineages_done += 1
                    continue
                elif index > len(self.lineage_accessions_map[lineage][dataset]):
                    continue
                acc = self.lineage_accessions_map[lineage][dataset][index]
                try:
                    genome = Genome(accession_id=acc,
                                    data_collector=self.data_collector)
                    genome_tensor = genome.getFeatureTensor(kmer_size=self.HP.kmer_size,
                                                            fragment_size=self.HP.d_model,
                                                            n=self.HP.n)
                except AccessionNotFoundLocally:
                    self._handleNotExistingAcc(acc_id=acc)
                    current_size += 1
                    continue
                except ValueError:
                    current_size += 1
                    continue
                yield genome_tensor, self.lineage_label_map[lineage]
                current_size += 1
            index += 1

    def _buildLineageLabelMap(self,
                              lineages: List[Lineage]) -> LineageLabelMap:
        lineages_copy = lineages.copy()
        lineages_copy.sort()
        lineage_label_map: LineageLabelMap = {}
        num_of_lineages = len(lineages_copy)
        for i in range(num_of_lineages):
            lineage_label_map.update({lineages_copy[i]: tf.squeeze(tf.one_hot(indices=[i],
                                                                              depth=num_of_lineages,
                                                                              axis=0))})
        self.lineage_label_map = lineage_label_map

    def getTrainSet(self,
                    size: int = 1000,
                    shuffle_buffer_size: int = 100):
        train_set: tf.data.Dataset =\
            tf.data.Dataset.from_generator(generator=lambda: self._genomeGenerator(size=size,
                                                                                   dataset_type='train'),
                                           output_signature=(tf.TensorSpec(shape=(self.HP.n,
                                                                                  self.HP.d_model,
                                                                                  base_count),
                                                                           dtype=tf.float32),
                                                             tf.TensorSpec(shape=(self.HP.d_out),
                                                                           dtype=tf.float32)))
        # Output signature is expected to be a tupple of observation and prediction.
        train_set = train_set.shuffle(shuffle_buffer_size).repeat(self.HP.epochs).\
            batch(self.HP.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        return train_set

    def getValidSet(self,
                    size: int = 100):
        valid_set: tf.data.Dataset =\
            tf.data.Dataset.from_generator(generator=lambda: self._genomeGenerator(size=size,
                                                                                   dataset_type='validation'),
                                           output_signature=(tf.TensorSpec(shape=(self.HP.n,
                                                                                  self.HP.d_model,
                                                                                  base_count),
                                                                           dtype=tf.float32),
                                                             tf.TensorSpec(shape=(self.HP.d_out),
                                                                           dtype=tf.float32)))
        return valid_set.batch(self.HP.batch_size).prefetch(tf.data.experimental.AUTOTUNE)



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
                 lineages: List[Lineage] = None,
                 num_lineages: int = -1,
                 batch_size: int = 32,
                 epochs: int = 1000):
        data_collector = DataCollector()
        if lineages == None:
            lineages = list(data_collector.getLocalLineages())
            if num_lineages > 0:
                lineages = self._chooseLineages(data_collector=data_collector,
                                                lineages=lineages,
                                                num_lineages=num_lineages)

        # Set hyper parameters
        self.HP: HyperParameters = HyperParameters()
        self.HP.d_out = len(lineages)   # Number of virus classes
        self.HP.batch_size = batch_size
        self.HP.epochs = epochs

        self.dataset: CoViTDataset = CoViTDataset(lineages=lineages,
                                                  data_collector=data_collector,
                                                  HP=self.HP)

        # Set model
        self.nn_model: tf.keras.Model = CoViTModel(N=self.HP.encoder_repeats,
                                                   d_out=self.HP.d_out,
                                                   d_model=self.HP.d_model,
                                                   d_val=self.HP.d_val,
                                                   d_key=self.HP.d_key,
                                                   d_ff=self.HP.d_ff,
                                                   heads=self.HP.heads,
                                                   dropout_rate=self.HP.dropout_rate)

    def _chooseLineages(self,
                        data_collector: DataCollector,
                        lineages: List[Lineage],
                        num_lineages: int):
        def lineage_key(elem):
            return elem[0]

        def count_key(elem):
            return elem[1]

        num_lineages = min(num_lineages, len(lineages))

        lin_size_list: List[(Lineage, int)] = []
        for lineage in lineages:
            lin_size_list.append((lineage, data_collector.getLocalAccessionsCount(lineage)))

        lin_size_list.sort(key=count_key,
                           reverse=True)
        print("====> The chosen lineages are: <====")
        print(lin_size_list[:num_lineages])

        chosen_lins = []
        for i in range(num_lineages):
            chosen_lins.append(lin_size_list[i][0])

        return chosen_lins[:num_lineages]


    def train(self,
              size: int = 1024,
              shuffle_buffer_size: int = 100):
        # Compile the model
        print("Started training")
        if self.HP.d_out == 2:
            loss = "binary_crossentropy"
        elif self.HP.d_out > 2:
            loss = "categorical_crossentropy"
        else:
            assert False, "The number of accessions provided must be at least 2."
        metrics = [tf.keras.metrics.TopKCategoricalAccuracy(k=1),
                   tf.keras.metrics.TopKCategoricalAccuracy(k=2),
                   tf.keras.metrics.TopKCategoricalAccuracy(k=5)
                   ]
        self.nn_model.compile(loss=loss,
                              optimizer=tf.keras.optimizers.Adam(clipnorm=1.0),
                              metrics=["accuracy"])  # the optimizer will change after debugging

        # Train the model
        # Note: should not specify batch_size since generator generate batches itself.
        # https://www.tensorflow.org/api_docs/python/tf/keras/Model
        return self.nn_model.fit(x=self.dataset.getTrainSet(size=size,
                                                               shuffle_buffer_size=shuffle_buffer_size),
                                    epochs=self.HP.epochs,
                                    steps_per_epoch=ceil(size/(self.HP.batch_size)))

    def evaluate(self,
                 size):
        """
        Just evaluate the results on the training set, on the validation set.
        """
        results = self.nn_model.evaluate(x=self.dataset.getValidSet(size),
                                         verbose=1)
        print(results)  # only for debug
        return results

    def plot(self,
             history):
        self.nn_model.summary()
        pd.DataFrame(history.history).plot(figsize=(8, 5))
        plt.grid(True)
        plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
        plt.savefig("results.png")

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
        return False

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
        self.nn_model.save(model_name)
        return

    def loadNNModel(self,
                    model_name):
        self.nn_model = keras.models.load_model(model_name,
                                                custom_objects=custom_objects)

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

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

num_lineages = 5
samples_per_lineage = 64
valid_cut = 0.2
batch_size = 64
epochs = 1

"""
covit = CoViT(batch_size=batch_size,
              num_lineages=num_lineages,
              epochs=epochs)

train_size = int((1-valid_cut)*num_lineages*samples_per_lineage)
valid_size = int(valid_cut*num_lineages*samples_per_lineage)

# dataset is 80-20 of 8192 approx.
history = covit.train(size=train_size,
                      shuffle_buffer_size=100)
covit.plot(history)
covit.evaluate(size=valid_size)

covit.saveNNModel("first_model.nnmodel")

print("loading the model")
covit2 = CoViT(batch_size=batch_size,
               num_lineages=num_lineages,
               epochs=epochs)
covit2.loadNNModel("first_model.nnmodel")
covit.evaluate(size=valid_size)
"""
covit = CoViT(batch_size=batch_size,
              num_lineages=num_lineages,
              epochs=epochs)
