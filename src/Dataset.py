import os
import pickle
import random
import threading

from Types import *
from DataCollector import DataCollectorv2
from Genome import Genome, base_count

from math import ceil

from enum import Enum

class DatasetHPs(object):
    def __init__(self,
                 lineages: List[Lineage],
                 max_accs_per_lineage: int,
                 frag_len: int,
                 kmer_size: int,
                 n: int,
                 validation_split: float):

        # TODO: Add sanity checks on the parameters of the Dataset params. To do so, please refer to the class of HPs
        #  in CoViT as an example
        self.lineages = lineages
        self.lineages.sort()
        self.max_accs_per_lineage = max_accs_per_lineage
        self.frag_len = frag_len
        self.kmer_size = kmer_size
        self.n = n
        self.validation_split = validation_split

        self.valid_size = 0
        self.train_size = 0

    def updateSizes(self,
                    train_size: int,
                    valid_size: int):
        self.train_size = train_size
        self.valid_size = valid_size

    def save(self, dataset_path):
        file_name = dataset_path + "hyperparameters.pickle"
        with open(file_name, 'wb') as outp:
            pickle.dump(self,
                        outp,
                        pickle.HIGHEST_PROTOCOL)

DatasetHPsV2 = DatasetHPs

def loadDatasetHPs(dataset_path: str) -> DatasetHPs:
    file_name = dataset_path + "hyperparameters.pickle"
    with open(file_name, "rb") as inp:
        return pickle.load(inp)


class Dataset(object):

    class State(Enum):
        NOT_CREATED = 1
        SAMPLES_AVAIL = 2
        NO_SAMPLES = 3

    def __init__(self,
                 project_path: str,
                 data_collector: DataCollectorv2,
                 hps: DatasetHPs = None):
        """
        Each dataset you build depends on some project.
        To build the dataset you have to pass the project path as an argument. The dataset will be built upon the TFRecord
        protocol. To create the dataset you will receive a list of lineages, compression factor (n), fragment length
        (which is equivalent to d_model) and a kmer_size. Those parameters will decide the Genome encoding.
        """

        # Save private members
        self.project_path = project_path
        self.dataset_path = project_path + "Dataset/"
        self.data_collector = data_collector
        self.hps = hps

        self.min_orig_accs_per_valid = 10
        self.min_orig_accs_per_train = 16

        # Check if dataset already exist in the project, if so, print a message and return.
        state = self.getDatasetState()
        if state == Dataset.State.NO_SAMPLES:
            self.hps = loadDatasetHPs(dataset_path=self.dataset_path)
            self.hps.updateSizes(0, 0)
            return
        elif state == Dataset.State.SAMPLES_AVAIL:
            self.hps = loadDatasetHPs(dataset_path=self.dataset_path)
            return

        assert hps != None, \
            "Must provide Hyper parameters for building the dataset via hps."

        # Build lineage label map
        self._buildLineageLabelMap(hps.lineages)

        # Iterate over the lineages, and for each lineage create a TFRecord file.
        train_set_size = 0
        valid_set_size = 0
        self._checkLinAccsNumValidity()
        for lineage in self.hps.lineages:
            print("Started serializing {}".format(lineage))
            lin_train, lin_valid = self._serializeLineage(lineage)
            train_set_size += lin_train
            valid_set_size += lin_valid
        self.hps.updateSizes(train_size=train_set_size,
                             valid_size=valid_set_size)
        self.hps.save(self.dataset_path)

    def getTrainSet(self,
                    batch_size: int,
                    epochs: int,
                    shuffle_buffer_size: int = 1024) -> tf.data.Dataset:
        """
        just returns the train set
        """
        if self.getDatasetState() != Dataset.State.SAMPLES_AVAIL:
            return None
        filepath_dataset = tf.data.Dataset.list_files(self._getTrainPath() + "*")
        return filepath_dataset.interleave(
            map_func=lambda filepath: tf.data.TFRecordDataset(filepath),
            num_parallel_calls=tf.data.AUTOTUNE,
            cycle_length=len(self.hps.lineages)).\
            map(map_func=lambda example_proto: self._deserializeGenomeTensor(example_proto),
                num_parallel_calls=tf.data.AUTOTUNE).\
            repeat(epochs).\
            shuffle(shuffle_buffer_size).\
            batch(batch_size).\
            prefetch(tf.data.AUTOTUNE)

    def getValidSet(self,
                    batch_size: int):
        """
        Returns the validation set
        """
        if self.getDatasetState() != Dataset.State.SAMPLES_AVAIL:
            return None
        filepath_dataset = tf.data.Dataset.list_files(self._getValidPath() + "*")
        return filepath_dataset.interleave(
            map_func=lambda filepath: tf.data.TFRecordDataset(filepath),
            num_parallel_calls=tf.data.AUTOTUNE).\
            map(map_func=lambda example_proto: self._deserializeGenomeTensor(example_proto),
                 num_parallel_calls=tf.data.AUTOTUNE).\
            batch(batch_size).\
            prefetch(tf.data.AUTOTUNE)

    def getLineages(self):
        return self.hps.lineages

    def getTrainSetSampleCount(self):
        return self.hps.train_size

    def getValidSetSampleCount(self):
        return self.hps.valid_size

    def getDatasetState(self):
        """
        Returns the state of the dataset, the state of the dataset can be
        one of those following three states:
        1. Not created yet.
        2. created and datasets available.
        3. created but datasets not available.
        """
        if os.path.isdir(self.dataset_path):
            self.hps = loadDatasetHPs(dataset_path=self.dataset_path)
            if self._doesSamplesExist():
                return Dataset.State.SAMPLES_AVAIL
            elif os.path.exists(self.dataset_path + "/hyperparameters.pickle"):
                return Dataset.State.NO_SAMPLES
            else:
                return Dataset.State.NOT_CREATED
        else:
            return Dataset.State.NOT_CREATED

    #######################################
    #######Private member functions########
    #######################################
    def _checkLinAccsNumValidity(self):
        for lin in self.hps.lineages:
            accs_num = len(self.data_collector.getLocalAccessions(lineage=lin))
            min_accs_num = self.min_orig_accs_per_valid + self.min_orig_accs_per_train
            assert accs_num >= min_accs_num, \
                "accessions number for the lineage {} is {}, but should be at least {}.".format(lin,
                                                                                                accs_num,
                                                                                                min_accs_num)

    def _doesSamplesExist(self):
        return os.path.exists(self.dataset_path + "Train/")

    def _getTrainIdxSplit(self,
                         orig_accs_num: int) -> int:
        max_train_ex_num = orig_accs_num - self.min_orig_accs_per_valid
        return min(int((1 - self.hps.validation_split)*orig_accs_num), max_train_ex_num)

    def _getReplicasPerAcc(self,
                           orig_train_ex_num: int):
        return ceil(self.hps.max_accs_per_lineage / orig_train_ex_num)

    def _serializeLineage(self,
                         lineage: Lineage) -> Tuple[int, int]:
        # Get local accessions set of the lineage
        accessions_set = self.data_collector.getLocalAccessions(lineage=lineage)

        # Convert it into a list, shuffle elements, and take maximum of max_accs_per_lineage.
        accessions = list(accessions_set)
        random.shuffle(accessions)
        accessions = accessions[:self.hps.max_accs_per_lineage]

        # split the accessions list to train_accs, valid_accs
        train_index_split = self._getTrainIdxSplit(len(accessions))
        train_accs = accessions[:train_index_split]

        replicas_per_acc = self._getReplicasPerAcc(len(train_accs))
        random.shuffle(train_accs)
        valid_accs = accessions[train_index_split:]
        random.shuffle(valid_accs)

        # Create the project path if does not yet exist
        os.makedirs(self._getTrainPath(), exist_ok=True)
        os.makedirs(self._getValidPath(), exist_ok=True)

        # Serialize the accessions belonging to the lineage
        train_path = self._getTrainPath(lineage)
        valid_path = self._getValidPath(lineage)
        train_ex = self._serializeAccessionsList(lineage, train_accs, train_path, replicas_per_acc=replicas_per_acc)
        valid_ex = self._serializeAccessionsList(lineage, valid_accs, valid_path)
        return train_ex, valid_ex

    def _buildLineageLabelMap(self,
                              lineages: List[Lineage]) -> LineageLabelMap:
        lineage_label_map: LineageLabelMap = {}
        num_of_lineages = len(lineages)
        for i in range(num_of_lineages):
            lineage_label_map.update({lineages[i]: tf.squeeze(tf.one_hot(indices=[i],
                                                                              depth=num_of_lineages,
                                                                              axis=0,
                                                                              dtype=tf.int8))})
        self.lineage_label_map = lineage_label_map

    def _serializeAccessionsList(self,
                                 lineage: Lineage,
                                 accs: List[Accession],
                                 tfrecordfile_path: str,
                                 replicas_per_acc: int = 1) -> int:
        serialized_tensors_num = 0
        with tf.io.TFRecordWriter(tfrecordfile_path) as f:
            # Iterate over the train accessions
            for acc in accs:
                # Create a genome tensor for each accession.
                genome = Genome(accession_id=acc,
                                data_collector=self.data_collector)
                tensors = genome.getFeatureTensor(kmer_size=self.hps.kmer_size,
                                                 fragment_size=self.hps.frag_len,
                                                 n=self.hps.n,
                                                 replicas_per_acc=replicas_per_acc)
                label = self.lineage_label_map[lineage]

                for tensor in tensors:
                    serialized_tensors_num += 1
                    # Serialize the Genome tensor to create a suitable tf.train.Example protobuf object.
                    serialized_tensor = self._serializeGenomeTensor(tensor,
                                                                    label)

                    # Dump serialized tensor to the file.
                    f.write(serialized_tensor.SerializeToString())

        return serialized_tensors_num

    def _getTrainPath(self,
                     lineage=None):
        if lineage is None:
            return self.dataset_path + "Train/"
        return self.dataset_path + "Train/" + lineage + ".tfrecord"

    def _getValidPath(self,
                      lineage=None):
        if lineage is None:
            return self.dataset_path + "Valid/"
        return self.dataset_path + "Valid/" + lineage + ".tfrecord"

    def _serializeGenomeTensor(self,
                               tensor: tf.Tensor,
                               label: tf.Tensor) -> tf.train.Example:

        # Serialize the tensor
        serialized_tensor = tf.io.serialize_tensor(tensor)
        serialized_label = tf.io.serialize_tensor(label)

        # Store the data in a tf.train.Feature (which is a protobuf object)
        feature_of_bytes_tensor = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[serialized_tensor.numpy()])
        )
        feature_of_bytes_label = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[serialized_label.numpy()])
        )

        # Put the tf.train.Feature message into a tf.train.Example (which is a protobuf object that will be written into the file)
        features_for_example = {
            'tensor': feature_of_bytes_tensor,
            'label': feature_of_bytes_label
        }
        example_proto = tf.train.Example(
            features=tf.train.Features(feature=features_for_example)
        )
        return example_proto

    feature_description = {
            'tensor': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.string)
    }

    def _deserializeGenomeTensor(self,
                                 example_proto):
        feature_map = tf.io.parse_single_example(example_proto, self.feature_description)
        tensor_shape = [self.hps.n, self.hps.frag_len, base_count]
        label_shape = [len(self.hps.lineages)]
        tensor = tf.ensure_shape(tf.io.parse_tensor(feature_map['tensor'],
                                                    out_type=tf.int8),
                                 tensor_shape)
        tensor = tf.cast(tensor, tf.float32)
        label = tf.ensure_shape(tf.io.parse_tensor(feature_map['label'],
                                                   out_type=tf.int8),
                                label_shape)
        label = tf.cast(label, tf.float32)
        return tensor, label
