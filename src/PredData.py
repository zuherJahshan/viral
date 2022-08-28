import os

from Types import *
from Genome import Genome, base_count
from Dataset import DatasetHPs
from math import ceil, floor

import threading
import csv

class PredData(object):
    def __init__(self,
                 hps: DatasetHPs,
                 path_to_fasta_dir: str,
                 num_parallel_calls: int,
                 record_res: bool = True):
        """
        1. First you choose the number of tfrecord files
        2. then you build a map from accession -> tfrecord file
        and a map from            tfrecord file -> accessions
        3. then you serialize the list of accessions for each tfrecord file in parallel
        4. then you create the dataset using interleave
        """
        assert os.path.exists(path_to_fasta_dir),\
            "The path you specified does not exist"
        self.path_to_fasta_dir = path_to_fasta_dir
        self.tfrecord_files_path = path_to_fasta_dir + "/" + "tfrecord/"
        self.record_res = record_res
        self.results_path = path_to_fasta_dir + "/" + "results.csv"
        self.hps = hps

        if not os.path.exists(self.tfrecord_files_path):
            # Create path
            os.makedirs(self.tfrecord_files_path, exist_ok=True)  # maybe should do exist not okay...
            self._createTfrecordFiles(path_to_fasta_dir=path_to_fasta_dir,
                                      num_parallel_calls=num_parallel_calls)


    def getData(self,
                batch_size: int = 1):
        # Create interleave dataset with cycle length of 1
        filepath_dataset = tf.data.Dataset.list_files(self.tfrecord_files_path + "*")
        return filepath_dataset.interleave(
            map_func=lambda filepath: tf.data.TFRecordDataset(filepath),
            num_parallel_calls=tf.data.AUTOTUNE).\
            map(map_func=lambda example_proto: self._deserializeGenomeTensor(example_proto),
                 num_parallel_calls=tf.data.AUTOTUNE).\
            batch(batch_size).\
            prefetch(tf.data.AUTOTUNE)

    def recordRes(self,
                  acc_list: List[Accession],
                  results):
        if not self.record_res:
            return
        # If file does not exist, create it and insert the head
        if not os.path.exists(self.results_path):
            self.csv_file = open(self.results_path, 'w', encoding='UTF-8', newline='')
            self.csv_writer = csv.writer(self.csv_file)

            # create the header and write it
            k = results.values.shape[-1]
            header = ["accession"]
            for i in range(1, k+1):
                header += ["top-{} prediction".format(i), "top-{} probability".format(i)]
            self.csv_writer.writerow(header)

        # Extract the data into lists
        batch_size = results.values.shape[0]
        k = results.values.shape[-1]
        indices = results.indices.numpy().tolist()
        probs = results.values.numpy().tolist()
        for i in range(batch_size):
            data = [acc_list[i]]
            for j in range(k):
                lineage = self.hps.lineages[indices[i][j]]
                prob = probs[i][j]
                data += [lineage, prob]
            self.csv_writer.writerow(data)

    #######################################
    #######Private member functions########
    #######################################
    def _createTfrecordFiles(self,
                                path_to_fasta_dir: str,
                                num_parallel_calls: int = 1):

        # List files from fasta dir
        fasta_files = []
        for file in os.listdir(path_to_fasta_dir):
            # check only text files
            if file.endswith('.fasta'):
                fasta_files.append(path_to_fasta_dir + "/" + file)

        # iterate over the number of tfrecord files you want to create and fill the maps.
        max_accs_in_tfrecord_file = self._getMaxGenomesInTfrecordFile(num_parallel_calls=num_parallel_calls,
                                                                      num_fasta_files=len(fasta_files))
        idx = 0
        tfrecord_file = 1
        threads = []
        while idx < len(fasta_files):
            new_idx = min(idx + max_accs_in_tfrecord_file, len(fasta_files))
            thread = threading.Thread(
                target=self._serializeAccessionsList,
                args=(fasta_files[idx: new_idx],
                      self.tfrecord_files_path + str(tfrecord_file) + ".tfrecord"
                      )
            )
            threads.append(thread)
            thread.start()
            idx = new_idx
            tfrecord_file += 1

        for thread in threads:
            thread.join()


    def _getTfrecordFilesNum(self,
                            num_parallel_calls: int,
                            num_fasta_files: int) -> int:
        return min(num_parallel_calls, num_fasta_files)

    def _getMaxGenomesInTfrecordFile(self,
                                     num_parallel_calls: int,
                                     num_fasta_files: int) -> int:
        return ceil(num_fasta_files / num_parallel_calls)

    def _serializeAccessionsList(self,
                                 accs_path: List[str],
                                 tfrecordfile_path: str):
        with tf.io.TFRecordWriter(tfrecordfile_path) as f:
            # Iterate over the train accessions
            for acc_path in accs_path:
                # Create a genome tensor for each accession.
                genome = Genome(accession_path=acc_path)
                tensor = genome.getFeatureTensor(kmer_size=self.hps.kmer_size,
                                                 fragment_size=self.hps.frag_len,
                                                 n=self.hps.n)[0]

                # Serialize the Genome tensor to create a suitable tf.train.Example protobuf object.
                serialized_tensor = self._serializeGenomeTensor(tensor,
                                                                genome.getAccId())

                # Dump serialized tensor to the file.
                f.write(serialized_tensor.SerializeToString())


    def _serializeGenomeTensor(self,
                               tensor: tf.Tensor,
                               acc: Accession) -> tf.train.Example:

        # Serialize the tensor
        serialized_tensor = tf.io.serialize_tensor(tensor)

        # Serialize the accession
        acc_tensor = tf.constant(acc)
        serialized_acc = tf.io.serialize_tensor(acc_tensor)

        # Store the data in a tf.train.Feature (which is a protobuf object)
        feature_of_bytes_tensor = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[serialized_tensor.numpy()])
        )
        feature_of_bytes_acc = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[serialized_acc.numpy()])
        )

        # Put the tf.train.Feature message into a tf.train.Example (which is a protobuf object that will be written into the file)
        features_for_example = {
            'tensor': feature_of_bytes_tensor,
            'accession': feature_of_bytes_acc
        }
        example_proto = tf.train.Example(
            features=tf.train.Features(feature=features_for_example)
        )
        return example_proto

    feature_description = {
            'tensor': tf.io.FixedLenFeature([], tf.string),
            'accession': tf.io.FixedLenFeature([], tf.string)
    }

    def _deserializeGenomeTensor(self,
                                 example_proto):
        feature_map = tf.io.parse_single_example(example_proto, self.feature_description)
        tensor_shape = [self.hps.n, self.hps.frag_len, base_count]
        acc_shape = []
        tensor = tf.ensure_shape(tf.io.parse_tensor(feature_map['tensor'],
                                                    out_type=tf.int8),
                                 tensor_shape)
        tensor = tf.cast(tensor, tf.float32)
        acc = tf.ensure_shape(tf.io.parse_tensor(feature_map['accession'],
                                                 out_type=tf.string),
                              acc_shape)
        return tensor, acc
