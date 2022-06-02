# Builtin packages
from os import path
from pathlib import Path
import numpy as np
from typing import Callable, List
from math import ceil, floor
import hashlib

# For debugging
import tensorflow as tf

# Third party packages
from numproto import ndarray_to_proto, proto_to_ndarray, numproto

# Project packages
from DataCollector import DataCollector

Hasher = Callable[[np.ndarray], np.uint64]

base_count = 4


class FileNotFound(Exception):
    pass

class AccessionNotFoundLocally(Exception):
    pass

def hashlib_hasher(tensor: np.ndarray):
    byte_view = tensor.view(np.uint8)
    hashed_str = hashlib.sha1(byte_view).hexdigest()
    return int(hashed_str,
               base=16)

class GenomeTensor(object):
    def __init__(self,
                 genome_vec: np.ndarray,
                 kmer_size: int,
                 fragment_size: int,
                 n: int,
                 hasher: Hasher):
        # Members
        self.kmer_size: int = kmer_size
        self.fragment_size = fragment_size
        self.n: int = n
        assert n < len(genome_vec) - kmer_size + 1, \
            "the compression factor, n must be smaller or equal to the genome length = {}" \
            .format(len(genome_vec) - kmer_size + 1)
        self.hasher: Hasher = hasher
        self.__makeTensor(genome_vec)   # Memory leak is in here!!!

    def __makeTensor(self,
                     genome_vec: np.ndarray) -> np.ndarray:

        # Finding the encoding of the genome vector
        def index_key(elem):
            return elem[0]

        def hasher_key(elem):
            return elem[1]


        sigs: List = []
        kmer_in_genome: int = len(genome_vec) - self.kmer_size + 1
        for i in range(kmer_in_genome):
            sigs.append((i, self.hasher(genome_vec[i: i+self.kmer_size])))  # hash and insert to the list
        sigs.sort(key=hasher_key)   # sort according to hashed value
        sigs = sigs[:self.n]
        sigs.sort(key=index_key)    # sort according to index (kmer relative position in the genome
        # print(sigs) - DEBUG

        # Making the tensor
        self.tensor = np.zeros(shape=(self.n, self.fragment_size, base_count))
        for i in range(self.n):
            # Deciding boundaries
            genome_len = genome_vec.shape[0]
            tensor_start_idx = 0
            tensor_end_idx = self.fragment_size
            right_ext: int = ceil((self.fragment_size - self.kmer_size) / 2)
            left_ext: int = floor((self.fragment_size - self.kmer_size) / 2)
            start_idx = sigs[i][0] - left_ext
            end_idx = sigs[i][0] + self.kmer_size + right_ext
            if start_idx < 0:
                tensor_start_idx = -start_idx
                start_idx = 0
            if end_idx > genome_len:
                tensor_end_idx = self.fragment_size - (end_idx - genome_len)
                end_idx = genome_len

            # Building feature vector
            self.tensor[i, tensor_start_idx: tensor_end_idx, :] = genome_vec[start_idx: end_idx, :]

        return self.tensor

    def getTensor(self) -> np.ndarray:
        return self.tensor

    def sameEncoding(self,
                     kmer_size: int,
                     fragment_size: int,
                     n: int,
                     hasher: Hasher) -> bool:
        return kmer_size == self.kmer_size and \
               fragment_size == self.fragment_size and \
               n == self.n and \
               hasher == self.hasher


# Global variables
A_vec: np.ndarray = np.array([1, 0, 0, 0])
C_vec: np.ndarray = np.array([0, 1, 0, 0])
G_vec: np.ndarray = np.array([0, 0, 1, 0])
T_vec: np.ndarray = np.array([0, 0, 0, 1])
N_vec: np.ndarray = np.array([0, 0, 0, 0])


class Genome(object):
    """
    This Class represents the genome in all necessary forms:
    1. the character based sequence - saved in a file
    2. the vectorized sequence - saved in a file
    3. the feature tensor of the sequence - encoded on the fly

    The purpose of this class is to return the different representations of the genome
    """
    def __init__(self,
                 accession_id: str,
                 data_collector = None):
        # Initializing DataCollector
        if data_collector is None:
            self.data_collector: DataCollector = DataCollector("../accessions.tsv")
        else:
            self.data_collector = data_collector

        # Checks if the accession exists, if not, raises an appropriate error.
        if not self.data_collector.exists(accession_id):
           raise AccessionNotFoundLocally()

        self.accession_id: str = accession_id
        self.seq_filepath: str = self.data_collector.getAccPath(accession_id)
        self.vec_filepath: str = ""
        self.vec: np.ndarray = None
        self.tensor: GenomeTensor = None

    def getSeq(self):
        # Open file in "read only" mode
        file = open(self.seq_filepath, "r")

        # Read lines from fasta file containing sequence
        seq = ""
        first_line: bool = True
        for line in file:
            if first_line is True:
                first_line = False
            else:
                seq += line[:-1]

        file.close()
        return seq

    def getVectorizedSeq(self):
        """
        Returns the vectorized version
        """
        self.__vectorizeSeq()
        return self.vec

    def __encodeBase(self,
                     base: str):
        if base == 'A':
            return A_vec
        elif base == 'G':
            return G_vec
        elif base == 'T':
            return T_vec
        elif base == 'C':
            return C_vec
        else:
            return N_vec

    def __vectorizeSeq(self):
        """
        if the vec is not None, then it is already vectorized, no need to do nothing
        If it is None, then one of the two options can occur:
            1. The vectorized file has not been loaded yet.
            2. A vectorized file representation of the file does not exist yet.
        If it is option 1, then it will load the vector from the file
        If it is option 2, then it will create the vector, serialize it, and wills save it into the appropriate file.

        In all cases, it is guaranteed,
        that the vectorized sequence will be found in the self.vec member after the call.
        """

        # Check if already sequenced
        if self.vec is not None:
            return

        # create path if does not exist
        vec_files_loc = "../data/vectorized/"
        Path(vec_files_loc).mkdir(parents=True,
                                  exist_ok=True)

        # Check if serialized file already exists
        self.vec_filepath = vec_files_loc + self.accession_id + ".bin"
        if path.exists(self.vec_filepath):
            vec_file = open(self.vec_filepath, 'rb')

            # Desirialize
            serialized_nda = numproto.NDArray()
            serialized_nda.ParseFromString(vec_file.read())
            self.vec = proto_to_ndarray(serialized_nda)
            vec_file.close()
            return

        # if not, vectorize
        seq = self.getSeq()
        self.vec = np.zeros(shape=(len(seq), base_count))
        for i in range(len(seq)):
            self.vec[i, :] = self.__encodeBase(seq[i])

        # And serialize
        vec_file = open("../data/vectorized/" + self.accession_id + ".bin", "wb")
        vec_file.write(ndarray_to_proto(self.vec).SerializeToString())
        vec_file.close()


    def getFeatureTensor(self,
                         kmer_size: int,
                         fragment_size: int,
                         n: int,
                         hasher: Hasher = hashlib_hasher) -> np.ndarray:
        """
        returns a created Genome tensor
        """
        # Check if exists
        if self.tensor is not None:
            # Check if the parameters for creating the tensor are the same, those are hyperparameters of the model
            if self.tensor.sameEncoding(kmer_size=kmer_size,
                                        fragment_size=fragment_size,
                                        n=n,
                                        hasher=hasher):
                return self.tensor.getTensor()

        # If does not exist, or the hyperparameters hava changed. Create it.
        self.__vectorizeSeq()
        self.tensor = GenomeTensor(genome_vec=self.vec,
                                   kmer_size=kmer_size,
                                   fragment_size=fragment_size,
                                   n=n,
                                   hasher=hasher)
        return self.tensor.getTensor()
