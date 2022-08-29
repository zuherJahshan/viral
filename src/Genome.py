# Builtin packages
import numpy as np
from typing import Callable
from math import ceil, floor
import hashlib
from Types import *
import os
from GenomeDuplicate import GenomeDuplicate

# Project packages
from DataCollector import DataCollectorv2

Hasher = Callable[[np.ndarray], np.uint64]

base_count = 4

## Starting data augmentation branch

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
        if n >= len(genome_vec) - kmer_size + 1:
            print("the compression factor, n must be smaller or equal to the genome length = {}" \
                  .format(len(genome_vec) - kmer_size + 1))
            raise ValueError

        self.hasher: Hasher = hasher
        self.__makeTensor(genome_vec)

    def __makeTensor(self,
                     genome_vec: np.ndarray) -> np.ndarray:
        """
        The tensor is made according to the minhash scheme.
        """
        # Finding the encoding of the genome vector
        def index_key(elem):
            return elem[0]

        def hasher_key(elem):
            return elem[1]


        # Generate the list of all kmers signatures
        sigs: List = []
        kmer_in_genome: int = len(genome_vec) - self.kmer_size + 1

        # Iterate over kmers in genome
        for i in range(kmer_in_genome):
            # hash and insert to the list O(hash operation)
            sigs.append((i, self.hasher(genome_vec[i: i+self.kmer_size])))
        # sort according to hashed value - O(genome_length*log(genome_length)) where n is the sequence length, in covid case 30000*log30000
        sigs.sort(key=hasher_key)

        # Take the smallest n kmers.
        sigs = sigs[:self.n]
        sigs.sort(key=index_key)    # sort according to index (kmer relative position in the genome

        # Generate the tensor
        self.tensor = np.zeros(shape=(self.n, self.fragment_size, base_count), dtype=np.int8)
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

# Global variables
A_vec: np.ndarray = np.array([1, 0, 0, 0], dtype=np.int8)
C_vec: np.ndarray = np.array([0, 1, 0, 0], dtype=np.int8)
G_vec: np.ndarray = np.array([0, 0, 1, 0], dtype=np.int8)
T_vec: np.ndarray = np.array([0, 0, 0, 1], dtype=np.int8)
N_vec: np.ndarray = np.array([0, 0, 0, 0], dtype=np.int8)


class Genome(object):
    """
    This Class represents the genome in all necessary forms:
    1. the character based sequence - saved in a file
    2. the vectorized sequence - saved in a file
    3. the feature tensor of the sequence - encoded on the fly

    The purpose of this class is to return the different representations of the genome
    """
    def __init__(self,
                 accession_id: str = None,
                 data_collector = None,
                 accession_path: str = None):
        # Initializing DataCollector
        assert accession_path != None or accession_id != None,\
            "Must specify either the acc_path or the accession_id"
        if accession_path is None:
            if data_collector is None:
                self.data_collector: DataCollectorv2 = DataCollectorv2()
            else:
                self.data_collector = data_collector

            self.accession_id: str = accession_id
            self.seq_filepath: str = self.data_collector.getAccPath(accession_id)
            assert self.seq_filepath != "",\
                "Accession {} was not found locally.".format(accession_id)

        else:
            # Check existence of the path
            assert os.path.exists(accession_path),\
                "File {} was not found.".format(accession_path)
            self.accession_id = accession_path.split("/")[-1].split(".")[0]
            self.seq_filepath = accession_path

    def getAccId(self):
        return self.accession_id

    def getSeq(self) -> Sequence:
        # Open file in "read only" mode
        file = open(self.seq_filepath, "r")

        # Read lines from fasta file containing sequence
        seq = ""
        first_line: bool = True
        # Iterate on the lines
        for line in file:
            # Ignore first line (only holds metadata)
            if first_line is True:
                self.accession_id = line.split("|")[1]
                first_line = False
            else:
                # Add the bases of the line to the sequence except of the last character which is always a ' '.
                seq += line[:-1]

        # Close the file
        file.close()
        # Sequence can not contain 0 bases
        assert len(seq) > 0,\
            "The length of the accession {} sequence is zero".format(self.accession_id)
        return seq

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

    def __vectorizeSeq(self,
                       seq: str):
        # Vectorize the sequence O(sequence_length)
        vec = np.zeros(shape=(len(seq), base_count))
        for i in range(len(seq)):
            vec[i, :] = self.__encodeBase(seq[i])
        return vec

    def getFeatureTensor(self,
                         kmer_size: int,
                         fragment_size: int,
                         n: int,
                         replica: bool = False,
                         hasher: Hasher = hashlib_hasher) -> List[np.ndarray]:
        """
        Creates a Genome tensor class and returns the tensor created by the class
        """
        # Get original sequence or duplicate sequence
        if not replica:
            seq = self.getSeq()
        else:
            genome_duplicate = GenomeDuplicate(genome=self.getSeq())
            seq = genome_duplicate.getDuplicate()

        # vectorize sequence
        vec = self.__vectorizeSeq(seq)

        # Create genome tensor.
        genome_tensor = GenomeTensor(genome_vec=vec,
                                     kmer_size=kmer_size,
                                     fragment_size=fragment_size,
                                     n=n,
                                     hasher=hasher)

        return genome_tensor.getTensor()
