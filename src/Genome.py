# Builtin packages
import numpy as np
from typing import Callable
from math import ceil, floor
import hashlib
from xxhash import xxh3_64_intdigest
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

def xxhash_hasher(tensor: np.ndarray):
    return xxh3_64_intdigest(tensor.tobytes())

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

    def _getSigs(self,
                genome_vec,
                ):
        kmer_in_genome: int = len(genome_vec) - self.kmer_size + 1
        for i in range(kmer_in_genome):
            yield i, self.hasher(genome_vec[i: i + self.kmer_size])


    def _getKmerVal(self,
                    kmer_vec,
                    old_val = None):
        if old_val == None:
            return np.sum(kmer_vec * base, dtype=np.uint64)
        else:
            return int(old_val / 8) + int(kmer_vec[-1, :] @ base[-1, :])

    def _getSigsAcc(self,
                    genome_vec):
        kmer_in_genome: int = len(genome_vec) - self.kmer_size + 1
        kmer_val = None
        for i in range(kmer_in_genome):
            kmer_val = self._getKmerVal(kmer_vec=genome_vec[i: i + self.kmer_size],
                                        old_val=kmer_val)
            yield i, kmer_val

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
        sigs: List = list(self._getSigs(genome_vec=genome_vec))
        sigs.sort(key=hasher_key)

        # Take the smallest n kmers.
        sigs = sigs[:self.n]
        sigs.sort(key=index_key)    # sort according to index (kmer relative position in the genome

        # Generate the tensor
        genome_len = genome_vec.shape[0]
        right_ext: int = ceil((self.fragment_size - self.kmer_size) / 2)
        left_ext: int = floor((self.fragment_size - self.kmer_size) / 2)
        new_genome_vec = np.zeros(shape=(genome_len + right_ext + left_ext, base_count), dtype=np.int8)
        new_genome_vec[left_ext: left_ext + genome_len, :] = genome_vec
        start_indices = np.array([sigs[i][0] for i in range(self.n)])
        I = tf.stack([start_indices + i for i in range(self.fragment_size)], -1)
        self.tensor = tf.gather(new_genome_vec, I)

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

    def __vectorizeSeq(self,
                       seq: str):
        # Vectorize the sequence O(sequence_length)
        vec = np.zeros(shape=(len(seq), base_count))
        vec2 = np.zeros(shape=(len(seq), base_count))

        # Alternative
        seq_narray = np.array([*seq])

        vec[seq_narray == 'A', :] = A_vec
        vec[seq_narray == 'C', :] = C_vec
        vec[seq_narray == 'G', :] = G_vec
        vec[seq_narray == 'T', :] = T_vec

        return vec

    def getFeatureTensor(self,
                         kmer_size: int,
                         fragment_size: int,
                         n: int,
                         replica: bool = False,
                         hasher: Hasher = xxhash_hasher) -> List[np.ndarray]:
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
        tensor = genome_tensor.getTensor()

        tensor = np.random.rand(n, fragment_size, base_count)

        return tensor
