import numpy as np
import math
import Genome_pb2

class Genome(object):
    """class that represent a Genome int the system

    This class provides an API for creating vectorized fragments out of a genome sequence.
    It gets a filename of the FASTA file in which the genome is written.
    And provides functionalities to store and vectorize the genome sequence.
    Also, it provides functionalities to serialize and deserialize the encoded genomes.
    The encoding of a genome occurs as follows:
    Genome:         ACCTAGT CCTATGA AANAA
    fragment size:  7
    vectorized:     (1, 2, 2, 4, 1, 3, 4)
                    (2, 2, 4, 1, 4, 3, 4)
                    (1, 1, 0, 1, 1, 0, 0)

    All of this assumes that the encoding is as follows:
    A: 1    C: 2    G: 3    T: 4    N: 0    padding: 0
    """
    def __init__(self,
                 filename: str,
                 fragment_size: int) -> None:
        """

        :param filename: the file name of the genome FASTA file. if does not exist returns an error.
        :param fragment_size: the basic size of fragments that we vectorize.

        """
        ###################
        ##### Members #####
        ###################
        self.filename: str = filename
        self.fragment_size: int = fragment_size
        self.name: str = ''
        self.genome_seq: str = ''
        self.genome_len: int = 0
        self.genome_tensor: np.ndarray = None
        ###################
        ###################
        ###################

        # Check if file exists, if not please return an exception
        # Check size of file, if too big, please return an exception
        # TODO

        # Open file in "read only" mode
        file = open(filename, "r")

        # Read lines from reference
        first_line: bool = True
        for line in file:
            if first_line is True:
                self.name += line[:-1]
                first_line = False
            else:
                self.genome_seq += line[:-1]

        # Configure genome length
        self.genome_len = len(self.genome_seq)

    def printGenome(self):
        print("The name is: {}".format(self.name))
        print("The length is: {}".format(self.genome_len))
        print("The fragment size is: {}".format(self.fragment_size))
        print("The number of fragments is: {}".format(math.ceil(self.genome_len / self.fragment_size)))

    def printEncodedGenome(self):
        print(self.genome_tensor.shape)
        print(self.genome_tensor)

    def encodeGenome(self) -> np.ndarray:
        nm_of_fragments: int = math.ceil(self.genome_len / self.fragment_size)
        self.genome_tensor = np.ndarray(shape=(nm_of_fragments,
                                               self.fragment_size))
        for frag_idx in range(nm_of_fragments):
            for base_idx in range(self.fragment_size):
                self.genome_tensor[frag_idx][base_idx] =\
                    self._encodeBase(frag_idx*self.fragment_size + base_idx)

    def _encodeBase(self,
                    idx: int):
        if idx < self.genome_len:
            if self.genome_seq[idx] == 'A':
                return 1
            elif self.genome_seq[idx] == 'C':
                return 2
            elif self.genome_seq[idx] == 'G':
                return 3
            elif self.genome_seq[idx] == 'T':
                return 4
            elif self.genome_seq[idx] == 'N':
                return 0
            # TODO: add exception if not of the above
        else:
            return 0

    def serialize(self):
        """

        """
        genome = Genome_pb2.Genome()

"""
    def deserialize(self,
                    filename: str):

        """


genome = Genome('../data/LR991698.2.fasta', 100)
genome.printGenome()
genome.encodeGenome()
genome.printEncodedGenome()
genome.serialize()

"""
TODO:
1. Serialize with protobufs
2. Add exception support
"""




