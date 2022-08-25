import numpy as np
import random

base_count = 4

base_array = ['A', 'C', 'G', 'T', 'N']

base_idx_dict = {
    'A': 0,
    'C': 1,
    'G': 2,
    'T': 3,
    'N': 4
}

class GenomeDuplicate(object):
    """
    The class will be named GenomeDuplicate, it will contain two main functionalities:
    1. constructor - which will receive the genome length and an error profile
    2. getNoisy - which will receive a base (one char representing a nucleic acid) and will return a seq of the noisy represntation according to the error profile.

    The error profile:

    imagine a reading head standing on top of a base:
          |
    ACCTATCAGTT

    the error profile has the following tree structure:
    1. successful read of the head
        1.1 read the right value (C)
            1.1.1 insertion (go again deciding 1.1.1 or 1.1.2)
            1.1.2 no insertion (halt)
        1.2 read a substitution (A, G, T) (with equal probabilities)
            1.2.1 insertion (go again deciding 1.2.1 or 1.2.2)
            1.2.2 no insertion (halt)
    2. failure read of the head (delete)
        2.1 insertion (go again deciding 2.1 or 2.2)
        2.2 no insertion (halt)
    """
    def __init__(self,
                 genome: str,
                 deletion_rate: float = 0.001,
                 insertion_rate: float = 0.001,
                 replacement_rate: float = 0.001):
        genome_length = len(genome)
        self.genome = genome
        self.deletion = np.random.uniform(size=genome_length) < deletion_rate
        self.replacement = np.random.uniform(size=genome_length) < replacement_rate
        self.base_replacement = np.random.randint(low=1, high=base_count, size=genome_length)
        self.insertion_rate = insertion_rate

    def getDuplicate(self):
        new_genome = ''
        for pos in range(len(self.genome)):
            new_genome += self._getNoisy(position=pos)
        return new_genome

    def _getNoisy(self,
                 position: int) -> str:
        seq = ''
        if not self.deletion[position]:
            seq += self._replacementNoise(position=position)
        return self._insertionNoise(seq)

    def _insertionNoise(self,
                        seq):
        while random.random() < self.insertion_rate:
            idx = np.random.randint(low=0, high=base_count, size=1)[0]
            seq += base_array[idx]
        return seq

    def _replacementNoise(self,
                          position):
        orig_base = self.genome[position]
        if not orig_base in base_idx_dict:
            orig_base = 'N'
        if self.replacement[position]:
            idx = (base_idx_dict[orig_base] + self.base_replacement[position]) % base_count
            return base_array[idx]
        else:
            return orig_base
