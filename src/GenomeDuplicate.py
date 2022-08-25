import numpy as np
import random

base_count = 4

base_array = ['A', 'C', 'G', 'T']

base_idx_dict = {
    'A': 0,
    'C': 1,
    'G': 2,
    'T': 3
}

class GenomeDuplicate(object):
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
        if self.replacement[position]:
            idx = (base_idx_dict[orig_base] + self.base_replacement[position]) % base_count
            return base_array[idx]
        else:
            return orig_base
