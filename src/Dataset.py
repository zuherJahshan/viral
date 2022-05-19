from Genome import Genome
import tensorflow as tf
import numpy as np
from DataCollector import DataCollector
from time import time
from os import listdir

data_collector: DataCollector = DataCollector("../accessions.tsv")

def create_generator():
    """
    1. Define a hand-crafted list of file Accessions
    2. Iterate over that list and build Genomes for each accession ID
    3. yield genome.getFeaturedTensor()
    """
    accessions_list = ["OD976731",
                       "OM842265",
                       "OU187031",
                       "OV567784",
                       "OV830007",
                       "OW216182"
                       ]

    for acc in accessions_list:
        kmer_size: int = 15
        fragment_size: int = 250
        n: int = 100

        genome = Genome(accession_id=acc,
                        data_collector=data_collector)
        yield genome.getFeatureTensor(kmer_size=kmer_size,
                                      fragment_size=fragment_size,
                                      n=n)


def getGenomeTensorDS(acc_id_tensor: tf.Tensor) -> tf.data.Dataset:
    kmer_size: int = 15
    fragment_size: int = 250
    n: int = 100
    print(acc_id_tensor)
    acc_id = tf.get_static_value(acc_id_tensor)
    genome = Genome(accession_id=acc_id,
                    data_collector=data_collector)
    return tf.data.Dataset.from_tensors(genome.getFeatureTensor(kmer_size=kmer_size,
                                                                fragment_size=fragment_size,
                                                                n=n))

fragment_size: int = 250
n: int = 100
dataset = tf.data.Dataset.from_generator(generator=create_generator,
                                         output_signature=tf.TensorSpec(shape=(None, fragment_size, 4), dtype=tf.float32))
dataset = dataset.shuffle(1000).repeat().prefetch(1)
old = None
for genome_tensor in dataset:
    start = time()
    if old is None:
        end = time()
        old = genome_tensor
        continue
    else:
        cnt = 0
        for i in range(100):
            roof = min(i + 5, 100)
            floor = max(i - 5, 0)
            for j in range(floor, roof):
                if np.isclose(old[i], genome_tensor[j]).all():
                    cnt += 1
                    continue
        print(cnt)
    end = time()
    print(end-start)

"""
the dataset will look like this
1. create a list_files dataset or you can use from_tensor_slices instead
2. create an interleave dataset
    a. map_func will be from_generator as appears upwards with the small modification that it gets a parameter which is 
    a list of all accessions
    
    b. cycle_length = it does not matter you can put 10 or 20 ...
    
    c. block_length = 1 but do not define it
    
    d. num_parallel_calls = tf.data.AUTOTUNE
"""
