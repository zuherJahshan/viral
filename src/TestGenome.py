import unittest


from Genome import Genome
from DataCollector import DataCollector

class TestGenome(unittest.TestCase):
    def test_basic(self):
        n = 250
        fragment_size = 100
        kmer_size = 30
        acc_id_list = ["MZ256063",
                       "OD959160",
                       "OM585158",
                       "OM785476",
                       "ON043700",
                       "OU127139",
                       "OU240533",
                       "OV379499",
                       "OV615556",
                       "OV741918",
                       "OV863928",
                       "OV995970",
                       "OW216182"]
        data_collector: DataCollector = DataCollector("../accessions.tsv")
        for acc_id in acc_id_list:
            genome = Genome(accession_id=acc_id,
                            data_collector=data_collector)
            genome_tensor = genome.getFeatureTensor(kmer_size=kmer_size,
                                                    fragment_size=fragment_size,
                                                    n=n)
            self.assertTrue(genome_tensor.shape == (n, fragment_size, 4))

if __name__ == '__main__':
    unittest.main()
