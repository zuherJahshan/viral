#!/usr/bin/env python3.8

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# from DataCollector import DataCollectorv2
# from covit import CovitProject
#
# dc = DataCollectorv2()
#
# covit = CovitProject(project_name="107Lins",
#                      data_collector=dc)
#
# model_name = covit.listNNModels()[-1]
#
# covit.loadNNModel(name=model_name)
#
# path_to_fasta_dir = "../../cmpCovit/covit/"
#
# covit.predict(model_name=model_name,
#               path_to_fasta_dir=path_to_fasta_dir)

import tensorflow as tf

from DataCollector import DataCollectorv2
from Dataset import DatasetHPs
from NNModel import NNModelHPs
from covit import CovitProject
import sys

dc = DataCollectorv2()

lins = dc.getLocalLineages(int(sys.argv[1]))

print(len(lins))

hps = DatasetHPs(lineages=lins,
                 max_accs_per_lineage=int(sys.argv[2]),
                 frag_len=256,
                 kmer_size=16,
                 n=256,
                 validation_split=0.05)

covit = CovitProject(project_name=str(len(lins)) + "LinsNew",
                     data_collector=dc,
                     dataset_hps=hps)
