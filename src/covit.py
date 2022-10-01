import os

from NNModel import NNModelHPs, NNModel
from Dataset import DatasetHPs, Dataset, base_count
from DataCollector import DataCollectorv2
from Types import *
from PredData import PredData
from matplotlib import pyplot as plt
import numpy as np
NameNNModelMap = Dict[str, NNModel]

class CovitProject(object):
    """
    This class defines the covit project. In the initialization one can create a new project or load an existing one.
    Every project will define only one Dataset, in the ways specified in the Dataset python file.

    Each covit project will have inside of it a list of small NNModel projects
    """
    def __init__(self,
                 project_name: str,
                 data_collector: DataCollectorv2,
                 dataset_hps: DatasetHPs = None):

        self.project_path = "../Projects/" + project_name + "/"
        self.nnmodels_path = self.project_path + "NNModels/"
        self.data_collector = data_collector

        self.name_nnmodel_map = {}

        # Create/Load a dataset, a project without a dataset is worthless! This call will handle creating the folders
        # of the project also.
        self.dataset: Dataset = Dataset(project_path=self.project_path,
                                        data_collector=self.data_collector,
                                        hps=dataset_hps)

    def addNNModel(self,
                   name: str,
                   nnmodel_hps: NNModelHPs=None,
                   other: str=None):

        # Check if exists
        if os.path.exists(self.nnmodels_path + name):
            print("A Neural Network model named {} already exists. loading it instead of creating".format(name))
            self.loadNNModel(name)
            return

        if other != None and nnmodel_hps != None:
            print("One of the arguments \"nnmodel_hps\" or \"other\" MUST be specified to create a new model.")

        # If does not exist, create one...
        if other == None:
            new_nnmodel = NNModel(name,
                                  nnmodels_path=self.nnmodels_path,
                                  hps=nnmodel_hps)
            self.name_nnmodel_map.update({name: new_nnmodel})
        else:
            if not other in self.name_nnmodel_map:
                print("A Neural Network model named {} does not exist in the system, please load it first.".format(name))
                return

            new_nnmodel = NNModel(name,
                                  nnmodels_path=self.nnmodels_path,
                                  hps=None,
                                  other=self.name_nnmodel_map[other])
            self.name_nnmodel_map.update({name: new_nnmodel})

    def loadNNModel(self,
                    name: str):
        if os.path.exists(self.nnmodels_path + name):
            if name in self.name_nnmodel_map:
                return
            nnmodel = NNModel(name,
                              nnmodels_path=self.nnmodels_path)
            self.name_nnmodel_map.update({name: nnmodel})
        else:
            print("Can not load the Neural Network model named {}, it does not exist in the project.".format(name))

    def getResults(self,
                   name: str):
        if name in self.name_nnmodel_map:
            return self.name_nnmodel_map[name].getResults()
        else:
            return None

    def getLineages(self):
        return self.dataset.getLineages()

    def train(self,
              name: str,
              epochs: int,
              batch_size: int,
              mini_batch_size: int = None,
              shuffle_buffer_size: int = 4096,
              min_mask_rate: float = 0.0,
              max_mask_rate: float = 0.0):
        if self.dataset.getDatasetState() != Dataset.State.SAMPLES_AVAIL:
            print("The dataset state can not allow training, only predicting!")
            print("To train please create a new project.")
        if not name in self.name_nnmodel_map:
            print("A Neural Network model named {} does not exist in the system, please load it first.".format(name))
            return
        if epochs < 0:
            print("epochs must be a positive number")

        if mini_batch_size == None:
            mini_batch_size = batch_size
        else:
            self.name_nnmodel_map[name].setBatchSize(batch_size=batch_size,
                                                     mini_batch_size=mini_batch_size)
        validset = self.dataset.getValidSet(mini_batch_size,
                                            min_mask_rate=min_mask_rate,
                                            max_mask_rate=max_mask_rate)
        trainset = self.dataset.getTrainSet(batch_size=mini_batch_size,
                                            epochs=epochs,
                                            min_mask_rate=min_mask_rate,
                                            max_mask_rate=max_mask_rate,
                                            shuffle_buffer_size=shuffle_buffer_size)
        self.name_nnmodel_map[name].train(trainset=trainset,
                                          trainset_size=self.dataset.getTrainSetSampleCount(),
                                          epochs=epochs,
                                          batch_size=mini_batch_size,
                                          validset=validset)
        self.name_nnmodel_map[name].save()

    def evaluate(self,
                 name: str,
                 batch_size: int,
                 min_mask_rate: float = 0.0,
                 max_mask_rate: float = 0.0):
        if self.dataset.getDatasetState() != Dataset.State.SAMPLES_AVAIL:
            print("The dataset state can not allow training, only predicting!")
            print("To train please create a new project.")
        if not name in self.name_nnmodel_map:
            print("A Neural Network model named {} does not exist in the system, please load it first.".format(name))
            return
        return self.name_nnmodel_map[name].evaluate(validset=self.dataset.getValidSet(batch_size=batch_size,
                                                                                      min_mask_rate=min_mask_rate,
                                                                                      max_mask_rate=max_mask_rate))

    def predict(self,
                model_name: str,
                path_to_fasta_dir: str,
                num_parallel_calls: int = 16,
                batch_size=64,
                k=5):
        """
        1. build a dataset containing the fasta files.
        2. predict
        3. build csv file
        """
        if not model_name in self.name_nnmodel_map:
            print("Model name \"{}\" is not loaded to the system, please use loadNNModel first.".format(model_name))
            return
        if not os.path.exists(path_to_fasta_dir):
            print("The path to the accessions directory \"{}\" does not exist.".format(path_to_fasta_dir))
            return
        pred_data = PredData(hps=self.dataset.hps,
                             num_parallel_calls=num_parallel_calls,
                             path_to_fasta_dir=path_to_fasta_dir)
        data = pred_data.getData(batch_size=batch_size)
        # Hold a directory of accessions and its results
        for batch in data:
            pred = self.name_nnmodel_map[model_name].predict(batch[0])
            batch_accs = [acc.decode("utf-8") for acc in batch[1].numpy()]
            results = tf.math.top_k(pred,
                                    k=k)
            pred_data.recordRes(acc_list=batch_accs,
                                results=results)

    def getAttentionImages(self,
                           model_name: str,
                           path_to_fasta_dir: str,
                           num_parallel_calls: int = 16,
                           batch_size=64,
                           head: int = 0):
        if not model_name in self.name_nnmodel_map:
            print("Model name \"{}\" is not loaded to the system, please use loadNNModel first.".format(model_name))
            return
        if not os.path.exists(path_to_fasta_dir):
            print("The path to the accessions directory \"{}\" does not exist.".format(path_to_fasta_dir))
            return

        pred_data = PredData(hps=self.dataset.hps,
                             num_parallel_calls=num_parallel_calls,
                             path_to_fasta_dir=path_to_fasta_dir)

        data = pred_data.getData(batch_size=batch_size)

        for batch in data:
            sim_matrix = self.name_nnmodel_map[model_name].getSimMatrix(batch[0],
                                                                        head=head)
            batch_accs = [acc.decode("utf-8") for acc in batch[1].numpy()]
            for i in range(len(batch[0])):
                print(batch_accs[i])
                print(sim_matrix[i])
                fname = path_to_fasta_dir + "/" + batch_accs[i] + ".png"
                print(fname)
                self._matToImage(fname=fname,
                                 X=sim_matrix[i])

    def _matToImage(self,
                    fname,
                    X):
        # Scale image
        X = X * 255
        X = X.astype(np.uint8)

        # Show image
        plt.imsave(fname=fname,
                   arr=X,
                   cmap="gray",
                   vmin=0,
                   vmax=255)



    def deepenNN(self,
                 name: str,
                 num_layers: int = 1,
                 trainable: bool = False):
        if name in self.name_nnmodel_map:
            self.name_nnmodel_map[name].deepenNN(new_layers=num_layers,
                                                 trainable=trainable)
        else:
            print("A Neural Network model named {} does not exist in the system, please load it first.".format(name))

    def changeNumClasses(self,
                         name,
                         classes):
        if name in self.name_nnmodel_map:
            self.name_nnmodel_map[name].changePredictorHead(classes=classes)
        else:
            print("A Neural Network model named {} does not exist in the system, please load it first.".format(name))

    def makeTrainable(self,
                      name):
        if name in self.name_nnmodel_map:
            self.name_nnmodel_map[name].makeTrainable()
        else:
            print("A Neural Network model named {} does not exist in the system, please load it first.".format(name))

    def listNNModels(self) -> List[str]:
        if os.path.exists(self.nnmodels_path):
            return os.listdir(self.nnmodels_path)
        else:
            return []
