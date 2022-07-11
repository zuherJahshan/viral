import os

from NNModel import NNModelHPs, NNModel
from Dataset import DatasetHPs, Dataset, base_count
from DataCollector import DataCollectorv2
from Types import *
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
                   nnmodel_hps = NNModelHPs):

        # Check if exists
        if os.path.exists(self.nnmodels_path + name):
            print("A Neural Network model named {} already exists. loading it instead of creating".format(name))
            self.loadNNModel(name)
            return

        # If does not exist, create one...
        new_nnmodel = NNModel(name,
                              nnmodels_path=self.nnmodels_path,
                              hps=nnmodel_hps)
        self.name_nnmodel_map.update({name: new_nnmodel})

    def loadNNModel(self,
                    name: str):
        if os.path.exists(self.nnmodels_path + name):
            if name in self.name_nnmodel_map:
                return
            input_shape = [1, self.dataset.hps.n, self.dataset.hps.frag_len, base_count]
            nnmodel = NNModel(name,
                              nnmodels_path=self.nnmodels_path,
                              input_shape=input_shape)
            self.name_nnmodel_map.update({name: nnmodel})
        else:
            print("Can not load the Neural Network model named {}, it does not exist.".format(name))

    def getResults(self,
                   name: str):
        return self.name_nnmodel_map[name].getResults()

    def train(self,
              name: str,
              epochs: int,
              batch_size: int):
        self.name_nnmodel_map[name].train(trainset=self.dataset.getTrainSet(batch_size=batch_size,
                                                                            epochs=epochs),
                                          trainset_size=self.dataset.getTrainSetSampleCount(),
                                          epochs=epochs,
                                          batch_size=batch_size)
        self.name_nnmodel_map[name].save()

    def evaluate(self,
                 name: str,
                 batch_size: int):
        return self.name_nnmodel_map[name].evaluate(validset=self.dataset.getValidSet(batch_size=batch_size))

    def deepenNN(self,
                 name: str,
                 trainable: bool = False):
        if name in self.name_nnmodel_map:
            self.name_nnmodel_map[name].deepenNN(trainable=trainable)
        else:
            print("No Neural Network named {} exists in the system".format(name))

    def listNNModels(self) -> List[str]:
        if os.path.exists(self.nnmodels_path):
            return os.listdir(self.nnmodels_path)
        else:
            return []
