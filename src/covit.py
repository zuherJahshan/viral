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
                   nnmodel_hps: NNModelHPs=None,
                   other: str=None):

        # Check if exists
        if os.path.exists(self.nnmodels_path + name):
            print("A Neural Network model named {} already exists. loading it instead of creating".format(name))
            self.loadNNModel(name)
            return

        assert other != None or nnmodel_hps != None, \
            "One of the arguments nnmodel_hps or other MUST be specified."

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
            print("Can not load the Neural Network model named {}, it does not exist.".format(name))

    def getResults(self,
                   name: str):
        return self.name_nnmodel_map[name].getResults()

    def train(self,
              name: str,
              epochs: int,
              batch_size: int,
              mini_batch_size: int = None):
        validset = self.dataset.getValidSet(batch_size)
        if mini_batch_size == None:
            mini_batch_size = batch_size
        else:
            self.name_nnmodel_map[name].setBatchSize(batch_size=batch_size,
                                                     mini_batch_size=mini_batch_size)
        self.name_nnmodel_map[name].train(trainset=self.dataset.getTrainSet(batch_size=mini_batch_size,
                                                                            epochs=epochs),
                                          trainset_size=self.dataset.getTrainSetSampleCount(),
                                          epochs=epochs,
                                          batch_size=mini_batch_size,
                                          validset=validset)
        self.name_nnmodel_map[name].save()

    def evaluate(self,
                 name: str,
                 batch_size: int):
        return self.name_nnmodel_map[name].evaluate(validset=self.dataset.getValidSet(batch_size=batch_size))

    def deepenNN(self,
                 name: str,
                 num_layers: int = 1,
                 trainable: bool = False):
        if name in self.name_nnmodel_map:
            self.name_nnmodel_map[name].deepenNN(num_layers=num_layers,
                                                 trainable=trainable)
        else:
            print("No Neural Network named {} exists in the system".format(name))

    def changeNumClasses(self,
                         name,
                         classes):
        if name in self.name_nnmodel_map:
            self.name_nnmodel_map[name].changePredictorHead(classes=classes)
        else:
            print("No Neural Network named {} exists in the system".format(name))

    def makeTrainable(self,
                      name):
        if name in self.name_nnmodel_map:
            self.name_nnmodel_map[name].makeTrainable()
        else:
            print("No Neural Network named {} exists in the system".format(name))

    def listNNModels(self) -> List[str]:
        if os.path.exists(self.nnmodels_path):
            return os.listdir(self.nnmodels_path)
        else:
            return []
