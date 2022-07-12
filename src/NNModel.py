import math
import os
import shutil
import pickle
from timeit import default_timer as timer

import tensorflow as tf

from Types import *
from Model import CoViTModel, custom_objects

class NNModelResults(object):
    def __init__(self):
        self.history_map: Dict[str: List] = {}
        self.train_times_map: Dict[str: float] = {}

    def getPerf(self):
        return self.history_map

    def getTimes(self):
        return self.train_times_map

    def appendTrainHist(self,
                        hist: Dict):
        for res in hist:
            if res in self.history_map:
                self.history_map[res].extend(hist[res])
            else:
                self.history_map.update({res: hist[res]})

    def appendTrainTimes(self,
                         hist: Dict):
        for res in hist:
            if res in self.train_times_map:
                self.train_times_map[res].extend([hist[res]])
            else:
                self.train_times_map.update({res: [hist[res]]})

    def save(self,
             nnmodel_path: str):
        file_name = nnmodel_path + "results.pickle"
        with open(file_name, 'wb') as outp:
            pickle.dump(self,
                        outp,
                        pickle.HIGHEST_PROTOCOL)

def loadNNModelResults(nnmodel_path: str) -> NNModelResults:
    file_name = nnmodel_path + "results.pickle"
    if os.path.exists(file_name):
        with open(file_name, "rb") as inp:
            return pickle.load(inp)
    else:
        return None

class NNModelHPs(object):
    def __init__(self,
                 encoder_repeats: int,
                 classes: int,
                 d_model: int,  # Must equal frag_len
                 d_val: int,
                 d_key: int,
                 d_ff: int,
                 heads: int,
                 dropout_rate: float):

        # TODO: Add sanity checks on the parameters of the Dataset params. To do so, please refer to the class of HPs
        #  in CoViT as an example
        self.encoder_repeats = encoder_repeats
        self.classes = classes
        self.d_model = d_model
        self.d_val = d_val
        self.d_key = d_key
        self.d_ff = d_ff
        self.heads = heads
        self.dropout_rate = dropout_rate

    def save(self,
             nnmodel_path: str):
        file_name = nnmodel_path + "hyperparameters.pickle"
        with open(file_name, 'wb') as outp:
            pickle.dump(self,
                        outp,
                        pickle.HIGHEST_PROTOCOL)

def loadNNModelHPs(nnmodel_path: str) -> NNModelHPs:
    file_name = nnmodel_path + "hyperparameters.pickle"
    if os.path.exists(file_name):
        with open(file_name, "rb") as inp:
            return pickle.load(inp)
    else:
        return None


class NNModel(object):
    def __init__(self,
                 name: str,
                 nnmodels_path: str,
                 input_shape=None,
                 hps: NNModelHPs=None,
                 other=None):
        assert hps != None or input_shape != None or other != None, \
            "if loading an existing model input_shape must be specified. If creating a new one hps must be specified!"
        self.nnmodel_path = nnmodels_path + name + "/"

        # Check if model exists
        if os.path.exists(self.nnmodel_path):
            if self.load(input_shape=input_shape):
                return

        # Check if it is a model to copy
        if other != None:
            assert input_shape != None, \
                "To build a new neural network based on other neural network, input shape must be specified"
            self.hps = other.hps
            self.results = NNModelResults()
            self.nn = self._copyNN(old_nn=other.nn,
                                   input_shape=input_shape)
            return


        # The model does not exist, create one and save it
        # First make sure the path exists
        os.makedirs(self.nnmodel_path, exist_ok=True)

        # Creating objects
        self.hps = hps
        self.results = NNModelResults()
        self.nn = CoViTModel(N=self.hps.encoder_repeats,
                             d_out=self.hps.classes,
                             d_model=self.hps.d_model,
                             d_val=self.hps.d_val,
                             d_key=self.hps.d_key,
                             d_ff=self.hps.d_ff,
                             heads=self.hps.heads,
                             dropout_rate=self.hps.dropout_rate)
        self._compileNN()

    def _exists(self):
        return os.path.exists(self._getNNPath()) and os.path.exists()

    def _compileNN(self,
                   nn=None):
        if nn == None:
            nn=self.nn
        metrics = [tf.keras.metrics.TopKCategoricalAccuracy(k=1,
                                                            name='top1_accuracy'),
                   tf.keras.metrics.TopKCategoricalAccuracy(k=2,
                                                            name='top2_accuracy'),
                   tf.keras.metrics.TopKCategoricalAccuracy(k=5,
                                                            name='top5_accuracy')
                   ]
        nn.compile(loss="categorical_crossentropy",
                   optimizer=tf.keras.optimizers.Adam(clipnorm=1.0),
                   metrics=metrics)  # the optimizer will change after debugging

    def deepenNN(self,
                 trainable: bool = False):
        self.hps.encoder_repeats += 1
        self.nn.deepen(trainable=trainable)
        self._compileNN()

    def changePredictorHead(self,
                            classes):
        self.hps.classes = classes
        self.nn.changePredictorHead(classes)
        self._compileNN()

    def train(self,
              trainset: tf.data.Dataset,
              trainset_size: int,
              epochs: int,
              batch_size: int,
              validset: tf.data.Dataset):
        # Will save newly trained model and results after training
        start = timer()
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(self._getNNPath(),
                                                           save_best_only=True)
        history = self.nn.fit(x=trainset,
                              epochs=epochs,
                              steps_per_epoch=math.floor(trainset_size / batch_size),
                              validation_data=validset,
                              callbacks=[checkpoint_cb])
        end = timer()
        self.results.appendTrainHist(history.history)
        self.results.appendTrainTimes({
            "epochs": epochs,
            "batch_size": batch_size,
            "trainset_size": trainset_size,
            "time": end - start
        })

    def getResults(self):
        return self.results


    def evaluate(self,
                 validset: tf.data.Dataset):
        return self.nn.evaluate(x=validset)

    def save(self):

        # Save HPs
        self.hps.save(nnmodel_path=self.nnmodel_path)

        # Save the results
        self.results.save(nnmodel_path=self.nnmodel_path)

        # Save the Neural Network
        # Remove old neural network and save the new one
        # nn_path = self._getNNPath()
        # if os.path.exists(nn_path):
        #     shutil.rmtree(nn_path)
        # self.nn.save(nn_path)


    def _getNNPath(self):
        return self.nnmodel_path + "nn/"


    def load(self,
             input_shape):
        # Load the HPs
        self.hps: NNModelHPs = loadNNModelHPs(nnmodel_path=self.nnmodel_path)
        if self.hps == None:
            print("here")
            return False

        # Load the results
        self.results: NNModelResults = loadNNModelResults(nnmodel_path=self.nnmodel_path)
        if self.results == None:
            return False

        # Load the existing neural network
        nn_path = self._getNNPath()
        if not os.path.exists(nn_path):
            return False

        # copy old version to the new one to retrieve CoViTModel unsaved functionalities.
        self.nn = self._copyNN(tf.keras.models.load_model(nn_path,
                                                          custom_objects=custom_objects),
                               input_shape=input_shape)

        return True


    def _copyNN(self,
                old_nn,
                input_shape):
        # Create a new neural network
        new_nn = CoViTModel(N=self.hps.encoder_repeats,
                            d_out=self.hps.classes,
                            d_model=self.hps.d_model,
                            d_val=self.hps.d_val,
                            d_key=self.hps.d_key,
                            d_ff=self.hps.d_ff,
                            heads=self.hps.heads,
                            dropout_rate=self.hps.dropout_rate)

        # compile
        self._compileNN(new_nn)

        # call it on a randomized input shape
        dummy = tf.random.uniform(input_shape)
        new_nn(dummy)

        # copy weights from previous NN
        for i in range(len(new_nn.layers)):
            new_nn.layers[i].set_weights(old_nn.layers[i].get_weights())

        return new_nn

