import math
import os
import shutil
import pickle
from timeit import default_timer as timer
from Genome import base_count

import tensorflow as tf
from GradientAccumulator import GradientAccumulator

from Types import *
from Model import CoViTModel, custom_objects

class NNModelResults(object):
    def __init__(self):
        self.history_map: Dict[str: List] = {}
        self.train_times_map: Dict[str: float] = {}
        self.train_history_map = self.history_map

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
                 hps: NNModelHPs=None,
                 other=None):
        self.nnmodel_path = nnmodels_path + name + "/"

        # Check if model exists
        if os.path.exists(self.nnmodel_path):
            if self.load():
                return

        assert hps != None or other != None, \
            "If creating a new mudel hps or other must be specified!"
        # Check if it is a model to copy
        if other != None:
            self.hps = other.hps
            self.results = NNModelResults()
            self.nn = self._copyNN(old_nn=other.nn)
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

        input_shape = (1, 1, self.hps.d_model, base_count)
        dummy = tf.random.uniform(input_shape)
        self.nn(dummy)


    def _exists(self):
        return os.path.exists(self._getNNPath()) and os.path.exists()

    def _compileNN(self,
                   nn=None,
                   steps=1):
        if nn == None:
            nn=self.nn
        metrics = [tf.keras.metrics.TopKCategoricalAccuracy(k=1,
                                                            name='top1_accuracy'),
                   tf.keras.metrics.TopKCategoricalAccuracy(k=2,
                                                            name='top2_accuracy'),
                   tf.keras.metrics.TopKCategoricalAccuracy(k=5,
                                                            name='top5_accuracy')
                   ]
        optimizer = tf.keras.optimizers.Adam(clipnorm=1.0)
        optimizer = GradientAccumulator(optimizer=optimizer,
                                        accum_steps=steps)
        nn.compile(loss="categorical_crossentropy",
                   optimizer=optimizer,
                   metrics=metrics)  # the optimizer will change after debugging

    def deepenNN(self,
                 new_layers: int = 1,
                 trainable: bool = False):
        self.hps.encoder_repeats += num_layers
        self.nn = self._copyNN(old_nn=self.nn,
                               encoder_repeats=self.hps.encoder_repeats,
                               trainable=trainable)

    def changePredictorHead(self,
                            classes):
        self.hps.classes = classes
        self.nn = self._copyNN(old_nn=self.nn,
                               classes=classes)

    def setBatchSize(self,
                     batch_size,
                     mini_batch_size):
        assert batch_size % mini_batch_size == 0,\
            "batch_size must be a multiplication of mini_batch_size."
        grad_accum_steps = int(batch_size / mini_batch_size)
        self.nn = self._copyNN(old_nn=self.nn,
                               grad_accum_steps=grad_accum_steps)

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
                              callbacks=[checkpoint_cb],
                              )
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

    def predict(self,
                data):
        return self.nn(data)

    def getSimMatrix(self,
                     data,
                     head: int):
        self.nn(data)
        return self.nn.getSimMatrix(head).numpy()

    def save(self):
        # Save HPs
        self.hps.save(nnmodel_path=self.nnmodel_path)

        # Save the results
        self.results.save(nnmodel_path=self.nnmodel_path)


    def _getNNPath(self):
        return self.nnmodel_path + "nn/"


    def load(self):
        # Load the HPs
        self.hps: NNModelHPs = loadNNModelHPs(nnmodel_path=self.nnmodel_path)
        if self.hps == None:
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
                                                          custom_objects=custom_objects))

        return True

    def makeTrainable(self):
        for layer in self.nn.layers:
            layer.trainable = True

    def _copyNN(self,
                old_nn,
                encoder_repeats: int = None,
                classes: int = None,
                grad_accum_steps=1,
                trainable=True):
        # Create a new neural network
        if encoder_repeats == None:
            encoder_repeats = old_nn.get_config()["N"]
        if classes == None:
            classes = old_nn.get_config()["d_out"]

        input_shape = (1, 1, old_nn.get_config()["d_model"], base_count)

        new_nn = CoViTModel(N=encoder_repeats,
                            d_out=classes,
                            d_model=old_nn.get_config()["d_model"],
                            d_val=old_nn.get_config()["d_val"],
                            d_key=old_nn.get_config()["d_key"],
                            d_ff=old_nn.get_config()["d_ff"],
                            heads=old_nn.get_config()["heads"],
                            dropout_rate=old_nn.get_config()["dropout_rate"])

        # compile
        self._compileNN(new_nn,
                        steps=grad_accum_steps)

        # call it on a randomized input shape
        dummy = tf.random.uniform(input_shape)
        new_nn(dummy)

        # Layers to copy from the old NN is the base embedding layer, and all of the encoders that can be copied
        ## If nothing changed, please copy all!
        if encoder_repeats == old_nn.get_config()["N"] and classes == old_nn.get_config()["d_out"]:
            layers_to_copy = len(old_nn.layers)
        else:
            layers_to_copy = 1 + min(encoder_repeats, old_nn.get_config()["N"])

        # copy weights from previous NN
        for i in range(layers_to_copy):
            new_nn.layers[i].set_weights(old_nn.layers[i].get_weights())
            new_nn.layers[i].trainable = trainable


        return new_nn
