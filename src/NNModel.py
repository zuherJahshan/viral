import math
import os
import pickle

from Types import *
from Model import CoViTModel, custom_objects

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
    with open(file_name, "rb") as inp:
        return pickle.load(inp)


class NNModel(object):
    def __init__(self,
                 name: str,
                 nnmodels_path: str,
                 hps: NNModelHPs = None):
        self.nnmodel_path = nnmodels_path + name + "/"
        # Check if exists
        if os.path.exists(self.nnmodel_path):
            self.load()


        # The model does not exist, create one and save it
        else:
            # Creating objects
            self.hps = hps
            self.nn = CoViTModel(N=self.hps.encoder_repeats,
                                 d_out=self.hps.classes,
                                 d_model=self.hps.d_model,
                                 d_val=self.hps.d_val,
                                 d_key=self.hps.d_key,
                                 d_ff=self.hps.d_ff,
                                 heads=self.hps.heads,
                                 dropout_rate=self.hps.dropout_rate)
            self._compileNN()


    def _compileNN(self):
        metrics = [tf.keras.metrics.TopKCategoricalAccuracy(k=1,
                                                            name='top1_accuracy'),
                   tf.keras.metrics.TopKCategoricalAccuracy(k=2,
                                                            name='top2_accuracy'),
                   tf.keras.metrics.TopKCategoricalAccuracy(k=5,
                                                            name='top5_accuracy')
                   ]
        self.nn.compile(loss="categorical_crossentropy",
                            optimizer=tf.keras.optimizers.Adam(clipnorm=1.0),
                            metrics=metrics)  # the optimizer will change after debugging

    def deepenNN(self,
                 trainable: bool = False):
        self.hps.encoder_repeats += 1
        self.nn.deepen(trainable=trainable)
        self._compileNN()

    def train(self,
              trainset: tf.data.Dataset,
              trainset_size: int,
              epochs: int,
              batch_size: int):      # Returns a history object
        # Will save newly trained model and results after training
        history = self.nn.fit(x=trainset,
                              epochs=epochs,
                              steps_per_epoch=math.floor(trainset_size / batch_size))
        return history


    def evaluate(self,
                 dataset: tf.data.Dataset):
        pass

    def save(self):
        # Make sure the path exists
        os.makedirs(self.nnmodel_path, exist_ok=True)

        # Save HPs
        self.hps.save(nnmodel_path=self.nnmodel_path)

        # Save the Neural Network
        nn_path = self.nnmodel_path + "nn/"
        self.nn.save(nn_path)

    #def remove(self):

    def load(self):
        # Load the HPs
        self.hps: NNModelHPs = loadNNModelHPs(nnmodel_path=self.nnmodel_path)

        # Load the existing neural network
        nn_path = self.nnmodel_path + "nn/"
        self.nn: CoViTModel = tf.keras.models.load_model(nn_path,
                                                         custom_objects=custom_objects)

