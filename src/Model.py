import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter
import math
from GradientAccumulator import GradientAccumulator

"""
To Remove warnings coming from model save.
as suggested in:
https://stackoverflow.com/questions/65697623/tensorflow-warning-found-untraced-functions-such-as-lstm-cell-6-layer-call-and
"""
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

class Linear(tf.keras.layers.Layer):
    """
    Linear layer, this layer receives as an input a matrix n*d
    and multiply it by the parameters that resides inside the weight matrix W of dimensions d*units
    after that it adds it to the bias b of size units*1.

    Long story short, it is just a layer that in the forward pass performs the following operation
    X@W + b
    The output is of course of size n*units
    """
    def __init__(self,
                 units: int = 1,    # the number of units (neurons)
                 **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self,
              batch_input_shape):
        self.kernel = self.add_weight(
            name="kernel",
            shape=[batch_input_shape[-1], self.units],
            initializer="glorot_normal",
            regularizer=tf.keras.regularizers.L2(0.00000001),
            #constraint=tf.keras.constraints.max_norm(2.0)
        )
        self.bias = self.add_weight(
            name="bias",
            shape=[self.units],
            initializer="zeros"
        )
        super().build(batch_input_shape)    # Call for father class to complete building

    def call(self,
             X):
        Z = X @ self.kernel + self.bias
        return Z

    def compute_output_shape(self,
                             batch_input_shape):
        return tf.TensorShape(batch_input_shape.as_list()[:-1] +
                              [self.units])    # batch shape, n, units

    def getHP(self):
        return {
            "units": self.units
        }

    def get_config(self):
        config = super().get_config()
        config.update(self.getHP())
        return config

class ScaledDotProductAttention(tf.keras.layers.Layer):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)
        self.softmax = tf.keras.layers.Softmax()
        self.similarity = None

    def call(self,
             K,
             V,
             Q):

        # The dimensionality of the key
        d_k = K.shape[-1]

        # Preparing permutation of tensor
        perm = [i for i in range(len(K.shape))]
        tmp = perm[-1]
        perm[-1] = perm[-2]
        perm[-2] = tmp

        # Matmul
        Z = Q @ tf.transpose(K, perm=perm)

        # Scale
        Z = Z / math.sqrt(d_k)

        # SoftMax
        Z = self.softmax(Z)
        self.similarity = Z

        # Matmul with values and return
        return Z @ V

    def getSimMatrix(self):
        return self.similarity


class MyMultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self,
                 d_key: int,
                 d_val: int,
                 d_model: int,
                 heads: int,
                 dropout_rate: float = 0.1,
                 **kwargs):
        super().__init__(**kwargs)
        self.heads = heads
        self.sim_matrices = [None] * self.heads
        self.lin_key_heads = [Linear(d_key) for _ in range(self.heads)]
        self.lin_qry_heads = [Linear(d_key) for _ in range(self.heads)]
        self.lin_val_heads = [Linear(d_val) for _ in range(self.heads)]
        self.scaled_dot_prod_attention = ScaledDotProductAttention()
        self.lin_out = Linear(d_model)
        self.dropout_rate = dropout_rate
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)

    # No need for build. No independent weights are defined in this layer. Every weight is a part of an inner layer.

    def call(self, X):

        # TODO: substitute for loop with vectorization.
        Z = []
        for i in range(self.heads):
            K = self.lin_key_heads[i](X)
            Q = self.lin_qry_heads[i](X)
            V = self.lin_val_heads[i](X)

            Z.append(self.scaled_dot_prod_attention(K=K,
                                                    V=V,
                                                    Q=Q))
            self.sim_matrices[i] = self.scaled_dot_prod_attention.getSimMatrix()

        Z = tf.concat(Z, -1)
        return self.dropout(self.lin_out(Z) + X)

    def compute_output_shape(self,
                             batch_input_shape):
        return tf.TensorShape(batch_input_shape)

    def getHP(self):
        return {
            "d_key": self.lin_key_heads[0].getHP()["units"],
            "d_val": self.lin_val_heads[0].getHP()["units"],
            "d_model": self.lin_out.getHP()["units"],
            "heads": self.heads,
            "dropout_rate": self.dropout_rate
        }

    def getSimMatrix(self,
                     head: int):
        if head >= self.heads:
            head = -1
        return self.sim_matrices[head]

    def get_config(self):
        config = super().get_config()
        config.update(self.getHP())
        return config


class PredictorBlock(Linear):
    """
    This class gets as an input a genome feature tensor in the form of n*d_m_model.
    this input describes n feature vectors (of size d_model) of the genome sitting horizontally.
    This layer is a dense layer with the objective of deciding which genomic class the input is from.

    This means calculates, softmax((X@W+b)[0])
    and outputs it (the dims of the output is exactly like the dims of the input).
    """
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

    def call(self, X):
        Z = super().call(X)     # Z is of shape (batch_size, compression factor<n>, d_out)
        return tf.keras.activations.softmax(Z, axis=-1)

    def getHP(self):
        return super().getHP()

    def get_config(self):
        base_config = super().get_config()
        return {**base_config}

class FeedForward(Linear):
    """
    This class gets as an input a genome in the form of n*d_m_model.
    this input describes n feature vectors (of size d_model) of the genome sitting horizontally.
    This FeedForward layer is just forwarding each feature vector (+ attention of course) through a dense layer.

    This means calculates, activation(X@W+b)
    and outputs it (the dims of the output is exactly like the dims of the input).
    """
    def __init__(self,
                 outer_units: int = 1,
                 activation: str = "relu",
                 dropout_rate: float = 0.1,
                 **kwargs):
        super().__init__(**kwargs)
        self.activation_name = activation
        self.activation = tf.keras.activations.get(activation)
        self.outer_layer = Linear(outer_units)
        self.dropout_rate = dropout_rate
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)

    def call(self, X):
        return self.dropout(self.outer_layer(self.activation(X@self.kernel + self.bias)) + X)

    def getHP(self):
        hp = super().getHP()
        hp.update({
            "activation": self.activation_name,
            "outer_units": self.outer_layer.getHP()["units"],
            "dropout_rate": self.dropout_rate,
        })
        return hp

    def get_config(self):
        config = super().get_config()
        config.update(self.getHP())
        return config


class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self,
                 d_model: int = 1,
                 d_val: int = 1,
                 d_key: int = 1,
                 d_ff: int = 1,
                 heads: int = 1,
                 dropout_rate: float = 0.1,
                 **kwargs):
        super().__init__(**kwargs)
        self.mha = MyMultiHeadAttention(d_val=d_val,
                                        d_key=d_key,
                                        d_model=d_model,
                                        heads=heads,
                                        dropout_rate=dropout_rate)
        self.ff = FeedForward(outer_units=d_model,
                              units=d_ff,
                              dropout_rate=dropout_rate)  # TODO: Maybe add gelu as the activation following ViT
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()

    def call(self,
             X):
        Z = self.norm1(X)
        Z = self.mha(Z)
        Z = self.norm2(Z)
        Z = self.ff(Z)
        return Z

    def getHP(self):
        hp = self.mha.getHP()
        hp.update({
            "d_ff": self.ff.getHP()["units"]
        })
        return hp

    def getSimMatrix(self,
                     head):
        return self.mha.getSimMatrix(head)

    def get_config(self):
        config = super().get_config()
        config.update(self.getHP())
        return config

class CoViTModel(tf.keras.Model):
    def __init__(self,
                 N: int = 1,    # Number of repeats of the EncoderBlock
                 d_out: int = 2,    # The number of classes
                 d_model: int = 1,
                 d_val: int = 1,
                 d_key: int = 1,
                 d_ff: int = 1,
                 heads: int = 1,
                 dropout_rate: float = 0.1,
                 **kwargs):
        super().__init__(**kwargs)
        self.N = N
        self.base_embedding = Linear(units=1)
        self.encoder_blocks = [EncoderBlock(d_model=d_model,
                                            d_val=d_val,
                                            d_key=d_key,
                                            d_ff=d_ff,
                                            heads=heads,
                                            dropout_rate=dropout_rate) for _ in range(self.N)]
        self.norm = tf.keras.layers.LayerNormalization()
        self.out = PredictorBlock(units=d_out)

    def call(self, X):
        Z = self.base_embedding(X)   # X of size [batch, n, d_model, 4] -> transforms to [batch, n, d_model]
        Z = tf.squeeze(Z,
                       axis=[-1])
        for encoder_block in self.encoder_blocks:
            Z = encoder_block(Z)
        Z = self.norm(Z)
        Z = tf.squeeze(tf.split(value=Z,
                                num_or_size_splits=Z.shape[-2],
                                axis=-2)[0],
                       axis=[-2])
        return self.out(Z)

    def train_step(self, data):
        """The logic for one training step.
        This method can be overridden to support custom training logic.
        For concrete examples of how to override this method see
        [Customizing what happends in fit](https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit).
        This method is called by `Model.make_train_function`.
        This method should contain the mathematical logic for one step of training.
        This typically includes the forward pass, loss calculation, backpropagation,
        and metric updates.
        Configuration details for *how* this logic is run (e.g. `tf.function` and
        `tf.distribute.Strategy` settings), should be left to
        `Model.make_train_function`, which can also be overridden.
        Args:
          data: A nested structure of `Tensor`s.
        Returns:
          A `dict` containing values that will be passed to
          `tf.keras.callbacks.CallbackList.on_train_batch_end`. Typically, the
          values of the `Model`'s metrics are returned. Example:
          `{'loss': 0.2, 'accuracy': 0.7}`.
        """
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        # Run forward pass.
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(x, y, y_pred, sample_weight)
        self._validate_target_and_loss(y, loss)
        # Run backwards pass.

        #self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        grads_and_vars = self.optimizer._compute_gradients(
            loss,
            var_list=self.trainable_variables,
            tape=tape
        )
        self.optimizer.apply_gradients(grads_and_vars)

        return self.compute_metrics(x, y, y_pred, sample_weight)

    def get_config(self):
        config = super().get_config()
        config.update({
            "N": self.N,
            "d_out": self.out.getHP()["units"]
        })
        config.update(self.encoder_blocks[0].getHP())
        return config

    def getSimMatrix(self,
                     head: int):
        return self.encoder_blocks[0].getSimMatrix(head)

class JensenShannonLoss(tf.keras.losses.Loss):
    def __init__(self,
                 pi: float = 0.1,
                 **kwargs):
        self.pi = pi
        self.kl = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE)
        super().__init__(**kwargs)

    def call(self,
             y_true,
             y_pred):
        pi1 = self.pi
        pi2 = 1 - pi1
        m = pi1 * y_true + pi2 * y_pred
        djs = pi1 * self.kl(y_true, m) + pi2 * self.kl(y_pred, m)
        z = -pi2 * tf.math.log(pi2)
        return djs / z

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "pi": self.pi}

custom_objects = {"Linear": Linear,
                  "ScaledDotProductAttention": ScaledDotProductAttention,
                  "MyMultiHeadAttention": MyMultiHeadAttention,
                  "PredictorBlock": PredictorBlock,
                  "FeedForward": FeedForward,
                  "EncoderBlock": EncoderBlock,
                  "CoViTModel": CoViTModel,
                  "GradientAccumulator": GradientAccumulator}
