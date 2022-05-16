import numpy as np
import tensorflow as tf
import math

class Linear(tf.keras.layers.Layer):
    """
    Linear layer, this layer receives as an input a matrix n*d_model
    and multiply it by the parameters that resides inside the weight matrix W of dimensions d_model*units
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
            initializer="glorot_normal" # TODO: check initializers
        )
        self.bias = self.add_weight(
            name="bias",
            shape=[self.units],
            initializer="zeros"
        )
        super().build(batch_input_shape)    # Call for father class to complete building

    def call(self,
             X):
        return X @ self.kernel + self.bias

    def compute_output_shape(self,
                             batch_input_shape):
        return tf.TensorShape(batch_input_shape.as_list()[:-1] +
                              [self.units])    # batch shape, n, units

    def get_config(self):
        base_config = super().get_config()
        return {**base_config,
                "units": self.units
                }

"""
def scaledDotProductAttention(K,
                              V,
                              Q):
    '''
    Receives three matrices, the Key matrix (K), the Value matrix (V), and the Query matrix(Q)
    shape(K) = n * d_k
    shape(Q) = n * d_k
    shape(V) = n * d_v
    '''
    d_k = K.shape[-2]

    # Matmul
    Z = Q @ tf.transpose(K)

    # Scale
    Z = Z / tf.sqrt(d_k)

    # SoftMax
    Z = tf.nn.softmax(Z)

    # Matmul with values and return
    return Z @ V

# ScaledDotProductAttention = tf.keras.layers.Lambda(lambda K, V, Q: scaledDotProductAttention(K, V, Q))
"""

class ScaledDotProductAttention(tf.keras.layers.Layer):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)
        self.softmax = tf.keras.layers.Softmax()

    def call(self,
             K,
             V,
             Q):
        d_k = K.shape[-1]
        last_dim = len(K.shape) - 1
        # Matmul
        Z = Q @ tf.transpose(K, perm=[last_dim, last_dim-1])
        # Scale
        Z = Z / math.sqrt(d_k)
        # SoftMax
        Z = self.softmax(Z)
        # Matmul with values and return
        return Z @ V



class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self,
                 d_key: int,
                 d_val: int,
                 d_model: int,
                 heads: int,
                 **kwargs):
        super().__init__(**kwargs)
        self.heads = heads
        self.lin_key_heads = [Linear(d_key) for _ in range(self.heads)]
        self.lin_qry_heads = [Linear(d_key) for _ in range(self.heads)]
        self.lin_val_heads = [Linear(d_val) for _ in range(self.heads)]
        self.scaled_dot_prod_attention = ScaledDotProductAttention()
        self.lin_out = Linear(d_model)  # Think about it

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
        Z = tf.concat(Z, -1)
        return self.lin_out(Z)

    def compute_output_shape(self,
                             batch_input_shape):
        return tf.TensorShape(batch_input_shape)

    def get_config(self):
        config = super().get_config()
        config += {"lin_key_heads_{}".format(i): self.lin_key_heads[i].get_config() for i in range(self.heads)}
        config += {"lin_val_heads_{}".format(i): self.lin_val_heads[i].get_config() for i in range(self.heads)}
        config += {"lin_qry_heads_{}".format(i): self.lin_qry_heads[i].get_config() for i in range(self.heads)}
        config += {"lin_out": self.lin_out.get_config()}
        config += {"heads": self.heads}
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
        return tf.keras.activations.softmax((X@self.kernel + self.bias),
                                            axis=-1)    # need to be changed

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
                 **kwargs):
        super().__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)
        self.outer_layer = Linear(outer_units)

    def call(self, X):
        return self.outer_layer(self.activation(X@self.kernel + self.bias))

    def get_config(self):
        base_config = super().get_config()
        return {**base_config,
                "activation": tf.keras.activations.serialize(self.activation),
                "outer_units": self.outer_layer.get_config()
                }

class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self,
                 layer: tf.keras.layers.Layer,
                 **kwargs):
        super().__init__(**kwargs)
        self.layer = layer

    def call(self, X):
        return self.layer(X) + X

    def get_config(self):
        config = super().get_config()
        config += self.layer.get_config()
        return config


class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self,
                 d_model: int = 1,
                 d_val: int = 1,
                 d_key: int = 1,
                 d_ff: int = 1,
                 heads: int = 1,
                 **kwargs):
        super().__init__(**kwargs)
        self.MHA_res_block = ResidualBlock(MultiHeadAttention(d_val=d_val,
                                                              d_key=d_key,
                                                              d_model=d_model,
                                                              heads=heads))
        self.FF_res_block = ResidualBlock(FeedForward(outer_units=d_model,
                                                      units=d_ff))  # TODO: Maybe add gelu as the activation following ViT
        self.norm = tf.keras.layers.Normalization()

    def call(self,
             X):
        Z = self.norm(X)
        Z = self.MHA_res_block(Z)
        Z = self.norm(Z)
        Z = self.FF_res_block(Z)
        return Z

    def get_config(self):
        config = super().get_config()
        config += self.MHA_res_block.get_config()
        config += self.FF_res_block.get_config()
        return config

class SarsVitModel(tf.keras.Model):
    def __init__(self,
                 N: int = 1,    # Number of repeats of the EncoderBlock
                 d_out: int = 2,    # The number of classes
                 d_model: int = 1,
                 d_val: int = 1,
                 d_key: int = 1,
                 d_ff: int = 1,
                 heads: int = 1,
                 **kwargs):
        super().__init__(**kwargs)
        self.N = N
        self.encoder_blocks = [EncoderBlock(d_model=d_model,
                                            d_val=d_val,
                                            d_key=d_key,
                                            d_ff=d_ff,
                                            heads=heads) for _ in range(self.N)]
        self.out = PredictorBlock(units=d_out)

    def call(self, X):
        Z = X
        cnt = 0
        for encoder_block in self.encoder_blocks:
            cnt += 1
            Z = encoder_block(Z)
        return self.out(Z)

    def get_config(self):
        config = super().get_config()
        config += {"N": self.N}
        for encoder_block in self.encoder_blocks:
            config += encoder_block.get_config()
        config += self.out.get_config()
        return config


"""
TODO: 
    1. Fix get_config's and try saving the model to see that it works.
    2. Try to include GPU in calculations
    3. Parallelize MultiHeadAttention loop.
    4. Work on training loop.
"""
