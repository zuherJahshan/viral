import unittest
import Model
import tensorflow as tf
import numpy as np
from Genome import Genome

class TestModel(unittest.TestCase):
    def test_scaledDotProductAttention(self):
        n = 200
        d_key = 64
        d_val = 128
        K = tf.random.uniform(shape=[n, d_key])
        Q = tf.random.uniform(shape=[n, d_key])
        V = tf.random.uniform(shape=[n, d_val])

        scaled_dot_prod_attention = Model.ScaledDotProductAttention()
        out = scaled_dot_prod_attention(K=K,
                                        V=V,
                                        Q=Q)

        # Check dimensions compatibility
        self.assertEqual(out.shape,
                         [n, d_val])  # add assertion here

    def test_linear(self):
        n = 200
        d_model = 256
        units = 10

        X = tf.random.uniform(shape=[n, d_model])
        linear_layer = Model.Linear(units)

        out = linear_layer(X)

        # Check out dimensions
        self.assertEqual(out.shape,
                         [n, units])

        # Check number of parameters
        vars_in_layer = np.sum([np.prod(v.get_shape()) for v in linear_layer.trainable_weights])
        self.assertEqual(vars_in_layer,
                         d_model*units + units)    # |W| + |b|

    def test_multiHeadAttention(self):
        n = 200
        d_model = 256
        d_key = 64
        d_val = 128
        heads = 8

        X = tf.random.uniform(shape=[n, d_model])
        multi_head_attention = Model.MultiHeadAttention(d_model=d_model,
                                                        d_key=d_key,
                                                        d_val=d_val,
                                                        heads=heads)
        out = multi_head_attention(X)

        # Check out dimensions
        self.assertEqual(out.shape,
                         [n, d_model])

        # Check trainable params
        true_vars_in_layer = (2 * (d_model * d_key + d_key) +
                              1 * (d_model * d_val + d_val)) * heads +\
                             heads * d_val * d_model + d_model

        vars_in_layer = np.sum([np.prod(v.get_shape()) for v in multi_head_attention.trainable_weights])
        self.assertEqual(vars_in_layer,
                         true_vars_in_layer)

    def test_predictorBlock(self):
        d_out = 2
        n = 200
        d_model = 256

        X = tf.random.uniform(shape=[n, d_model])
        pred_block = Model.PredictorBlock(units=d_out)

        out = pred_block(X)

        # Check dimensions
        self.assertEqual(out.shape,
                         [n, d_out])

        # Check vars
        vars_in_layer = np.sum([np.prod(v.get_shape()) for v in pred_block.trainable_weights])
        self.assertEqual(vars_in_layer,
                         d_model * d_out + d_out)

    def test_feedForward(self):
        n = 200
        d_model = 256
        d_ff = 1024

        X = tf.random.uniform(shape=[n, d_model])
        ff_layer = Model.FeedForward(units=d_ff,
                                     outer_units=d_model)
        out = ff_layer(X)

        # Check dimensions
        self.assertEqual(out.shape,
                         [n, d_model])

        # Check Params
        true_vars_in_layer = d_model * d_ff + d_ff +\
                             d_ff * d_model + d_model
        vars_in_layer = np.sum([np.prod(v.get_shape()) for v in ff_layer.trainable_weights])
        self.assertEqual(vars_in_layer,
                         true_vars_in_layer)

    def test_resBlock(self):
        n = 200
        d_model = 256
        d_key = 64
        d_val = 128
        heads = 8
        d_ff = 1024

        X = tf.random.uniform(shape=[n, d_model])
        multi_head_attention = Model.MultiHeadAttention(d_model=d_model,
                                                        d_key=d_key,
                                                        d_val=d_val,
                                                        heads=heads)
        ff_layer = Model.FeedForward(units=d_ff,
                                     outer_units=d_model)

        res_block1 = Model.ResidualBlock(multi_head_attention)
        res_block2 = Model.ResidualBlock(ff_layer)

        Z1 = res_block1(X)
        Z2 = res_block2(X)

        # Check dimensions
        self.assertEqual(Z1.shape,
                         Z2.shape)
        self.assertEqual(Z1.shape,
                         [n, d_model])

    def test_encoderBlock(self):
        n: int = 200
        d_model: int = 256
        d_val: int = 128
        d_key: int = 64
        d_ff: int = 1024
        heads: int = 8

        print("This is encode block test")
        X = tf.random.uniform(shape=[n, d_model])
        encoder_block = Model.EncoderBlock(d_model=d_model,
                                           d_val=d_val,
                                           d_key=d_key,
                                           d_ff=d_ff,
                                           heads=heads)
        Z = encoder_block(X)

        # Check dimensions
        self.assertEqual(Z.shape,
                         [n, d_model])

    def test_modelPredict(self):
        d_model = 200
        N = 4
        n = 256
        d_out = 2
        d_ff = 2048
        d_key = 64
        d_val = 64
        heads = 2

        X = tf.random.uniform(shape=[n, d_model])

        model = Model.SarsVitModel(N=N,
                             d_out=d_out,
                             d_ff=d_ff,
                             d_key=d_key,
                             d_val=d_val,
                             heads=heads)
        pred = model.predict(X)

        # Check dimensions
        self.assertEqual(pred.shape,
                         (n, d_out))

        # Check prediction is probability
        def isProbsTensor(t):
            ones = tf.math.reduce_sum(t, axis=-1)
            is_ones = tf.math.logical_and(ones < 1.0001, ones > 0.9999)
            return tf.reduce_all(is_ones)
        self.assertTrue(isProbsTensor(pred))

if __name__ == '__main__':
    unittest.main(verbosity=10)
