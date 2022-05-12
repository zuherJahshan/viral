import model
from model import SarsVitModel, ScaledDotProductAttention
from Genome import Genome
import tensorflow as tf

def testScaledDotProductAttention():
    n = 200
    d_key = 64
    d_val = 128
    K = tf.random.uniform(shape=[n, d_key])
    Q = tf.random.uniform(shape=[n, d_key])
    V = tf.random.uniform(shape=[n, d_val])
    print(K)

    scaled_dot_prod_attention = ScaledDotProductAttention()
    scaled_dot_prod_attention(K=K,
                              V=V,
                              Q=Q)

def testLinear():
    n = 200
    d_model = 256
    units = 10

    X = tf.random.uniform(shape=[n, d_model])
    linear_layer = model.Linear(units)
    assert linear_layer(X).shape == [n, units]

def testMultiHeadAttention():
    n = 200
    d_model = 256
    d_key = 64
    d_val = 128
    heads = 8

    X = tf.random.uniform(shape=[n, d_model])
    multi_head_attention = model.MultiHeadAttention(d_model=d_model,
                                                    d_key=d_key,
                                                    d_val=d_val,
                                                    heads=heads)
    Z = multi_head_attention(X)
    print(X)


def testPredictorBlock():
    d_out = 2
    n = 200
    d_model = 256

    X = tf.random.uniform(shape=[n, d_model])
    pred_block = model.PredictorBlock(units=d_out)
    print(pred_block(X))

def testFF():
    n = 200
    d_model = 256
    d_ff = 1024

    X = tf.random.uniform(shape=[n, d_model])
    ff_layer = model.FeedForward(units=d_ff,
                                 outer_units=d_model)
    Z = ff_layer(X)
    print(Z)
    assert Z.shape == [n, d_model]


def testResBlock():
    n = 200
    d_model = 256
    d_key = 64
    d_val = 128
    heads = 8
    d_ff = 1024

    X = tf.random.uniform(shape=[n, d_model])
    multi_head_attention = model.MultiHeadAttention(d_model=d_model,
                                                    d_key=d_key,
                                                    d_val=d_val,
                                                    heads=heads)
    ff_layer = model.FeedForward(units=d_ff,
                                 outer_units=d_model)

    res_block1 = model.ResidualBlock(multi_head_attention)
    res_block2 = model.ResidualBlock(multi_head_attention)

    Z1 = res_block1(X)
    Z2 = res_block2(X)
    print(Z1)
    print(Z2)

def testEncoderBlock():
    n: int = 200
    d_model: int = 256
    d_val: int = 128
    d_key: int = 64
    d_ff: int = 1024
    heads: int = 8

    print("This is encode block test")
    X = tf.random.uniform(shape=[n, d_model])
    encoder_block = model.EncoderBlock(d_model=d_model,
                                       d_val=d_val,
                                       d_key=d_key,
                                       d_ff=d_ff,
                                       heads=heads)
    Z = encoder_block(X)
    print(Z)


def basic_test():
    kmer_size = 30
    d_model = fragment_size = 200
    n = 256
    d_out = 2
    d_ff = 1024
    d_key = 256
    d_val = 256
    heads = 8

    genome = Genome("MZ256063")
    genome_tensor = genome.getFeatureTensor(kmer_size=kmer_size,
                                             fragment_size=fragment_size,
                                             n=n)
    #print(genome_tensor)
    model = SarsVitModel(N=4,
                         d_out=d_out,
                         d_ff=d_ff,
                         d_key=d_key,
                         d_val=d_val,
                         heads=heads)
    print("=====> Genome tensor is: ========")
    print(genome_tensor)
    model.predict(genome_tensor)

testScaledDotProductAttention()
testLinear()
testMultiHeadAttention()
testPredictorBlock()
testFF()
testResBlock()
testEncoderBlock()
basic_test()
