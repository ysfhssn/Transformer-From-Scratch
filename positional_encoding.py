import tensorflow as tf


def positional_encoding(position_dim, embedding_dim, n=10_000):
    P = tf.zeros((position_dim, embedding_dim))
    for pos in range(position_dim):
        for i in tf.range(embedding_dim//2):
            denominator = tf.pow(n, 2*i/embedding_dim)
            P = tf.tensor_scatter_nd_update(P, [[pos, 2*i]], [tf.sin(pos/denominator)])
            P = tf.tensor_scatter_nd_update(P, [[pos, 2*i+1]], [tf.cos(pos/denominator)])
    return P
