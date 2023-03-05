import tensorflow as tf


class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, dim=512, nb_head=8, masked=False, **kwargs):
        self.dim = dim
        self.head_dim = dim // nb_head
        self.nb_head = nb_head
        self.masked = masked
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.query_dense = tf.keras.layers.Dense(512)
        self.key_dense = tf.keras.layers.Dense(512)
        self.value_dense = tf.keras.layers.Dense(512)
        if self.masked:
            self.mask = tf.where(
                tf.sequence_mask(tf.range(input_shape[0][-2]) + 1, input_shape[1][-2]),
                0.,
                -float('inf')
            )
        self.dense = tf.keras.layers.Dense(512)
        super().build(input_shape)

    def call(self, x):

        Q, K, V = x
        batch_size = tf.shape(Q)[0]
        Q_seq_len = tf.shape(Q)[1]
        K_seq_len = tf.shape(K)[1]
        V_seq_len = tf.shape(V)[1]

        Q = self.query_dense(Q)
        K = self.key_dense(K)
        V = self.value_dense(V)

        Q = tf.reshape(Q, (batch_size, Q_seq_len, self.nb_head, self.head_dim))
        K = tf.reshape(K, (batch_size, K_seq_len, self.nb_head, self.head_dim))
        V = tf.reshape(V, (batch_size, V_seq_len, self.nb_head, self.head_dim))

        Q = tf.transpose(Q, (0, 2, 1, 3))
        K = tf.transpose(K, (0, 2, 1, 3))
        V = tf.transpose(V, (0, 2, 1, 3))

        Q = tf.reshape(Q, (batch_size * self.nb_head, Q_seq_len, self.head_dim))
        K = tf.reshape(K, (batch_size * self.nb_head, K_seq_len, self.head_dim))
        V = tf.reshape(V, (batch_size * self.nb_head, V_seq_len, self.head_dim))

        # Scaled dot product attention
        QK = tf.matmul(Q, K, transpose_b=True)
        QK = QK / tf.math.sqrt(float(self.head_dim))

        if self.masked:
            softmax_QK = tf.nn.softmax(QK + self.mask)
        else:
            softmax_QK = tf.nn.softmax(QK)

        attention = tf.matmul(softmax_QK, V)
        attention = tf.reshape(attention, (batch_size, self.nb_head, Q_seq_len, self.head_dim))
        attention = tf.transpose(attention, (0, 2, 1, 3))
        attention = tf.reshape(attention, (batch_size, Q_seq_len, self.nb_head*self.head_dim)) # Concat

        return self.dense(attention)
