import tensorflow as tf
from multi_head_attention import MultiHeadAttention


class Encoder(tf.keras.layers.Layer):

    def __init__(self, nb_encoder, **kwargs):
        self.nb_encoder = nb_encoder
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.encoder_layers = []
        for _ in range(self.nb_encoder):
            self.encoder_layers.append(EncoderLayer())
        super().build(input_shape)

    def call(self, x):
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
        return x

class EncoderLayer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.self_attention = MultiHeadAttention()
        self.norm = tf.keras.layers.LayerNormalization()
        self.dense = tf.keras.layers.Dense(512)
        super().build(input_shape)

    def call(self, x):
        self_attention = self.self_attention([x, x, x])
        self_attention_add_norm = self.norm(self_attention + x)

        dense = self.dense(self_attention_add_norm)
        return self.norm(dense + self_attention_add_norm)
