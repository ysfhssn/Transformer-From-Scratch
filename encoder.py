import tensorflow as tf
from multi_head_attention import MultiHeadAttention
from feed_forward import FeedForward
import hyperparameters as hp


class Encoder(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        self.encoder_layers = []
        for _ in range(hp.num_encoders):
            self.encoder_layers.append(EncoderLayer())
        super().build(input_shape)

    def call(self, x):
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
        return x

class EncoderLayer(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        self.self_attention = MultiHeadAttention()
        self.norm = tf.keras.layers.LayerNormalization()
        self.ff = FeedForward()
        super().build(input_shape)

    def call(self, x):
        self_attention = self.self_attention([x, x, x])
        self_attention_add_norm = self.norm(self_attention + x)

        ff = self.ff(self_attention_add_norm)
        return self.norm(ff + self_attention_add_norm)
