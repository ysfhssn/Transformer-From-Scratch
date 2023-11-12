import tensorflow as tf
from multi_head_attention import MultiHeadAttention
from feed_forward import FeedForward
import hyperparameters as hp


class Decoder(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        self.decoder_layers = []
        for _ in range(hp.num_decoders):
            self.decoder_layers.append(DecoderLayer())
        super().build(input_shape)

    def call(self, x):
        decoder_input, encoder_output = x
        for decoder_layer in self.decoder_layers:
            decoder_input = decoder_layer([decoder_input, encoder_output])
        return decoder_input

class DecoderLayer(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        self.self_attention = MultiHeadAttention(masked=True)
        self.attention = MultiHeadAttention()
        self.norm = tf.keras.layers.LayerNormalization()
        self.ff = FeedForward()
        super().build(input_shape)

    def call(self, x):
        decoder_input, encoder_output = x

        self_attention = self.self_attention((decoder_input, decoder_input, decoder_input))
        self_attention_add_norm = self.norm(self_attention + decoder_input)

        attention = self.attention((self_attention_add_norm, encoder_output, encoder_output)) # Q K V
        attention_add_norm = self.norm(attention + self_attention_add_norm)

        ff = self.ff(attention_add_norm)
        return self.norm(ff + attention_add_norm)
