import tensorflow as tf
import hyperparameters as hp


class FeedForward(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(hp.dff, activation='relu'),
            tf.keras.layers.Dense(hp.d_model),
            tf.keras.layers.Dropout(hp.dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()
        super().build(input_shape)

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x
