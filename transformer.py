import tensorflow as tf
from positional_encoding import positional_encoding
from encoder import Encoder
from decoder import Decoder
import hyperparameters as hp


class Transformer(tf.keras.Model):

    def __init__(self, *, input_vocab_size, output_vocab_size=None):
        super().__init__()
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size or input_vocab_size

    def build(self, input_shape):
        # Embeddings
        self.input_embedding = tf.keras.layers.Embedding(self.input_vocab_size, hp.d_model)
        self.output_embedding = tf.keras.layers.Embedding(self.output_vocab_size, hp.d_model)

        # Positional encodings
        self.input_encoding = positional_encoding(input_shape[0][-1], hp.d_model)
        self.output_encoding = positional_encoding(input_shape[1][-1], hp.d_model)

        # Encoders
        self.encoders = Encoder()

        # Decoders
        self.decoders = Decoder()

        # Output
        self.output_probs = tf.keras.layers.Dense(self.output_vocab_size, activation='softmax')

        # Hyperparameters
        self.scce = tf.keras.losses.SparseCategoricalCrossentropy()
        self.learning_rate = CustomSchedule(d_model=hp.d_model)
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        self.compile(loss=self.scce, optimizer=self.optimizer)

        super().build(input_shape)

    def call(self, x):
        encoder_input, decoder_input = x

        encoder_embedding = self.input_embedding(encoder_input)
        encoder_input = encoder_embedding + self.input_encoding

        encoder_output = self.encoders(encoder_input)

        decoder_embedding = self.output_embedding(decoder_input)
        decoder_input = decoder_embedding + self.output_encoding

        decoder_output = self.decoders([decoder_input, encoder_output])
        return self.output_probs(decoder_output)

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()
        self.d_model = float(d_model)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        # lr = d_model^(-0.5) * min(step^(-0.5), step*warmup_steps^(-1.5))
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)



if __name__ == '__main__':
    # https://www.tensorflow.org/text/tutorials/transformer?hl=en#:~:text=This%20setup%20is%20called%20%22Teacher%20forcing%22
    # https://datascience.stackexchange.com/questions/104179/is-the-transformer-decoder-an-autoregressive-model
    # https://ai.stackexchange.com/questions/40140/how-is-the-next-token-predicted-in-transformers
    transformer = Transformer(input_vocab_size=100)

    # tokenizer = { '<START>': 0, 'It': 1, 'is': 2, 'in': 3, 'this': 4, 'spirit': 5, '<END>': 99 }

    # Autoregressive inference
    sequences_input = tf.constant([[0,1,2,3,4]])
    sequences_output = tf.constant([[0]])
    for _ in range(3):
        probs = transformer([sequences_input, sequences_output])
        next_token = tf.argmax(probs[:, -1, :], axis=-1, output_type=sequences_output.dtype)
        sequences_output = tf.concat([sequences_output, next_token[tf.newaxis]], axis=-1)
        print(sequences_output)

    # Teacher forcing
    sequences_input = tf.constant([[0,1,2,3,4]])
    sequences_output = tf.constant([[0,1,2,3,4]])
    sequences_labels = tf.constant([[1,2,3,4,5]])
    transformer.fit([sequences_input, sequences_output], sequences_labels)
