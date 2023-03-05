import tensorflow as tf
from positional_encoding import positional_encoding
from encoder import Encoder
from decoder import Decoder


class Transformer(tf.keras.Model):

    def __init__(self, input_dim, output_dim, vocab_size, embedding_dim=512):
        super().__init__()
        self.encoder_input = tf.keras.layers.Input(shape=(input_dim,))
        self.decoder_input = tf.keras.layers.Input(shape=(output_dim,))

        # Embeddings
        self.input_embedding = tf.keras.layers.Embedding(input_dim, embedding_dim)
        self.output_embedding = tf.keras.layers.Embedding(output_dim, embedding_dim)

        # Positional encodings
        self.input_encoding = positional_encoding(input_dim, embedding_dim)
        self.output_encoding = positional_encoding(output_dim, embedding_dim)

        # Encoders
        self.encoders = Encoder(nb_encoder=6)

        # Decoders
        self.decoders = Decoder(nb_decoder=6)

        # Output
        self.output_probs = tf.keras.layers.Dense(vocab_size, activation='softmax')

        # Hyperparameters
        self.scce = tf.keras.losses.SparseCategoricalCrossentropy()
        self.learning_rate = CustomSchedule(d_model=embedding_dim)
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        self.compile(loss=self.scce, optimizer=self.optimizer)

    def call(self, x):
        encoder_input, decoder_input = x

        encoder_embedding = self.input_embedding(encoder_input)
        encoder_input = encoder_embedding + self.input_encoding

        encoder_output = self.encoders(encoder_input)

        decoder_embedding = self.output_embedding(decoder_input)
        decoder_input = decoder_embedding + self.output_encoding

        decoder_output = self.decoders([decoder_input, encoder_output])
        return self.output_probs(decoder_output)

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self.call(x)
            loss = self.scce(y, y_pred)

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return { "loss": loss }

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()
        self.d_model = float(d_model)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        # lr = d_model^(-0.5) * min(step^(-0.5), step*warmup_steps^(-1.5))
        arg1 = tf.math.rsqrt(tf.cast(step, dtype=tf.float32))
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)



if __name__ == '__main__':
    transformer = Transformer(5, 6, 100)

    # Suppose { 0: '<START>', 6: '<END>'}
    sequences_input = tf.constant([[0,1,2,3,4]])
    sequences_output = tf.constant([[0,1,2,3,4,5]])
    sequences_labels = tf.constant([[1,2,3,4,5,6]])

    output = transformer([sequences_input, sequences_output])
    print(tf.argmax(output, axis=-1))

    transformer.fit([sequences_input, sequences_output], sequences_labels)
