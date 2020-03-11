import tensorflow as tf


class Encoder(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.resnet = tf.keras.applications.resnet_v2.ResNet101V2(include_top=False, weights="imagenet")
        # average pooling to resize to 14 x 14 ?
        for layer in self.resnet.layers:
            layer.trainable = False

    def call(self, inp):
        x = tf.keras.applications.resnet_v2.preprocess_input(inp)
        x = self.resnet(x)
        x = tf.reshape(x, shape=(x.shape[0], -1, x.shape[3]))
        return x


class EncoderGeneratorAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.encoder_attention = tf.keras.layers.Dense(units)
        self.generator_attention = tf.keras.layers.Dense(units)
        self.relu = tf.keras.layers.ReLU()
        self.full_attention = tf.keras.layers.Dense(1)
        self.softmax = tf.keras.layers.Softmax()

    def call(self, encoder_output, carry_state):
        attention_1 = self.encoder_attention(encoder_output)
        attention_2 = self.generator_attention(carry_state)
        attention_alpha = self.relu(attention_1 + tf.expand_dims(attention_2, axis=1))
        attention_alpha = self.full_attention(attention_alpha)
        attention_alpha = tf.squeeze(attention_alpha, axis=2)
        attention_alpha = self.softmax(attention_alpha)
        attention_weighted_encoding = tf.reduce_sum((encoder_output * tf.expand_dims(attention_alpha, axis=2)), axis=1)
        return attention_weighted_encoding, attention_alpha


class Generator(tf.keras.Model):
    def __init__(self, vocab_size, embedding_units, attention_units, lstm_units, encoder_units, lstm_dropout):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_units = embedding_units
        self.attention_units = attention_units
        self.lstm_units = lstm_units
        self.encoder_units = encoder_units
        self.lstm_dropout = lstm_dropout

        self.attention = EncoderGeneratorAttention(units=self.attention_units)
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=self.embedding_units)
        self.lstm = tf.keras.layers.LSTMCell(units=lstm_units, activation="tanh")
        self.dense_init_carry_state = tf.keras.layers.Dense(units=lstm_units)
        self.dense_init_memory_state = tf.keras.layers.Dense(units=lstm_units)
        self.dense_f_beta = tf.keras.layers.Dense(units=encoder_units, activation="sigmoid")
        self.dense_lstm_output = tf.keras.layers.Dense(units=self.vocab_size, activation="softmax")
        self.dropout = tf.keras.layers.Dropout(rate=0.5)

    def call(self, encoder_output, sequences, sequence_lengths, training=False):
        batch_size, num_pixels, encoder_size = encoder_output.shape

        sorted_indices = tf.argsort(sequence_lengths, direction="DESCENDING")
        encoder_output = tf.gather(encoder_output, indices=sorted_indices)
        sequences = tf.gather(sequences, indices=sorted_indices)
        sequence_lengths = tf.gather(sequence_lengths, indices=sorted_indices)

        iteration_lengths = (sequence_lengths - 1).numpy()
        predictions = tf.Variable(tf.zeros((batch_size, max(iteration_lengths), self.vocab_size)),
                                  name="predictions", trainable=False, dtype=tf.float32)
        attention_alphas = tf.Variable(tf.zeros((batch_size, max(iteration_lengths), num_pixels)),
                                       name="attention_alphas", trainable=False, dtype=tf.float32)

        embeddings = self.embedding(sequences)
        memory_state, carry_state = self.init_lstm_states(encoder_output)

        for t in range(max(iteration_lengths)):
            batch_size_t = sum([l > t for l in iteration_lengths])
            attention_weighted_encoding, attention_alpha = self.attention(encoder_output[:batch_size_t],
                                                                          carry_state[:batch_size_t])
            beta_gate = self.dense_f_beta(carry_state[:batch_size_t])
            attention_weighted_encoding *= beta_gate
            lstm_inputs = tf.concat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], axis=1)
            _, (memory_state, carry_state) = self.lstm(lstm_inputs,
                                                       [memory_state[:batch_size_t], carry_state[:batch_size_t]])
            prediction = self.dense_lstm_output(self.dropout(carry_state, training=training))
            predictions[:batch_size_t, t, :].assign(prediction)
            attention_alphas[:batch_size_t, t, :].assign(attention_alpha)

        return predictions, attention_alphas, sequences, iteration_lengths, sorted_indices

    def init_lstm_states(self, encoder_output):
        mean_encoder_output = tf.reduce_mean(encoder_output, axis=1)
        memory_state = self.dense_init_memory_state(mean_encoder_output)
        carry_state = self.dense_init_carry_state(mean_encoder_output)
        return memory_state, carry_state


class Discriminator(tf.keras.Model):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=256)
        self.gru1 = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(units=512, activation="tanh", return_sequences=True))
        self.gru2 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=512, activation="tanh"))
        self.dense1 = tf.keras.layers.Dense(units=2048, activation="relu")
        self.dropout = tf.keras.layers.Dropout(rate=0.2)
        self.dense2 = tf.keras.layers.Dense(units=1, activation="sigmoid")

    def call(self, inp, training=False):
        x = self.embedding(inp)
        x = self.gru1(x)
        x = self.gru2(x)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        return x
