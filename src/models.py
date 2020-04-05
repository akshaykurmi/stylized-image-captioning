import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class Encoder(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.resnet = tf.keras.applications.resnet_v2.ResNet101V2(include_top=False, weights="imagenet")
        for layer in self.resnet.layers:
            layer.trainable = False

    def call(self, inp):
        x = tf.keras.applications.resnet_v2.preprocess_input(inp)
        x = self.resnet(x)
        return x

    @property
    def output_shape(self):
        return self.resnet.layers[-1].output_shape


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
    def __init__(self, vocab_size, embedding_units, attention_units, lstm_units, encoder_units, lstm_dropout, z_units):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_units = embedding_units
        self.attention_units = attention_units
        self.lstm_units = lstm_units
        self.encoder_units = encoder_units
        self.lstm_dropout = lstm_dropout
        self.z_units = z_units

        self.z_distribution = tfp.distributions.Normal(loc=tf.zeros(self.z_units), scale=tf.ones(self.z_units))

        self.attention = EncoderGeneratorAttention(units=self.attention_units)
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=self.embedding_units)
        self.lstm = tf.keras.layers.LSTMCell(units=lstm_units, activation="tanh")
        self.dense_init_carry_state = tf.keras.layers.Dense(units=lstm_units)
        self.dense_init_memory_state = tf.keras.layers.Dense(units=lstm_units)
        self.dense_f_beta = tf.keras.layers.Dense(units=encoder_units, activation="sigmoid")
        self.dense_lstm_output = tf.keras.layers.Dense(units=self.vocab_size, activation="linear")
        self.dropout = tf.keras.layers.Dropout(rate=0.5)

        self.forward(tf.ones((3, 10, 10, 2048)), tf.ones((3, 10)), "deterministic", 1, False)  # initialize weights

    def call(self, encoder_output, sequences_t, memory_state, carry_state, training):
        embeddings = self.embedding(sequences_t)
        attention_weighted_encoding, attention_alpha = self.attention(encoder_output, carry_state)
        beta_gate = self.dense_f_beta(carry_state)
        attention_weighted_encoding *= beta_gate
        lstm_inputs = tf.concat([embeddings, attention_weighted_encoding], axis=1)
        _, (memory_state, carry_state) = self.lstm(lstm_inputs, [memory_state, carry_state])
        prediction = self.dense_lstm_output(self.dropout(carry_state, training=training))
        return prediction, attention_alpha, memory_state, carry_state

    def forward(self, encoder_output, sequences, mode, teacher_forcing_rate, training, z=None):
        if mode not in ["stochastic", "deterministic"]:
            raise ValueError(f"Mode must be one of - stochastic, deterministic")

        batch_size = encoder_output.shape[0]
        sequence_lengths = tf.cast(tf.reduce_sum(tf.sign(tf.abs(sequences)), 1), tf.int32)
        sort_indices = tf.argsort(sequence_lengths, direction="DESCENDING")
        sequence_lengths = tf.gather(sequence_lengths, sort_indices, axis=0) - 1
        encoder_output = tf.gather(encoder_output, sort_indices, axis=0)
        sequences = tf.gather(sequences, sort_indices, axis=0)
        max_seq_len = tf.reduce_max(sequence_lengths)

        encoder_output = self._reshape_encoder_output(encoder_output)
        memory_state, carry_state = self.init_lstm_states(encoder_output, z)
        predictions = tf.TensorArray(tf.float32, size=max_seq_len, clear_after_read=False,
                                     element_shape=(batch_size, self.vocab_size))
        attention_alphas = tf.TensorArray(tf.float32, size=max_seq_len, clear_after_read=False,
                                          element_shape=(batch_size, encoder_output.shape[1]))
        for t in range(max_seq_len):
            batch_size_t = tf.reduce_sum(tf.cast(sequence_lengths > t, dtype=tf.int32))
            sequences_t = sequences[:, t]
            if t > 0 and np.random.uniform() > teacher_forcing_rate:
                if mode == "deterministic":
                    sequences_t = tf.argmax(predictions.read(t - 1), axis=1, output_type=tf.int64)
                else:
                    sequences_t = tfp.distributions.Categorical(probs=predictions.read(t - 1),
                                                                dtype=tf.int64).sample()
            prediction, attention_alpha, memory_state_t, carry_state_t = self.call(
                encoder_output[:batch_size_t], sequences_t[:batch_size_t], memory_state[:batch_size_t],
                carry_state[:batch_size_t], training=training)
            memory_state_t = tf.pad(memory_state_t, [[0, batch_size - batch_size_t], [0, 0]])
            memory_state = tf.ensure_shape(memory_state_t, (batch_size, self.lstm_units))
            carry_state_t = tf.pad(carry_state_t, [[0, batch_size - batch_size_t], [0, 0]])
            carry_state = tf.ensure_shape(carry_state_t, (batch_size, self.lstm_units))
            prediction = tf.pad(prediction, [[0, batch_size - batch_size_t], [0, 0]])
            attention_alpha = tf.pad(attention_alpha, [[0, batch_size - batch_size_t], [0, 0]])
            predictions = predictions.write(t, prediction)
            attention_alphas = attention_alphas.write(t, attention_alpha)
        return (tf.transpose(predictions.stack(), (1, 0, 2)),
                tf.transpose(attention_alphas.stack(), (1, 0, 2)), sort_indices)

    def sample(self, encoder_output, initial_sequence, sequence_length, mode, n_samples, training, z=None):
        if mode not in ["stochastic", "deterministic"]:
            raise ValueError(f"Mode must be one of - stochastic, deterministic")

        initial_sequence_length = initial_sequence.shape[1]
        initial_sequence = tf.split(initial_sequence, num_or_size_splits=initial_sequence_length, axis=1)
        initial_sequence = [tf.cast(tf.squeeze(s, axis=1), dtype=tf.int64) for s in initial_sequence]

        initial_probabilities = []
        encoder_output = self._reshape_encoder_output(encoder_output)
        initial_memory_state, initial_carry_state = self.init_lstm_states(encoder_output, z)
        for t in range(initial_sequence_length):
            prediction, _, initial_memory_state, initial_carry_state = self.call(
                encoder_output, initial_sequence[t], initial_memory_state, initial_carry_state, training=training
            )
            initial_probabilities.append(prediction)
        samples = []
        sample_probabilities = []
        for n in range(n_samples):
            sequence = [tf.identity(s) for s in initial_sequence]
            probabilities = [tf.identity(p) for p in initial_probabilities]
            memory_state, carry_state = tf.identity(initial_memory_state), tf.identity(initial_carry_state)
            for t in range(initial_sequence_length, sequence_length):
                prediction, _, memory_state, carry_state = self.call(encoder_output, sequence[t - 1],
                                                                     memory_state, carry_state, training=training)
                if mode == "deterministic":
                    sequence.append(tf.argmax(prediction, axis=1, output_type=tf.int64))
                else:
                    sequence.append(tfp.distributions.Categorical(probs=prediction, dtype=tf.int64).sample())
                probabilities.append(prediction)
            samples.append(tf.stack(sequence, axis=1))
            sample_probabilities.append(tf.stack(probabilities, axis=1))
        return samples, sample_probabilities

    def beam_search(self, encoder_output, sequence_length, beam_size, sos, z=None):
        batch_size = encoder_output.shape[0]

        encoder_output = self._reshape_encoder_output(encoder_output)
        encoder_output = tf.reshape(tf.tile(encoder_output, [1, beam_size, 1]),
                                    ((batch_size * beam_size), *encoder_output.shape[1:]))

        memory_state, carry_state = self.init_lstm_states(encoder_output, z)

        sequences = tf.constant(sos, shape=((batch_size * beam_size), 1), dtype=tf.int64)  # (8*5, 1)
        sequences_log_probs = tf.constant(0, shape=((batch_size * beam_size), 1), dtype=tf.float32)  # (8*5, 1)

        for t in range(sequence_length - 1):
            current_tokens = sequences[:, -1]
            current_probs = sequences_log_probs[:, -1]
            prediction, _, memory_state, carry_state = self.call(encoder_output, current_tokens,
                                                                 memory_state, carry_state, training=False)
            prediction = tf.math.log(prediction)
            probs_l1, indices_l1 = tf.math.top_k(prediction, k=beam_size)  # (8*5, 5)
            probs_l1 += tf.transpose(tf.reshape(
                tf.tile(current_probs, [beam_size]), (beam_size, batch_size * beam_size)
            ))
            probs_l1 = tf.reshape(probs_l1, (batch_size, beam_size * beam_size))  # (8, 5*5)
            indices_l1 = tf.cast(indices_l1, dtype=tf.int64)
            indices_l1 = tf.reshape(indices_l1, (batch_size, beam_size * beam_size))  # (8, 5*5)
            probs_l2, indices_l2 = tf.math.top_k(probs_l1, k=beam_size)  # (8, 5)

            next_tokens = tf.gather_nd(indices_l1, tf.stack([
                tf.repeat(tf.range(batch_size), beam_size, axis=0),
                tf.reshape(indices_l2, (-1,))
            ], axis=1))

            current_reordered_indices = tf.math.reduce_sum(tf.stack([
                tf.repeat(tf.range(batch_size), beam_size, axis=0) * beam_size,
                tf.reshape(tf.cast(indices_l2 / 5, dtype=tf.int32), (-1,))
            ], axis=1), axis=1)
            current_tokens = tf.gather(current_tokens, current_reordered_indices)
            memory_state = tf.gather(memory_state, current_reordered_indices, axis=0)
            carry_state = tf.gather(carry_state, current_reordered_indices, axis=0)

            sequences = tf.slice(sequences, [0, 0], [batch_size * beam_size, sequences.shape[1] - 1])
            sequences = tf.concat([
                sequences, tf.expand_dims(current_tokens, axis=1), tf.expand_dims(next_tokens, axis=1)
            ], axis=1)
            sequences_log_probs = tf.concat([
                sequences_log_probs, tf.reshape(probs_l2, (-1, 1))
            ], axis=1)

        sequences = tf.reshape(sequences, (batch_size, beam_size, -1))
        sequences_log_probs = tf.reshape(sequences_log_probs[:, -1], (batch_size, beam_size))
        return sequences, sequences_log_probs

    def init_lstm_states(self, encoder_output, z):
        if z is None:
            z = self.z_distribution.sample(sample_shape=(encoder_output.shape[0],))
        mean_encoder_output = tf.reduce_mean(encoder_output, axis=1)
        mean_encoder_output_with_z = tf.concat([mean_encoder_output, z], axis=1)
        memory_state = self.dense_init_memory_state(mean_encoder_output_with_z)
        carry_state = self.dense_init_carry_state(mean_encoder_output_with_z)
        return memory_state, carry_state

    @staticmethod
    def _reshape_encoder_output(encoder_output):
        return tf.reshape(encoder_output, shape=(encoder_output.shape[0], -1, encoder_output.shape[3]))


class Discriminator(tf.keras.Model):
    def __init__(self, vocab_size, embedding_units, lstm_units):
        super().__init__()
        self.pooling = tf.keras.layers.GlobalAveragePooling2D()
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_units)
        self.lstm1 = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=lstm_units, activation="tanh", return_sequences=True))
        self.lstm2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=lstm_units, activation="tanh"))
        self.dense1 = tf.keras.layers.Dense(units=1024, activation="relu")
        self.dropout = tf.keras.layers.Dropout(rate=0.2)
        self.dense2 = tf.keras.layers.Dense(units=1, activation="sigmoid")

        self.call(tf.ones((1, 10, 10, 2048)), tf.ones((1, 10)))  # initialize weights

    def call(self, encoder_output, sequences, training=False):
        encoder_output = self.pooling(encoder_output)
        embeddings = self.embedding(sequences)
        lstm1_out = self.lstm1(embeddings)
        lstm2_out = self.lstm2(lstm1_out)
        all_features = tf.concat([encoder_output, lstm2_out], axis=1)
        x = self.dense1(all_features)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        return x
