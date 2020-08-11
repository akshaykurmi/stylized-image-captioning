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
        attention_1 = self.encoder_attention(encoder_output)  # Transforms from (3, 100, 2048) to (3, 100, units)
        attention_2 = self.generator_attention(carry_state)  # Transforms from (3, lstm_units) to (3, units)
        attention_alpha = self.relu(attention_1 + tf.expand_dims(attention_2, axis=1))  # (3, 100, units)
        attention_alpha = self.full_attention(attention_alpha)  # (3, 100, 1)
        attention_alpha = tf.squeeze(attention_alpha, axis=2)  # (3, 100)
        attention_alpha = self.softmax(attention_alpha)  # (3, 100)
        attention_weighted_encoding = tf.reduce_sum((encoder_output * tf.expand_dims(attention_alpha, axis=2)), axis=1)
        return attention_weighted_encoding, attention_alpha # (3, 2048); (3, 100)

class SemanticDiscriminator(tf.keras.layers.Layer):
    def __init__(self, token_vocab_size, token_embedding_units, lstm_units):
        super().__init__()
        #self.stylize = stylize
        self.pooling = tf.keras.layers.GlobalAveragePooling2D()
        self.token_embedding = tf.keras.layers.Embedding(input_dim=token_vocab_size, output_dim=token_embedding_units,
                                                         mask_zero=True)
        #self.style_embedding = tf.keras.layers.Embedding(input_dim=style_vocab_size, output_dim=style_embedding_units)
        self.lstm1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            units=lstm_units, activation="tanh", return_sequences=True, dropout=0.2, recurrent_dropout=0.2
        ))
        self.lstm2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            units=lstm_units, activation="tanh", dropout=0.2, recurrent_dropout=0.2
        ))
        self.dense1 = tf.keras.layers.Dense(units=1024, activation="relu")
        self.dropout1 = tf.keras.layers.Dropout(rate=0.5)
        self.dense2 = tf.keras.layers.Dense(units=512, activation="relu")
        self.dense3 = tf.keras.layers.Dense(units=1, activation="sigmoid")

        self.call(tf.ones((3, 10, 10, 2048)), tf.ones((3, 10)), False)

    def call(self, encoder_output, sequences, training):
        encoder_output = self.pooling(encoder_output)
        token_embeddings = self.token_embedding(sequences)
        mask = self.token_embedding.compute_mask(sequences) #Needed to mask the token emb
        lstm1_out = self.lstm1(token_embeddings, mask=mask, training=training)
        lstm2_out = self.lstm2(lstm1_out, mask=mask, training=training)
        all_features = tf.concat([encoder_output, lstm2_out], axis=1)
        # if self.stylize:
        #     style_embeddings = self.style_embedding(styles)
        #     all_features = tf.concat([all_features, style_embeddings], axis=1)
        x = self.dense1(all_features)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dense3(x)
        return x


class StyleDiscriminator(tf.keras.layers.Layer):
    def __init__(self, token_vocab_size, style_vocab_size, token_embedding_units, style_embedding_units,
                 lstm_units):
        super().__init__()
        #self.stylize = stylize
        self.pooling = tf.keras.layers.GlobalAveragePooling2D()
        self.token_embedding = tf.keras.layers.Embedding(input_dim=token_vocab_size, output_dim=token_embedding_units,
                                                         mask_zero=True)
        self.style_embedding = tf.keras.layers.Embedding(input_dim=style_vocab_size, output_dim=style_embedding_units)
        self.lstm1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            units=lstm_units, activation="tanh", return_sequences=True, dropout=0.2, recurrent_dropout=0.2
        ))
        self.lstm2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            units=lstm_units, activation="tanh", dropout=0.2, recurrent_dropout=0.2
        ))
        self.dense1 = tf.keras.layers.Dense(units=1024, activation="relu")
        self.dropout1 = tf.keras.layers.Dropout(rate=0.5)
        self.dense2 = tf.keras.layers.Dense(units=512, activation="relu")
        self.dense3 = tf.keras.layers.Dense(units=1, activation="sigmoid")

        self.call(tf.ones((3, 10)), tf.ones((3,)), False)

    def call(self, sequences, styles, training):
        #encoder_output = self.pooling(encoder_output)
        token_embeddings = self.token_embedding(sequences)
        mask = self.token_embedding.compute_mask(sequences) #Needed to mask the token emb
        lstm1_out = self.lstm1(token_embeddings, mask=mask, training=training)
        lstm2_out = self.lstm2(lstm1_out, mask=mask, training=training)
        #all_features = tf.concat([encoder_output, lstm2_out], axis=1)
        #if self.stylize:
        style_embeddings = self.style_embedding(styles)
        all_features = tf.concat([lstm2_out, style_embeddings], axis=1)
        x = self.dense1(all_features)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

class Generator(tf.keras.Model):
    def __init__(self, token_vocab_size, style_vocab_size, token_embedding_units, style_embedding_units,
                 attention_units, lstm_units, encoder_units, lstm_dropout, z_units, stylize):
        super().__init__()
        self.token_vocab_size = token_vocab_size
        self.style_vocab_size = style_vocab_size
        self.token_embedding_units = token_embedding_units
        self.style_embedding_units = style_embedding_units
        self.attention_units = attention_units
        self.lstm_units = lstm_units
        self.encoder_units = encoder_units
        self.lstm_dropout = lstm_dropout
        self.z_units = z_units
        self.stylize = stylize

        self.z_distribution = tfp.distributions.Normal(loc=tf.zeros(self.z_units), scale=tf.ones(self.z_units))

        self.attention = EncoderGeneratorAttention(units=self.attention_units)
        self.token_embedding = tf.keras.layers.Embedding(input_dim=self.token_vocab_size,
                                                         output_dim=self.token_embedding_units)
        self.style_embedding = tf.keras.layers.Embedding(input_dim=self.style_vocab_size,
                                                         output_dim=self.style_embedding_units)
        self.lstm = tf.keras.layers.LSTMCell(units=lstm_units, activation="tanh")
        self.dense_init_carry_state = tf.keras.layers.Dense(units=lstm_units)
        self.dense_init_memory_state = tf.keras.layers.Dense(units=lstm_units)
        self.dense_f_beta = tf.keras.layers.Dense(units=encoder_units, activation="sigmoid")
        self.dense_lstm_output = tf.keras.layers.Dense(units=self.token_vocab_size, activation="linear")
        self.dropout = tf.keras.layers.Dropout(rate=self.lstm_dropout)

        self.forward(tf.ones((3, 10, 10, 2048)), tf.ones((3, 10)), tf.ones((3,)), "deterministic", 1, False)

    def call(self, encoder_output, sequences_t, style_embeddings, memory_state, carry_state, training):
        """
        Summary
        1. Takes encoder output and current lstm cell state (carry_state) and uses that to compute attention and
        attention_weighted_encoding and attention_alpha
        2. Takes token at timestep t, embeds it
        3. generates the logits of the next_token

        returns logits_t, attention_alpha, memory_state (h), carry_state(c)

        """
        token_embeddings = self.token_embedding(sequences_t) # (batch_size, token_embedding_units)
        attention_weighted_encoding, attention_alpha = self.attention(encoder_output, carry_state)
        # (3, 2048); (3, 100)
        beta_gate = self.dense_f_beta(carry_state)
        # transforms carry_state from (batch, lstm_units) to (batch, encoder_units) and sigmoids output
        attention_weighted_encoding *= beta_gate #Not sure
        lstm_inputs = tf.concat([token_embeddings, attention_weighted_encoding], axis=1)
        if self.stylize:
            lstm_inputs = tf.concat([style_embeddings, lstm_inputs], axis=1)
        _, (memory_state, carry_state) = self.lstm(lstm_inputs, [memory_state, carry_state]) #h, h, c = lstm(x, [h,c])
        logits_t = self.dense_lstm_output(self.dropout(carry_state, training=training))
        return logits_t, attention_alpha, memory_state, carry_state

    def forward(self, encoder_output, sequences, styles, mode, teacher_forcing_rate, training, z=None):
        if mode not in ["stochastic", "deterministic"]:
            raise ValueError(f"Mode must be one of - stochastic, deterministic")

        batch_size = encoder_output.shape[0]
        image_features_size = encoder_output.shape[1] * encoder_output.shape[2]

        sequence_lengths = tf.cast(tf.reduce_sum(tf.sign(tf.abs(sequences)), 1), tf.int32) - 1
        max_seq_len = tf.reduce_max(sequence_lengths)
        sort_indices = tf.argsort(sequence_lengths, direction="DESCENDING")
        sequence_lengths = tf.gather(sequence_lengths, sort_indices, axis=0)
        encoder_output = tf.gather(encoder_output, sort_indices, axis=0)
        sequences = tf.gather(sequences, sort_indices, axis=0)
        styles = tf.gather(styles, sort_indices, axis=0)

        encoder_output = self._reshape_encoder_output(encoder_output)
        style_embeddings = self.style_embedding(styles)
        memory_state, carry_state = self.init_lstm_states(encoder_output, style_embeddings, z)
        logits = tf.TensorArray(tf.float32, size=max_seq_len, clear_after_read=False,
                                element_shape=(batch_size, self.token_vocab_size))
        attention_alphas = tf.TensorArray(tf.float32, size=max_seq_len, clear_after_read=False,
                                          element_shape=(batch_size, image_features_size))
        for t in range(max_seq_len):
            batch_size_t = tf.reduce_sum(tf.cast(sequence_lengths > t, dtype=tf.int32))
            # number of ex in batch with length greater than t

            sequences_t = sequences[:, t]
            if t > 0 and np.random.uniform() > teacher_forcing_rate:
                if mode == "deterministic":
                    sequences_t = tf.argmax(logits.read(t - 1), axis=1, output_type=tf.int64)
                else:
                    sequences_t = tfp.distributions.Categorical(logits=logits.read(t - 1),
                                                                dtype=tf.int64).sample()
            logits_t, attention_alpha_t, memory_state_t, carry_state_t = self.call(
                encoder_output[:batch_size_t], sequences_t[:batch_size_t], style_embeddings[:batch_size_t],
                memory_state[:batch_size_t], carry_state[:batch_size_t], training=training
            )
            memory_state_t = tf.pad(memory_state_t, [[0, batch_size - batch_size_t], [0, 0]])
            memory_state = tf.ensure_shape(memory_state_t, (batch_size, self.lstm_units))
            carry_state_t = tf.pad(carry_state_t, [[0, batch_size - batch_size_t], [0, 0]])
            carry_state = tf.ensure_shape(carry_state_t, (batch_size, self.lstm_units))

            logits_t = tf.pad(logits_t, [[0, batch_size - batch_size_t], [0, 0]])
            attention_alpha_t = tf.pad(attention_alpha_t, [[0, batch_size - batch_size_t], [0, 0]])
            logits = logits.write(t, logits_t)
            attention_alphas = attention_alphas.write(t, attention_alpha_t)
        return (tf.transpose(logits.stack(), (1, 0, 2)),
                tf.transpose(attention_alphas.stack(), (1, 0, 2)), sort_indices)

    def sample(self, encoder_output, initial_sequence, styles, sequence_length, mode, n_samples, training,
               sos, eos, z=None):
        """
        Summary
        1. Takes an initial token sequence
        2. Generates the next token
        3. If deterministic, next token is the argmax. If stochastic, the next token is sampled

        Returns samples, sample_logits
        """
        if mode not in ["stochastic", "deterministic"]:
            raise ValueError(f"Mode must be one of - stochastic, deterministic")

        initial_sequence_length = initial_sequence.shape[1]
        initial_sequence = tf.split(initial_sequence, num_or_size_splits=initial_sequence_length, axis=1)
        initial_sequence = [tf.cast(tf.squeeze(s, axis=1), dtype=tf.int64) for s in initial_sequence]

        encoder_output = self._reshape_encoder_output(encoder_output)
        style_embeddings = self.style_embedding(styles)
        initial_memory_state, initial_carry_state = self.init_lstm_states(encoder_output, style_embeddings, z)

        initial_logits = []
        for t in range(initial_sequence_length):
            logits_t, _, initial_memory_state, initial_carry_state = self.call(
                encoder_output, initial_sequence[t], style_embeddings,
                initial_memory_state, initial_carry_state, training=training
            )
            initial_logits.append(logits_t)

        samples = []
        sample_logits = []
        for n in range(n_samples):
            sequence = [tf.identity(s) for s in initial_sequence] # tf.identity is like tensor.copy()
            logits = [tf.identity(l) for l in initial_logits]
            memory_state, carry_state = tf.identity(initial_memory_state), tf.identity(initial_carry_state)
            for t in range(initial_sequence_length, sequence_length):
                logits_t, _, memory_state, carry_state = self.call(encoder_output, sequence[t - 1], style_embeddings,
                                                                   memory_state, carry_state, training=training)
                if mode == "deterministic":
                    sequence.append(tf.argmax(logits_t, axis=1, output_type=tf.int64))
                else:
                    sequence.append(tfp.distributions.Categorical(logits=logits_t, dtype=tf.int64).sample())
                logits.append(logits_t)
            sequence = tf.stack(sequence, axis=1) # converts the list of tensors into a tensor
            logits = tf.stack(logits, axis=1)

            oh = tf.expand_dims(tf.math.log(
                tf.one_hot([sos] * logits.shape[0], depth=logits.shape[-1], dtype=tf.float32)
            ), axis=1)
            logits = tf.concat([oh, logits], axis=1)
            logits = logits[:, :-1, :]

            mask = self._get_mask(sequence, eos)
            samples.append(sequence * tf.cast(mask, dtype=tf.int64))
            sample_logits.append(logits * tf.cast(
                tf.repeat(tf.expand_dims(mask, axis=2), self.token_vocab_size, axis=2),
                dtype=tf.float32
            ))
        return samples, sample_logits

    def beam_search(self, encoder_output, styles, sequence_length, beam_size, sos, eos, z=None):
        batch_size = encoder_output.shape[0]

        encoder_output = self._reshape_encoder_output(encoder_output)
        encoder_output = tf.reshape(tf.tile(encoder_output, [1, beam_size, 1]),
                                    ((batch_size * beam_size), *encoder_output.shape[1:]))
        style_embeddings = self.style_embedding(styles)
        style_embeddings = tf.reshape(tf.tile(style_embeddings, [1, beam_size]),
                                      ((batch_size * beam_size), *style_embeddings.shape[1:]))

        memory_state, carry_state = self.init_lstm_states(encoder_output, style_embeddings, z)

        sequences = tf.constant(sos, shape=((batch_size * beam_size), 1), dtype=tf.int64)  # (8*5, 1)
        sequences_logits = tf.constant(0, shape=((batch_size * beam_size), 1), dtype=tf.float32)  # (8*5, 1)

        for t in range(sequence_length - 1):
            current_tokens = sequences[:, -1]
            current_logits = sequences_logits[:, -1]
            logits_t, _, memory_state, carry_state = self.call(encoder_output, current_tokens, style_embeddings,
                                                               memory_state, carry_state, training=False)
            logits_t = tf.math.log(tf.nn.softmax(logits_t))
            logits_l1, indices_l1 = tf.math.top_k(logits_t, k=beam_size)  # (8*5, 5)
            logits_l1 += tf.transpose(tf.reshape(
                tf.tile(current_logits, [beam_size]), (beam_size, batch_size * beam_size)
            ))
            logits_l1 = tf.reshape(logits_l1, (batch_size, beam_size * beam_size))  # (8, 5*5)
            indices_l1 = tf.cast(indices_l1, dtype=tf.int64)
            indices_l1 = tf.reshape(indices_l1, (batch_size, beam_size * beam_size))  # (8, 5*5)
            logits_l2, indices_l2 = tf.math.top_k(logits_l1, k=beam_size)  # (8, 5)

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
            sequences_logits = tf.concat([
                sequences_logits, tf.reshape(logits_l2, (-1, 1))
            ], axis=1)

        mask = self._get_mask(sequences, eos)
        sequences = sequences * tf.cast(mask, dtype=tf.int64)
        sequences = tf.reshape(sequences, (batch_size, beam_size, -1))
        sequences_logits = tf.reshape(sequences_logits[:, -1], (batch_size, beam_size))
        return sequences, sequences_logits

    def init_lstm_states(self, encoder_output, style_embeddings, z):
        if z is None:
            z = self.z_distribution.sample(sample_shape=(encoder_output.shape[0],))
        mean_encoder_output = tf.reduce_mean(encoder_output, axis=1)
        concatenated = tf.concat([mean_encoder_output, z], axis=1)
        if self.stylize:
            concatenated = tf.concat([style_embeddings, concatenated], axis=1)
        memory_state = self.dense_init_memory_state(concatenated)
        carry_state = self.dense_init_carry_state(concatenated)
        return memory_state, carry_state

    @staticmethod
    def _reshape_encoder_output(encoder_output):
        return tf.reshape(encoder_output, shape=(encoder_output.shape[0], -1, encoder_output.shape[3]))

    @staticmethod
    def _get_mask(sequence, eos):
        batch_size, sequence_length = sequence.shape[0], sequence.shape[1]
        mask = tf.tensor_scatter_nd_update(
            sequence, tf.stack([tf.range(batch_size), tf.constant(sequence_length - 1, shape=(batch_size,))], axis=1),
            tf.constant(eos, dtype=tf.int64, shape=(batch_size,))
        )
        mask = tf.broadcast_to(tf.expand_dims(tf.argmax(tf.cast(mask == eos, tf.int64), axis=1), axis=1),
                               (batch_size, sequence_length))
        mask = mask >= tf.broadcast_to(tf.range(sequence_length, dtype=tf.int64), (batch_size, sequence_length))
        return mask


class Discriminator(tf.keras.Model):
    def __init__(self, token_vocab_size, style_vocab_size, token_embedding_units, style_embedding_units,
                 lstm_units, stylize):
        super().__init__()
        self.stylize = stylize
        self.pooling = tf.keras.layers.GlobalAveragePooling2D()
        self.token_embedding = tf.keras.layers.Embedding(input_dim=token_vocab_size, output_dim=token_embedding_units,
                                                         mask_zero=True)
        self.style_embedding = tf.keras.layers.Embedding(input_dim=style_vocab_size, output_dim=style_embedding_units)
        self.lstm1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            units=lstm_units, activation="tanh", return_sequences=True, dropout=0.2, recurrent_dropout=0.2
        ))
        self.lstm2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            units=lstm_units, activation="tanh", dropout=0.2, recurrent_dropout=0.2
        ))
        self.dense1 = tf.keras.layers.Dense(units=1024, activation="relu")
        self.dropout1 = tf.keras.layers.Dropout(rate=0.5)
        self.dense2 = tf.keras.layers.Dense(units=512, activation="relu")
        self.dense3 = tf.keras.layers.Dense(units=1, activation="sigmoid")

        self.call(tf.ones((3, 10, 10, 2048)), tf.ones((3, 10)), tf.ones((3,)), False)

    def call(self, encoder_output, sequences, styles, training):
        encoder_output = self.pooling(encoder_output)
        token_embeddings = self.token_embedding(sequences)
        mask = self.token_embedding.compute_mask(sequences) #Needed to mask the token emb
        lstm1_out = self.lstm1(token_embeddings, mask=mask, training=training)
        lstm2_out = self.lstm2(lstm1_out, mask=mask, training=training)
        all_features = tf.concat([encoder_output, lstm2_out], axis=1)
        if self.stylize:
            style_embeddings = self.style_embedding(styles)
            all_features = tf.concat([all_features, style_embeddings], axis=1)
        x = self.dense1(all_features)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

class Discriminator2(tf.keras.Model):
    def __init__(self, token_vocab_size, style_vocab_size, token_embedding_units, style_embedding_units,
                 lstm_units, stylize, alpha=0.5):
        super().__init__()
        self.stylize = stylize
        self.alpha = alpha
        self.semantic_discriminator = SemanticDiscriminator(token_vocab_size, token_embedding_units, lstm_units)
        self.style_discriminator = StyleDiscriminator(token_vocab_size, style_vocab_size, token_embedding_units, style_embedding_units, lstm_units)

        self.call(tf.ones((3, 10, 10, 2048)), tf.ones((3, 10)), tf.ones((3,)), False)

    def call(self, encoder_output, sequences, styles, training):

        semantic_score = self.semantic_discriminator(encoder_output, sequences, training)
        style_score = self.style_discriminator(sequences, styles, training)

        return semantic_score*self.alpha + style_score*(1-self.alpha)
