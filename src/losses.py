import tensorflow as tf


class GeneratorMLELoss:
    def __call__(self, captions, logits, attention_alphas, dsa_lambda):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(captions, logits)
        mask = tf.cast(tf.sign(tf.abs(captions)), tf.float32)
        loss *= mask
        loss = tf.reduce_sum(loss, axis=1)
        loss /= tf.reduce_sum(mask, axis=1)
        loss = tf.reduce_mean(loss)
        loss += dsa_lambda * tf.reduce_mean((1 - tf.reduce_sum(attention_alphas, axis=1)) ** 2)
        return loss


class PolicyGradientLoss:
    def __call__(self, captions, logits, rewards):
        probabilities = tf.nn.softmax(logits)
        mask = tf.cast(tf.sign(tf.abs(captions)), tf.float32)
        probabilities = tf.reshape(probabilities, shape=(-1, probabilities.shape[-1]))
        captions = tf.reshape(captions, shape=(-1,))
        rewards = tf.reshape(rewards, shape=(-1,))
        mask = tf.reshape(mask, shape=(-1,))
        indices = tf.stack([tf.range(captions.shape[0], dtype=tf.int64), captions], axis=1)
        probabilities = tf.gather_nd(probabilities, indices)
        loss = -tf.reduce_sum(tf.math.log(probabilities) * rewards * mask)
        return loss
