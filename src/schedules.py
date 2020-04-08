import tensorflow as tf


class ConstantSchedule(tf.optimizers.schedules.LearningRateSchedule):
    def __init__(self, rate, name="constant_schedule"):
        super().__init__()
        self.rate = rate
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name):
            return self.rate

    def get_config(self):
        return {
            "rate": self.rate,
            "name": self.name
        }


class LinearSchedule(tf.optimizers.schedules.LearningRateSchedule):
    def __init__(self, epsilon, k, c, name="linear_schedule"):
        super().__init__()
        self.epsilon = epsilon
        self.k = k
        self.c = c
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name):
            return tf.math.maximum(self.epsilon, (self.k - self.c * step))

    def get_config(self):
        return {
            "epsilon": self.epsilon,
            "k": self.k,
            "c": self.c,
            "name": self.name
        }


class InverseSigmoidSchedule(tf.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_rate, k, name="inverse_sigmoid_schedule"):
        super().__init__()
        self.initial_rate = initial_rate
        self.k = k
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name):
            decay = self.k / (self.k + tf.math.exp(step / self.k))
            return self.initial_rate * decay

    def get_config(self):
        return {
            "initial_rate": self.initial_rate,
            "k": self.k,
            "name": self.name
        }
