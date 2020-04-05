import tensorflow as tf


class InverseSigmoidDecay(tf.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_rate, k, name="inverse_sigmoid_decay"):
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


class Constant(tf.optimizers.schedules.LearningRateSchedule):
    def __init__(self, rate, name="constant"):
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
