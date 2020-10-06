from tensorflow.keras import Model
import tensorflow as tf
import tensorflow_probability as tfp


class GaussianLayer(Model):
    def __init__(self, **kwargs):
        super(GaussianLayer, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        mu, log_sig = inputs

        log_sig_clip = tf.clip_by_value(log_sig, -20, 2)
        sig = tf.exp(log_sig_clip)

        distribution = tfp.distributions.Normal(mu, sig)
        output = distribution.sample()
        actions = tf.tanh(output)

        return actions, \
            distribution.log_prob(output) - \
            tf.reduce_sum(tf.math.log(1 - actions ** 2 + 1e-6), axis=1, keepdims=True), \
            tf.stop_gradient(tf.keras.backend.abs(actions - tf.tanh(mu)))
