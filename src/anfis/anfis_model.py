from tensorflow.keras.models import Model
import tensorflow as tf
from src.anfis.anfis_layers import FuzzificationLayer, RulesLayer, SumNormalisationLayer, DefuzzificationLayer


@tf.function
def backward_pass(model, inputs, outputs):
    with tf.GradientTape() as tape:
        result = model(inputs)
        loss_result = model.loss(outputs, result)

        # Compute gradients
        trainable_vars = model.trainable_variables
        gradients = tape.gradient(loss_result, trainable_vars)

    # Update weights
    model.optimizer.apply_gradients(zip(gradients, trainable_vars))

    # Set metrics
    # model.compiled_metrics.update_state(outputs, result)
    #
    # return {m.name: m.result() for m in model.metrics}


@tf.function
def forward_step(models, inputs, outputs, learning_rate):

    weights = {}
    for m_key in models:
        weights[m_key] = models[m_key](inputs)

    wsr = weights["forward"]
    _units = models["anfis"].layers[4].units
    _vals = models["anfis"].layers[4].input_values

    padding = tf.constant([[0, 0], [0, 1]])
    xin = tf.pad(inputs, paddings=padding, mode="CONSTANT", constant_values=1.0)
    target = tf.matmul(tf.expand_dims(wsr, axis=-1), tf.expand_dims(xin, axis=-1), transpose_b=True)

    result = tf.linalg.lstsq(
        tf.reshape(target, shape=[1, -1, _units*_vals]),
        tf.expand_dims(outputs, axis=0),
        l2_regularizer=tf.constant(5e-2)
    )

    result = tf.reshape(result, shape=[-1, _units, _vals])
    collective_result = tf.reduce_mean(result, axis=0, keepdims=False)
    model_variable = models["anfis"].non_trainable_variables[0]
    delta = tf.subtract(collective_result, model_variable)*learning_rate
    result_delta = tf.add(models["anfis"].layers[4].p, delta)
    models["anfis"].layers[4].p.assign(result_delta)

    return result


class AnfisGD(tf.keras.Model):

    def __init__(self, parameters_sets_count, **kwargs):
        super(AnfisGD, self).__init__(**kwargs)
        self.parameters_sets_count = parameters_sets_count

        self.f1 = FuzzificationLayer(fuzzy_sets_count=self.parameters_sets_count)
        self.f2 = RulesLayer(fuzzy_sets_count=self.parameters_sets_count)
        self.f3 = SumNormalisationLayer()
        self.f4 = DefuzzificationLayer(summation_enabled=True, hybrid=False)

        self.counter = 0
        self.learning_rate = 1e-2

    def call(self, inputs, **kwargs):
        x = self.f1(inputs)
        #x = self.f2(x)
        #x = self.f3(x)
        #x = self.f4([inputs, x])
        return x

    def anfis_forward(self, inputs, **kwargs):
        x = self.f1(inputs)
        x = self.f2(x)
        x = self.f3(x)
        return x

    # UNCOMMENT IN TF 2.2 (NOT YET ON CONDA)
    # def train_step(self, data): # TODO ADD FORWARD PASS FOR ANFIS HYBRID
    #
    #     x, y = data
    #
    #     #--------------
    #     # GD Forward Pass
    #     #--------------
    #
    #     if True:
    #
    #         wsr = self.anfis_forward(x)
    #         _units = self.f4.units
    #         _vals = self.f4.input_values
    #
    #         padding = tf.constant([[0, 0], [0, 1]])
    #         xin = tf.pad(x, paddings=padding, mode="CONSTANT", constant_values=1.0)
    #         target = tf.matmul(tf.expand_dims(wsr, axis=-1), tf.expand_dims(xin, axis=-1), transpose_b=True)
    #
    #         result = tf.linalg.lstsq(
    #             tf.reshape(target, shape=[1, -1, _units * _vals]),
    #             tf.expand_dims(y, axis=0),
    #             l2_regularizer=tf.constant(5e-2)
    #         )
    #
    #         result = tf.reshape(result, shape=[-1, _units, _vals])
    #         collective_result = tf.reduce_mean(result, axis=0, keepdims=False)
    #         model_variable = self.f4.p
    #         delta = tf.subtract(collective_result, model_variable) * self.learning_rate
    #         result_delta = tf.add(self.f4.p, delta)
    #         self.f4.p.assign(result_delta)
    #
    #     # --------------
    #     # GD Backward Pass
    #     # --------------
    #     if True:
    #         self.counter = 0
    #
    #         with tf.GradientTape() as tape:
    #             y_pred = self(x, training=True)  # Forward pass
    #             # Compute the loss value
    #             # (the loss function is configured in `compile()`)
    #             loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
    #
    #         # Compute gradients
    #         trainable_vars = [self.f1.a, self.f1.b, self.f1.c] # Bell equation only
    #         gradients = tape.gradient(loss, trainable_vars)
    #         # Update weights
    #         self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    #         # Update metrics (includes the metric that tracks the loss)
    #         self.compiled_metrics.update_state(y, y_pred)
    #
    #     # Return a dict mapping metric names to current value
    #     return {m.name: m.result() for m in self.metrics}