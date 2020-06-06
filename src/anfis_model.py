import tensorflow as tf


@tf.function
def train_step(model, forward, inputs, outputs):

    wsr = forward(inputs)
    _units = model.layers[4].units
    _vals = model.layers[4].input_values

    padding = tf.constant([[0, 0], [0, 1]])
    xin = tf.pad(inputs, paddings=padding, mode="CONSTANT", constant_values=1.0)
    target = tf.matmul(tf.expand_dims(wsr, axis=-1), tf.expand_dims(xin,axis=-1), transpose_b=True)

    result = tf.linalg.lstsq(tf.reshape(target, shape=[-1, 1, _units*_vals]), tf.expand_dims(outputs, axis=-1))
    collective_result = tf.reduce_mean(tf.reshape(result, shape=[-1, _units, _vals]), axis=0, keepdims=False)
    model.layers[4].p.assign(collective_result)

    return collective_result
