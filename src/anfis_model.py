import tensorflow as tf


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
    # model.compiled_metrics.update_state(outputs, result)
    #
    # return {m.name: m.result() for m in model.metrics}


@tf.function
def forward_step(model, forward, inputs, outputs, learning_rate, dummy):

    # dmy0 = dummy[0](inputs)
    # dmy1 = dummy[1](inputs)
    # dmy2 = dummy[2](inputs)
    # dmy3 = dummy[3](inputs)
    # dmy4 = dummy[4](inputs)

    wsr = forward(inputs)
    _units = model.layers[4].units
    _vals = model.layers[4].input_values

    padding = tf.constant([[0, 0], [0, 1]])
    xin = tf.pad(inputs, paddings=padding, mode="CONSTANT", constant_values=1.0)
    target = tf.matmul(tf.expand_dims(wsr, axis=-1), tf.expand_dims(xin, axis=-1), transpose_b=True)

    result = tf.linalg.lstsq(
        tf.reshape(target, shape=[-1, 1, _units*_vals]),
        tf.expand_dims(outputs, axis=-1),
        l2_regularizer=tf.constant(0.8)
    )

    collective_result = tf.reduce_mean(tf.reshape(result, shape=[-1, _units, _vals]), axis=0, keepdims=False)
    model_variable = model.non_trainable_variables[0]
    delta = tf.subtract(collective_result, model_variable)*learning_rate
    model.layers[4].p.assign_add(delta)

    return collective_result


def train_anfis(model, forward, inputs, outputs, epochs, batch_size, learning_rate, dummy):

    input_size = inputs.shape[0]
    indices = tf.range(start=0, limit=input_size, dtype=tf.int32)

    batches_number = input_size // batch_size
    rest = input_size % batch_size
    split = [batch_size] * batches_number + ([rest] if rest > 0 else [])

    for i in range(epochs):
        print("epoch: ", i)
        #Shuffle inputs before batching
        shuffled_indices = tf.random.shuffle(indices)

        shuffled_inputs  = tf.gather(inputs, shuffled_indices)
        shuffled_outputs = tf.gather(outputs, shuffled_indices)

        #Split to batches
        batched_inputs = tf.split(shuffled_inputs, split, axis=0)
        batched_outputs = tf.split(shuffled_outputs, split, axis=0)
        num_batches = len(batched_inputs)

        for j in range(num_batches):
            #if j % 2 == 0:
            forward_step(model, forward, batched_inputs[j], batched_outputs[j],
                         learning_rate=learning_rate, dummy=dummy)
            #else:
            backward_pass(model, batched_inputs[j], batched_outputs[j])


def train_anfis_ng(model, forward, inputs, outputs, epochs, batch_size):
    input_size = inputs.shape[0]
    indices = tf.range(start=0, limit=input_size, dtype=tf.int32)

    batches_number = input_size // batch_size
    rest = input_size % batch_size
    split = [batch_size] * batches_number + ([rest] if rest > 0 else [])

    for i in range(epochs):
        print("epoch: ", i)
        # Shuffle inputs before batching
        shuffled_indices = tf.random.shuffle(indices)

        shuffled_inputs = tf.gather(inputs, shuffled_indices)
        shuffled_outputs = tf.gather(outputs, shuffled_indices)

        # Split to batches
        batched_inputs = tf.split(shuffled_inputs, split, axis=0)
        batched_outputs = tf.split(shuffled_outputs, split, axis=0)
        num_batches = len(batched_inputs)

        for j in range(num_batches):
            forward_step(model, forward, batched_inputs[j], batched_outputs[j])
            model.train_on_batch(batched_inputs[j], batched_outputs[j])
