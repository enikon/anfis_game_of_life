import tensorflow as tf
import numpy as np
from scipy.stats import truncnorm
from sklearn.model_selection import train_test_split


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


def train_anfis(models, inputs, outputs, epochs, batch_size, learning_rate):

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

        forward_step(models, inputs, outputs, learning_rate=learning_rate)
        for j in range(num_batches):
            backward_pass(models["anfis"], batched_inputs[j], batched_outputs[j])


def train_sac(models, epochs, max_steps, simulation, learning_rate):

    #deterministic random
    np.random.seed(0)

    gamma = 0.05
    history = []

    update_counter_step = 500
    update_counter_limit = 1000

    running_reward = 0
    for i in range(epochs):
        print("epoch: ", i)
        simulation.reset()
        episode_reward = 0

        for step in range(0, max_steps):

            # ---------------------------
            # Observe state s and select action according to current policy
            # ---------------------------

            # Get simulation state
            state = simulation.get_normalised()
            state_tf = tf.convert_to_tensor(state)

            # Get actions distribution from current model
            # and their approx value from critic
            actions_dist = models['actor'](state_tf).numpy().reshape((-1, 2))

            # Choose from action by sampling probability distribution
            #actions = np.tanh([
            #    mi + np.random.normal(loc=0.0, scale=1.0)*np.exp(sig) for mi, sig in actions_dist
            #])
            actions = [truncnorm.rvs(truncate(mi, sig, 0, 1)) for mi, sig in actions_dist]

            # ---------------------------
            # Execute action in the environment
            # ---------------------------
            reward, done = simulation.step_normalised(actions)
            episode_reward += reward

            # ---------------------------
            # Observe next state
            # ---------------------------

            state_l = simulation.get_normalised()

            # ---------------------------
            # Store information in replay buffer
            # ---------------------------

            history.append((state, actions, reward, state_l, done))

            if done:
                break
        # Update running reward to check condition for solving
        running_reward = gamma * episode_reward + (1 - gamma) * running_reward

        # ---------------------------
        # Updating network
        # ---------------------------
        if len(history) > update_counter_limit:
            history_tmp, experience = train_test_split(history, test_size=update_counter_step)
            history = history_tmp

            for i in experience:
                pass


def truncate(mi, sig, f_min, f_max):
    return (f_min - mi) / sig, (f_max - mi) / sig
