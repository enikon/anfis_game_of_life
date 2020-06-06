from tensorflow.keras.layers import Input
from anfis_model import *
from anfis_layers import *
import tensorflow as tf

# reset console, just in case
tf.keras.backend.clear_session()

#   values are random TODO DEBUG ONLY
# input layer, 3 inputs (predator, prey, food)
# fuzzification layer predator falls in 3 categories, prey in 4 categories, food in 3
parameters_count = 3
parameters_sets_count = [2, 4, 3]

f0 = Input(shape=(parameters_count,))
f1 = FuzzificationLayer(fuzzy_sets_count=parameters_sets_count)(f0)
f2 = RulesLayer(fuzzy_sets_count=parameters_sets_count)(f1)
f3 = SumNormalisationLayer()(f2)
f4 = DefuzzificationLayer()([f0, f3])

anfis = tf.keras.Model(inputs=f0, outputs=f4)
forward = tf.keras.Model(inputs=f0, outputs=f3)

data_input = tf.convert_to_tensor(
    [[0.1, 0.1, 0.2], [0.5, 0.5, 0.5],
     [0.6, 0.1, 0.8], [0.2, 0.5, 0.8],
     [0.3, 0.5, 0.3], [0.0, 0.7, 0.3]])

data_output = tf.convert_to_tensor(
    [[0.2], [0.5],
     [0.5], [0.6],
     [0.1], [0.0]])

b = train_step(anfis, forward, data_input, data_output)

a = 1