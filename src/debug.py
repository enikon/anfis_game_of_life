from tensorflow.keras.layers import Input
from anfis_layers.fuzzification_layer import FuzzificationLayer
import tensorflow as tf

# reset console, just in case
tf.keras.backend.clear_session()

# input layer, 3 inputs (predator, prey, food)
f0 = Input(shape=(3,))

# fuzzification layer predator falls in 3 categories, prey in 4 categories, food in 3
#   values are random TODO DEBUG ONLY
fl = FuzzificationLayer(fuzzy_sets_count=[3, 4, 3])(f0)

a = 1