from tensorflow.keras.layers import Input
from anfis_layers import FuzzificationLayer, RulesLayer, SumNormalisationLayer, DefuzzificationLayer
import tensorflow as tf

# reset console, just in case
tf.keras.backend.clear_session()

# input layer, 3 inputs (predator, prey, food)
f0 = Input(shape=(3,))

# fuzzification layer predator falls in 3 categories, prey in 4 categories, food in 3
#   values are random TODO DEBUG ONLY
fuzzy_sets_count = [3, 4, 3]
f1 = FuzzificationLayer(fuzzy_sets_count=fuzzy_sets_count)(f0)

# rules layer as before
f2 = RulesLayer(fuzzy_sets_count=fuzzy_sets_count)(f1)

# normalise values to 1.0 for each node
f3 = SumNormalisationLayer()(f2)

# defuzzification layer
f4 = DefuzzificationLayer()([f0, f3])
a = 1