from tensorflow.keras import constraints
from tensorflow.keras import layers
from tensorflow.keras import initializers
import numpy as np
import tensorflow as tf


class FuzzificationLayer(layers.Layer):
    # TODO : ADD DIFFERENT MEMBER FUNCTIONS
    def __init__(self, fuzzy_sets_count, **kwargs):
        """Fuzzification layer for ANFIS.

          Args:
            fuzzy_sets_count: a list of the number of fuzzy sets for each input parameter

            Example:
                 for the following set labels [['warm','normal','cold'],['dark','bright'],['very slow','slow','fast','very fast']]
                 fuzzy_sets_count = [3,2,4]
                 (3 for temperature, 2 for lightness and 4 for speed)

        """
        super(FuzzificationLayer, self).__init__(**kwargs)
        self.fuzzy_sets_count = fuzzy_sets_count

        if self.fuzzy_sets_count is None:
            raise Exception('FuzzificationLayer', '\'units_dimensions\' can\'t be None.')
        if len(self.fuzzy_sets_count) < 1:
            raise Exception('FuzzificationLayer', '\'units_dimensions\' must have at least one element.')

        self.units = None
        self.a = None
        self.b = None
        self.c = None

    def build(self, input_shape):
        # tensorflow reported bug troubleshooting  https://stackoverflow.com/questions/56094714/how-can-i-call-a-custom-layer-in-keras-with-the-functional-api
        input_shape = input_shape.as_list()

        if input_shape[1] < 1:
            raise Exception('FuzzificationLayer', '\'input_size\' must be at least 1.')
        len_ud = len(self.fuzzy_sets_count)
        if input_shape[1] > len_ud:
            self.units = np.sum(self.fuzzy_sets_count) + self.fuzzy_sets_count[len_ud - 1] * (input_shape[1] - len_ud)
        elif input_shape[1] == len_ud:
            self.units = np.sum(self.fuzzy_sets_count)
        else:
            raise Exception('FuzzificationLayer',
                            '\'units_dimensions\' have more elements than there are input parameters.')

        # generating initial weights
        #   spaced evenly between 0 and 1.
        #   initial weights are constant, no seed required
        location = []
        width = []
        for u in self.fuzzy_sets_count:
            for i in range(u):
                location.append((1.0 + i) / (1.0 + u))
                width.append(0.5 / u)

        # TODO MIGRATE : DIFFERENT MEMBER FUNCTION

        # member function shape is bell member function with range
        #   ( equivalents to gaussian member function in parenthesis, (not yet implemented))

        # a (sigma width) - width, 0-1
        self.a = self.add_weight(shape=(self.units,), trainable=True,
                                 initializer=initializers.constant(width),
                                 constraint=constraints.MinMaxNorm(min_value=0.0, max_value=1.0))

        # b (sigma 1/height) - slope, 0-1
        self.b = self.add_weight(shape=(self.units,), trainable=True,
                                 initializer=initializers.constant(0.5),
                                 constraint=constraints.MinMaxNorm(min_value=0.0, max_value=1.0))
        # c (mu) - mean, 0-1
        self.c = self.add_weight(shape=(self.units,), trainable=True,
                                 initializer=initializers.constant(location),
                                 constraint=constraints.MinMaxNorm(min_value=0.0, max_value=1.0))

    def call(self, inputs, **kwargs):
        # comp = 1/(1+abs((x-c)/a)^(2/b)))
        # used Bell equation
        #   2/b instead of 2b for normalisation purpose
        # tf.repeat allows to expand input to fuzzy set count
        return 1 / (1 + tf.pow(tf.abs((tf.repeat(inputs, self.fuzzy_sets_count, axis=1) - self.c) / self.a), 2 / self.b))

    def get_config(self):
        # for serialisation only, in case we want to save model directly
        base_config = super(FuzzificationLayer, self).get_config()
        base_config['fuzzy_sets_count'] = self.fuzzy_sets_count
        base_config['output_dim'] = self.units
        return base_config


class RulesLayer(layers.Layer):
    # TODO : ADD APRIORI RULES
    def __init__(self, fuzzy_sets_count, **kwargs):
        """Rules layer for ANFIS.

          Args:
            fuzzy_sets_count: a list of the number of fuzzy sets for each input parameter

            Example:
                 for the following set labels [['warm','normal','cold'],['dark','bright'],['very slow','slow','fast','very fast']]
                 fuzzy_sets_count = [3,2,4]
                 (3 for temperature, 2 for lightness and 4 for speed)

        """
        super(RulesLayer, self).__init__(**kwargs)
        self.fuzzy_sets_count = fuzzy_sets_count

        if self.fuzzy_sets_count is None:
            raise Exception('FuzzificationLayer', '\'units_dimensions\' can\'t be None.')
        if len(self.fuzzy_sets_count) < 1:
            raise Exception('FuzzificationLayer', '\'units_dimensions\' must have at least one element.')

        self.units = None

    def build(self, input_shape):
        # tensorflow reported bug troubleshooting  https://stackoverflow.com/questions/56094714/how-can-i-call-a-custom-layer-in-keras-with-the-functional-api
        input_shape = input_shape.as_list()

        if input_shape[1] < 1:
            raise Exception('FuzzificationLayer', '\'input_size\' must be at least 1.')
        len_ud = len(self.fuzzy_sets_count)
        if input_shape[1] > len_ud:
            self.units = np.prod(self.fuzzy_sets_count) * self.fuzzy_sets_count[len_ud - 1] ** (input_shape[1] - len_ud)
        elif input_shape[1] == len_ud:
            self.units = np.prod(self.fuzzy_sets_count)
        else:
            raise Exception('FuzzificationLayer',
                            '\'units_dimensions\' have more elements than there are input parameters.')

    def call(self, inputs, **kwargs):
        split = tf.split(inputs, self.fuzzy_sets_count, axis=1)
        res = tf.expand_dims(split[0], 2)
        for i in range(1, len(split)):
            res = tf.matmul(res, tf.expand_dims(split[i], 2), transpose_b=True)
            print(res.shape)
            res = tf.reshape(res, shape=[-1, res.shape[1]*res.shape[2], 1])
        return tf.reshape(res, shape=[-1, res.shape[1]])

    def get_config(self):
        # for serialisation only, in case we want to save model directly
        base_config = super(RulesLayer, self).get_config()
        base_config['fuzzy_sets_count'] = self.fuzzy_sets_count
        base_config['output_dim'] = self.units
        return base_config


class SumNormalisationLayer(layers.Layer):
    def __init__(self, target_sum=1.0, **kwargs):
        """Normalisation layer for ANFIS.

          Args:
            target_sum: all weights will sum to this value

        """
        super(SumNormalisationLayer, self).__init__(**kwargs)
        self.target_sum = target_sum
        self.units = None

    def build(self, input_shape):
        # tensorflow reported bug troubleshooting  https://stackoverflow.com/questions/56094714/how-can-i-call-a-custom-layer-in-keras-with-the-functional-api
        input_shape = input_shape.as_list()
        if input_shape[1] < 1:
            raise Exception('FuzzificationLayer', '\'input_size\' must be at least 1.')
        self.units = input_shape[1]

    def call(self, inputs, **kwargs):
        summed = tf.math.reduce_sum(inputs, 1)
        return inputs/summed

    def get_config(self):
        # for serialisation only, in case we want to save model directly
        base_config = super(SumNormalisationLayer, self).get_config()
        base_config['target_sum'] = self.target_sum
        base_config['output_dim'] = self.units
        return base_config
