from tensorflow.keras.layers import Input
from anfis_layers import *
import tensorflow as tf
from custom_losses import SacHuberLoss

from mpl_toolkits.mplot3d import Axes3D

Axes3D = Axes3D  # pycharm auto import

# reset console, just in case
tf.keras.backend.clear_session()

np.random.seed(1)
np.set_printoptions(precision=3, suppress=True)


class SimulationModel:
    def __init__(self, training):

        self.parameters_count = 2
        self.parameters_sets_count = [3, 4]
        self.parameters_sets_total_count = sum(self.parameters_sets_count)

        self.debug = False
        self.models = {}
        self._initialise_layers()  # initialises self.models[]

        self.training = training

        self.train()

    def _initialise_layers(self):
        # ------------
        # LAYERS & DEBUG
        # ------------

        f0 = Input(shape=(self.parameters_count,))
        f1 = FuzzificationLayer(fuzzy_sets_count=self.parameters_sets_count)(f0)
        f2 = RulesLayer(fuzzy_sets_count=self.parameters_sets_count)(f1)
        f3 = SumNormalisationLayer()(f2)
        f4 = DefuzzificationLayer()([f0, f3])
        f5 = tf.keras.layers.Dense(1)(f4) # SAC Critic

        self.models["anfis"] = tf.keras.Model(inputs=f0, outputs=f4)
        self.models["forward"] = tf.keras.Model(inputs=f0, outputs=f3)

        self.models["sac"] = tf.keras.Model(inputs=f0, outputs=[f4, f5])

        if self.debug:
            self.models["rules"] = tf.keras.Model(inputs=f0, outputs=f2)
            self.models["fuzzyfication"] = tf.keras.Model(inputs=f0, outputs=f1)
            self.models["input"] = tf.keras.Model(inputs=f0, outputs=f0)

        self.models["anfis"].compile(
            loss=tf.metrics.MAE,
            optimizer=tf.keras.optimizers.SGD(
                clipnorm=0.5,
                learning_rate=1e-4),
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )
        self.models["forward"].compile(
            loss=tf.metrics.MAE,
            optimizer=tf.keras.optimizers.SGD(
                clipnorm=0.5,
                learning_rate=1e-4),
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )
        self.models["sac"].compile(
            loss=SacHuberLoss(0.9, 0.1).sac_huber_loss,
            optimizer=tf.keras.optimizers.Adam(
                clipnorm=0.5,
                learning_rate=1e-4),
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )

    def act(self, din):
        data_input = tf.convert_to_tensor([din], dtype='float32')
        data_output = self.models["anfis"].predict(data_input)
        return data_output[0]

    def train(self):
        self.training.train(self)
