from tensorflow.keras.layers import Input
from anfis_model import *
from anfis_layers import *
import tensorflow as tf
import math

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

Axes3D = Axes3D  # pycharm auto import

# reset console, just in case
tf.keras.backend.clear_session()

np.random.seed(1)
np.set_printoptions(precision=3, suppress=True)


class SimulationModel:
    def __init__(self, normalisation_function):

        self.parameters_count = 2
        self.parameters_sets_count = [3, 4]
        self.parameters_sets_total_count = sum(self.parameters_sets_count)

        self.debug = False
        self.models = {}
        self._initialise_layers()  # initialises self.models[]

        self.normalisation_function = normalisation_function

        din, dout = self._generate_dataset(1024)
        tin, tout = self._generate_dataset(32)

        dpd, tpd = self._fit_predict_test(din, dout, tin, tout)
        self._plot(din, dout, dpd, tin, tout, tpd)

    def _initialise_layers(self):
        # ------------
        # LAYERS & DEBUG
        # ------------

        f0 = Input(shape=(self.parameters_count,))
        f1 = FuzzificationLayer(fuzzy_sets_count=self.parameters_sets_count)(f0)
        f2 = RulesLayer(fuzzy_sets_count=self.parameters_sets_count)(f1)
        f3 = SumNormalisationLayer()(f2)
        f4 = DefuzzificationLayer()([f0, f3])

        self.models["anfis"] = tf.keras.Model(inputs=f0, outputs=f4)
        self.models["forward"] = tf.keras.Model(inputs=f0, outputs=f3)

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

    def _generate_dataset(self, input_data_size):
        # ------------
        # GENERATION
        # ------------

        din = []
        dout = []
        for i in range(input_data_size):
            x = list(np.random.uniform(0.0, 1.0, size=2))
            x.sort(reverse=True) #TODO BETTER GENERATOR
            din.append(x)
            y = self.normalisation_function(x)
            dout.append([y])

        data_input = tf.convert_to_tensor(din, dtype='float32')
        data_output = tf.convert_to_tensor(dout, dtype='float32')

        return data_input, data_output

    def act(self, din):
        data_input = tf.convert_to_tensor([din], dtype='float32')
        data_output = self.models["anfis"].predict(data_input)
        return data_output[0]

    def _fit_predict_test(self, data_input, data_output, test_input, test_output):
        # ------------
        # TRAINING
        # ------------
        train_anfis(self.models, data_input, data_output,
                    epochs=10, batch_size=32,
                    learning_rate=1 - 1e-3)
        # ------------
        # PREDICTING
        # ------------
        dpd = self.models["anfis"].predict(data_input)

        # ------------
        # TESTING
        # ------------
        tpd = self.models["anfis"].predict(test_input)

        return dpd, tpd

    def _plot(self, din, dout, dpd, tin, tout, tpd):
        # ------------
        # PLOTTING - TRAINING/TESTING
        # ------------

        x_d = np.array(din)[:, 0]
        x_t = np.array(tin)[:, 0]

        y_d = np.array(din)[:, 1]
        y_t = np.array(tin)[:, 1]

        z_do, z_dp = np.array(dout), np.array(dpd)
        z_to, z_tp = np.array(tout), np.array(tpd)

        fig = plt.figure(figsize=(12, 6))
        ax = [
            fig.add_subplot(1, 2, 1, projection='3d'),
            fig.add_subplot(1, 2, 2, projection='3d')
        ]

        ax[0].scatter(x_d, y_d, z_do)
        ax[0].scatter(x_d, y_d, z_dp)

        ax[1].scatter(x_t, y_t, z_to)
        ax[1].scatter(x_t, y_t, z_tp)

        for i in range(2):
            ax[i].set_xlabel('x')
            ax[i].set_ylabel('y')
            ax[i].set_zlabel('z')

        # ------------
        # PLOTTING - MEMBERSHIP PARAMETERS FUNCTION
        # ------------

        mf_a = self.models["anfis"].trainable_variables[0].numpy()
        mf_b = self.models["anfis"].trainable_variables[1].numpy()
        mf_c = self.models["anfis"].trainable_variables[2].numpy()
        mf_x = np.linspace(0.0, 1.0, 1000)

        mf_rows = math.floor(math.sqrt(self.parameters_count))
        mf_cols = math.ceil(self.parameters_count / mf_rows)

        fig, ax = plt.subplots(nrows=mf_rows, ncols=mf_cols, figsize=(12, 6))
        ax = np.reshape(ax, (mf_rows, mf_cols))

        t = 0
        j = self.parameters_sets_count[t]
        for i in range(self.parameters_sets_total_count):
            if i >= j:
                t += 1
                j += self.parameters_sets_count[t]

            # bell function
            ax[t // mf_cols][t % mf_cols].plot(mf_x,
                                               1.0 / (1.0 + (((mf_x - mf_c[i]) / mf_a[i]) ** 2) ** (1.0 / mf_b[i])))

        plt.show()
