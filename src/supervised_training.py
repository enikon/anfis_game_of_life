from training import Training
import numpy as np
import math
import tensorflow as tf
from anfis_model import train_anfis
import matplotlib.pyplot as plt


class SupervisedTraining(Training):

    def __init__(self, normalisation_function):
        super().__init__()
        self.normalisation_function = normalisation_function
        self.models = None
        self.parameters_sets_count = None
        self.parameters_sets_total_count = 0
        self.parameters_count = 0

    def train(self, simulation_model):
        self.models = simulation_model.models
        self.parameters_count = simulation_model.parameters_count
        self.parameters_sets_count = simulation_model.parameters_sets_count
        self.parameters_sets_total_count = simulation_model.parameters_sets_total_count

        din, dout = self._generate_dataset(1024)
        tin, tout = self._generate_dataset(32)

        dpd, tpd = self._fit_predict_test(din, dout, tin, tout)
        self._plot(din, dout, dpd, tin, tout, tpd)

    def _generate_dataset(self, input_data_size):
        # ------------
        # GENERATION
        # ------------

        din = []
        dout = []
        for i in range(input_data_size):
            x = list(np.random.uniform(0.0, 1.0, size=2))
            x.sort(reverse=True)  # TODO BETTER GENERATOR
            din.append(x)
            y = self.normalisation_function(x)
            dout.append([y])

        data_input = tf.convert_to_tensor(din, dtype='float32')
        data_output = tf.convert_to_tensor(dout, dtype='float32')

        return data_input, data_output

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
