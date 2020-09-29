from src.model.training import Training
import numpy as np
import math
import tensorflow as tf
import src.anfis.anfis_model as anfis_model
import matplotlib.pyplot as plt


class SupervisedTraining(Training):

    def __init__(self, normalisation_function):
        super().__init__()
        self.normalisation_function = normalisation_function
        self.models = None
        self.parameters_sets_count = None
        self.parameters_sets_total_count = 0
        self.parameters_count = 0

        self.epochs = 500
        self.learning_rate = 1e-5

    def train(self, simulation_model, **kwargs):
        self.models = simulation_model.models
        self.parameters_count = simulation_model.parameters_count
        self.parameters_sets_count = simulation_model.parameters_sets_count
        self.parameters_sets_total_count = simulation_model.parameters_sets_total_count

        din, dout = self._generate_dataset(1024)
        tin, tout = self._generate_dataset(32)

        hybrid = kwargs.get('hybrid', True)

        if hybrid:
            dpd, tpd = self._fit_predict_test(din, dout, tin, tout) # TODO FIX THIS MESS
        else:
            print('GD Everywhere')
            self.models['anfis'].fit(din, dout, epochs=self.epochs, shuffle=True,
                                     callbacks=[tf.keras.callbacks.EarlyStopping(
                                         monitor='loss',
                                         patience=5,
                                         min_delta=1e-3
                                     )])
            dpd = self.models['anfis'].predict(din)
            tpd = self.models['anfis'].predict(tin)

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
        self.train_anfis(
            self.models, data_input, data_output,
            epochs=self.epochs,
            learning_rate=self.learning_rate)
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
            ax[t // mf_cols][t % mf_cols].plot(
                mf_x,
                1.0 / (1.0 + (((mf_x - mf_c[i]) / mf_a[i]) ** 2) ** (1.0 / mf_b[i])))
        plt.show()

    def train_anfis(self, models, inputs, outputs, epochs, batch_size, learning_rate):
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

            anfis_model.forward_step(models, inputs, outputs, learning_rate=learning_rate)
            for j in range(num_batches):
                anfis_model.backward_pass(models["anfis"], batched_inputs[j], batched_outputs[j])