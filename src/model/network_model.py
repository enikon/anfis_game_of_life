import tensorflow as tf
from tensorflow.keras.layers import *
from src.anfis.anfis_layers import *
from src.anfis.anfis_model import AnfisGD


class NetworkModel:
    def __init__(self, training):

        self.parameters_count = 2
        self.results_count = 1
        self.parameters_sets_count = [3, 4]
        self.parameters_sets_total_count = sum(self.parameters_sets_count)

        self.models = {}
        self._initialise_layers()  # initialises self.models[]

        self.training = training

        self.train()

    def _initialise_layers(self):
        # ------------
        # LAYERS & DEBUG
        # ------------

        f_states = Input(shape=(self.parameters_count,))
        f_actions = Input(shape=(self.results_count,))

        model_anfis = tf.keras.layers.Dense(10)# AnfisGD(self.parameters_sets_count)
        f_anfis = model_anfis(f_states)

        f_mu = tf.keras.layers.Dense(self.results_count, activation='softmax')(f_anfis) # mu SAC Actor
        f_sig = tf.keras.layers.Dense(self.results_count, activation='softmax')(f_anfis) # sigma SAC Actor

        # SAC Critic Value (Estimating rewards of being in state s)
        f_critic_v1 = tf.keras.layers.Dense(10)(f_states)
        f_critic_v2 = tf.keras.layers.Dense(10)(f_critic_v1)
        f_critic_v = tf.keras.layers.Dense(1, activation='linear')(f_critic_v2)

        # SAC Critic Q (Estimating rewards of takig action a while in state s)
        f_critic_q_concatenate = tf.keras.layers.Concatenate()([f_states, f_actions])
        f_critic_q1 = tf.keras.layers.Dense(10)(f_critic_q_concatenate)
        f_critic_q2 = tf.keras.layers.Dense(10)(f_critic_q1)
        f_critic_q = tf.keras.layers.Dense(1, activation='linear')(f_critic_q2)

        self.models["anfis"] = tf.keras.Model(inputs=f_states, outputs=f_anfis)
        #self.models["forward"] = tf.keras.Model(inputs=f_states, outputs=model_anfis.anfis_forward(f_states))

        self.models["actor"] = tf.keras.Model(inputs=f_states, outputs=[f_mu, f_sig])
        self.models["critic-q"] = tf.keras.Model(inputs=[f_states, f_actions], outputs=f_critic_q)
        self.models["critic-v"] = tf.keras.Model(inputs=f_states, outputs=f_critic_v)

        self.models["anfis"].compile(
            loss=tf.losses.mean_absolute_error,
            optimizer=tf.keras.optimizers.SGD(
                clipnorm=0.5,
                learning_rate=1e-3),
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )
        # self.models["forward"].compile(
        #     loss=tf.losses.mean_absolute_error,
        #     optimizer=tf.keras.optimizers.SGD(
        #         clipnorm=0.5,
        #         learning_rate=1e-3),
        #     metrics=[tf.keras.metrics.RootMeanSquaredError()]
        # )
        self.models["actor"].compile(
            loss=tf.losses.mean_squared_error,
            optimizer=tf.keras.optimizers.SGD(
                clipnorm=0.5,
                learning_rate=1e-4),
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )
        self.models["critic-v"].compile(
            loss=tf.losses.mean_squared_error,
            optimizer=tf.keras.optimizers.SGD(
                clipnorm=0.5,
                learning_rate=1e-3),
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )
        self.models["critic-q"].compile(
            loss=tf.losses.mean_squared_error,
            optimizer=tf.keras.optimizers.SGD(
                clipnorm=0.5,
                learning_rate=1e-3),
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )

    def act(self, din):
        data_input = tf.convert_to_tensor([din], dtype='float32')
        data_output = self.models["actor"](data_input)[0]
        return data_output.numpy()[0]

    def train(self):
        self.training.train(self, hybrid=False)


def mean(y_true, y_pred): #ignore y_pred
    return tf.reduce_mean(y_true)
