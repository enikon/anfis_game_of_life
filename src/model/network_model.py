import tensorflow as tf
from tensorflow.keras.layers import *
from src.anfis.anfis_layers import *
from src.model.sac_layer import *
from src.anfis.anfis_model import AnfisGD

hidden_activation = 'elu'
output_activation = 'linear'


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

        # = tf.keras.layers.Dense(10)# AnfisGD(self.parameters_sets_count)
        #f_anfis = model_anfis(densanf)#model_anfis(f_states)
        f_policy_1 = tf.keras.layers.Dense(5, activation=hidden_activation)(f_states)
        f_policy_2 = tf.keras.layers.Dense(5, activation=hidden_activation)(f_policy_1)
        f_policy_mu = tf.keras.layers.Dense(self.results_count, activation=output_activation)(f_policy_2)
        f_policy_sig = tf.keras.layers.Dense(self.results_count, activation=output_activation)(f_policy_2)
        f_policy = GaussianLayer()([f_policy_mu, f_policy_sig])

        #self.models["anfis"] = tf.keras.Model(inputs=f_states, outputs=f_anfis)
        #self.models["forward"] = tf.keras.Model(inputs=f_states, outputs=model_anfis.anfis_forward(f_states))

        self.models["actor"] = tf.keras.Model(inputs=f_states, outputs=f_policy)

        self.models["critic-q-1"] = generate_q_network([f_states, f_actions])
        self.models["critic-q-2"] = generate_q_network([f_states, f_actions])

        self.models["critic-v"] = generate_value_network(f_states)
        self.models["critic-v-t"] = generate_value_network(f_states)

        # self.models["anfis"].compile(
        #     loss=tf.losses.mean_absolute_error,
        #     optimizer=tf.keras.optimizers.SGD(
        #         clipnorm=0.5,
        #         learning_rate=1e-3),
        #     metrics=[tf.keras.metrics.RootMeanSquaredError()]
        # )
        # self.models["forward"].compile(
        #     loss=tf.losses.mean_absolute_error,
        #     optimizer=tf.keras.optimizers.SGD(
        #         clipnorm=0.5,
        #         learning_rate=1e-3),
        #     metrics=[tf.keras.metrics.RootMeanSquaredError()]
        # )
        self.models["actor"].compile(
            loss=tf.losses.mean_squared_error,
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=1e-3),
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )

    def act(self, din):
        data_input = tf.convert_to_tensor([din], dtype='float64')
        data_output = self.models["actor"](data_input)[0]
        return data_output.numpy()[0]

    def train(self):
        self.training.train(self, hybrid=False)


def mean(y_true, y_pred): #ignore y_pred
    return tf.reduce_mean(y_true)


def generate_value_network(inputs):
    # SAC Critic Value (Estimating rewards of being in state s)
    f_critic_v1 = tf.keras.layers.Dense(5, activation=hidden_activation)(inputs)
    f_critic_v2 = tf.keras.layers.Dense(5, activation=hidden_activation)(f_critic_v1)
    f_critic_v = tf.keras.layers.Dense(1, activation=output_activation)(f_critic_v2)
    m_value = tf.keras.Model(inputs=inputs, outputs=f_critic_v)
    m_value.compile(
        loss=tf.losses.mean_squared_error,
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3),
        metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )
    return m_value


def generate_q_network(inputs):
    # SAC Critic Q (Estimating rewards of taking action a while in state s)
    f_critic_q_concatenate = tf.keras.layers.Concatenate()(inputs)
    f_critic_q1 = tf.keras.layers.Dense(5, activation=hidden_activation)(f_critic_q_concatenate)
    f_critic_q2 = tf.keras.layers.Dense(5, activation=hidden_activation)(f_critic_q1)
    f_critic_q = tf.keras.layers.Dense(1, activation=output_activation)(f_critic_q2)

    m_q = tf.keras.Model(inputs=inputs, outputs=f_critic_q)
    m_q.compile(
        loss=tf.losses.mean_squared_error,
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3),
        metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )
    return m_q
