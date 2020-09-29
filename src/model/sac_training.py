from src.model.training import Training
import math
import numpy as np
import tensorflow as tf
import itertools
from src.constructs.experience_holder import ExperienceHolder


class SACTraining(Training):

    def __init__(self, environment):
        super().__init__()
        self.environment = environment
        self.models = None
        self.parameters_sets_count = None
        self.parameters_sets_total_count = 0
        self.parameters_count = 0

        self.gamma = 0.99
        self.alpha = 1.0

        self.experience = ExperienceHolder(capacity=10000, cells=5)  # state, action, reward, state', done

    def train(self, simulation_model, **kwargs):
        self.models = simulation_model.models
        self.parameters_count = simulation_model.parameters_count
        self.parameters_sets_count = simulation_model.parameters_sets_count
        self.parameters_sets_total_count = simulation_model.parameters_sets_total_count

        self.train_sac(
            self.models,
            epochs=100, max_steps=100, simulation=self.environment,
            learning_rate=1 - 1e-4)

    def train_sac(self, models, epochs, max_steps, simulation, learning_rate):

        # deterministic random
        np.random.seed(0)

        run_gamma = 0.05
        history = []

        update_counter_step = 1000
        update_counter_limit = 2000

        epoch_steps = 10000

        running_reward = 0
        for i in range(epochs):
            print("epoch: ", i)
            simulation.reset()
            episode_reward = 0

            for _ in range(0, epoch_steps):

                # ---------------------------
                # Observe state s and select action according to current policy
                # ---------------------------

                # Get simulation state
                state_raw = simulation.get_normalised()
                #state_unwound = [[i for t in state for i in t]]

                state = [state_raw[0]] #TODO
                state_tf = tf.convert_to_tensor(state)

                # Get actions distribution from current model
                # and their approx value from critic
                actions_mu_dist, actions_sig_dist = models['actor'](state_tf)
                actions_tf, _ = choose_action(actions_mu_dist, actions_sig_dist)
                actions = list(actions_tf.numpy()[0])

                # ---------------------------
                # Execute action in the environment
                # ---------------------------
                reward, done = simulation.step_nominalised(actions)
                episode_reward += reward

                # ---------------------------
                # Observe next state
                # ---------------------------

                state_l_raw = simulation.get_normalised()
                state_l = [state_l_raw[0]]  # TODO

                # ---------------------------
                # Store information in replay buffer
                # ---------------------------

                self.experience.save((state, actions, reward, state_l, 1 if not done else 0))

                if done or simulation.step_counter > max_steps:
                    simulation.reset()

            # Update running reward to check condition for solving
            running_reward = run_gamma * episode_reward + (1 - run_gamma) * running_reward

            # ---------------------------
            # Updating network
            # ---------------------------
            if self.experience.size() > 500: #update_counter_limit:
                exp = self.experience.replay(update_counter_step)
                states_tf = tf.convert_to_tensor(exp[0], dtype=tf.float32)
                actions_tf = tf.convert_to_tensor(exp[1], dtype=tf.float32)
                rewards_tf = tf.convert_to_tensor(exp[2], dtype=tf.float32)
                states_l_tf = tf.convert_to_tensor(exp[3], dtype=tf.float32)
                not_dones_tf = tf.convert_to_tensor(exp[4], dtype=tf.float32)

                #----------------------
                # VALUE NETWORK UPDATE
                #----------------------
                #   ,, I want to guess how good the set state is. I try to be close to Q for
                #   action decided in current policy because it would be the action I would take.
                #   Reward is decreased based on entropy as taking less likely decision in a state
                #   shouldn't be comparable with taking the most likely decision''

                def critic_value_update(tape):
                    # target(V) =  Q_theta(s_t,a_t) - log pi_theta(a_t | s_t), a_t ~ pi, s_t ~ D

                    actions_policy_tf, policy_loss = sample_action_from_policy(models['actor'], states_tf)
                    q_policy_tf = models['critic-q']([states_tf, actions_policy_tf])

                    critic_target_v = q_policy_tf - policy_loss
                    return critic_target_v

                vl = custom_loop(critic_value_update, models['critic-v'], states_tf)

                #---------------------
                # Q NETWORK UPDATE
                #---------------------
                #   ,, I want to guess how reward is tied to doing set action in set state.
                #   That is why i target the reward response with replay params and then
                #   subtract the reward of state I'm in ''

                def critic_q_update(tape):
                    # target(Q) =  r(s_t,a_t) + gamma * V_psi(s_{t+1})
                    critic_target_q = rewards_tf + not_dones_tf * self.gamma * models['critic-v'](states_l_tf)
                    return critic_target_q

                cl = custom_loop(critic_q_update, models['critic-q'], [states_tf, actions_tf])

                # TODO add delay and soft value update
                #----------------------
                # POLICY NETWORK UPDATE
                #----------------------

                def actor_update(tape):
                    actions_policy_tf, policy_loss = sample_action_from_policy(models['actor'], states_tf)
                    q_policy_tf = models['critic-q']([states_tf, actions_policy_tf])

                    # loss(A) = - alpha * log(pi_theta(a_t|s_t) + Q(s_t,a_t)
                    loss = policy_loss - q_policy_tf
                    loss_grad = tf.reduce_mean(loss)
                    return loss_grad

                al = custom_loop(actor_update, models['actor'], states_tf, target=False)

                print('Loss:\n\tvalue: {}\n\tq    : {}\n\tactor: {}'.format(
                    tf.reduce_mean(vl),
                    tf.reduce_mean(cl),
                    tf.reduce_mean(al)
                ))
                # TODO add value soft copy ???


def custom_loop(function, model, model_input, target=True):
    with tf.GradientTape() as tape:
        if target:
            loop_target = function(tape)
            loop_loss = model.loss_functions[0](loop_target, model(model_input))
        else:
            _, _  = model(model_input)
            loop_loss = function(tape)
    grads = tape.gradient(loop_loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loop_loss


def choose_action(mu, sig):
    norms = tf.random.normal(shape=(tf.shape(mu)))
    action = norms * sig + mu
    resampled = (tf.tanh(action)+1.0)/2.0
    return resampled, action


def sample_action_from_policy(model, states_tf):
    mu_tf, sig_tf = model(states_tf)

    actions_policy_tf, actions_choice_tf = choose_action(mu_tf, sig_tf)
    policy_loss = log_policy(actions_choice_tf, actions_policy_tf, mu_tf, sig_tf)
    return actions_policy_tf, policy_loss


def log_policy(x, res, mu, sig):

    sig = tf.keras.backend.clip(sig, 1e-8, 1-1e-8)
    clip = tf.keras.backend.clip(tf.sqrt(2 * math.pi * sig ** 2), 1e-8, 1 - 1e-8)
    loss_result = - (x - mu) ** 2 / (2 * sig ** 2) \
                  - tf.math.log(clip) \
                  - tf.math.log(1.0 - res**2 + 1e-12)
    return loss_result
