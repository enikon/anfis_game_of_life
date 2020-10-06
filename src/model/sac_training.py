from src.model.training import Training
import numpy as np
import tensorflow as tf
from src.constructs.experience_holder import ExperienceHolder


class SACTraining(Training):

    def __init__(self, environment):
        super().__init__()
        self.environment = environment
        self.models = None
        self.parameters_sets_count = None
        self.parameters_sets_total_count = 0
        self.parameters_count = 0

        self.gamma = 0.75
        self.alpha = 0.000001
        self.beta = 0.003
        self.tau = 0.02

        self.experience = ExperienceHolder(capacity=1000000, cells=5)  # state, action, reward, state', done

    def train(self, simulation_model, **kwargs):
        self.models = simulation_model.models
        self.parameters_count = simulation_model.parameters_count
        self.parameters_sets_count = simulation_model.parameters_sets_count
        self.parameters_sets_total_count = simulation_model.parameters_sets_total_count

        self.train_sac(
            self.models,
            epochs=1000, max_steps=100, experience_batch=4000, simulation=self.environment)

    def train_sac(self, models, epochs, max_steps, experience_batch, simulation):

        # deterministic random
        np.random.seed(0)

        history = []
        epoch_steps_ = 50
        simulation.reset()
        update_net(models['critic-v'], models['critic-v-t'], 1.0)
        init = True

        for i in range(epochs):
            print("epoch: ", i)
            episode_reward = 0
            reset = False
            if init:
                epoch_steps = experience_batch+1000
                init = False
            else:
                epoch_steps = epoch_steps_
            j = 0

            while not(j > epoch_steps):# and reset):
                j += 1
                reset = False
                # ---------------------------
                # Observe state s and select action according to current policy
                # ---------------------------

                # Get simulation state
                state_raw = simulation.get_normalised()
                # state_unwound = [[i for t in state for i in t]]

                state = [state_raw[0]]  # TODO
                state_tf = tf.convert_to_tensor(state)

                # Get actions distribution from current model
                # and their approx value from critic
                actions_tf, _, _ = models['actor'](state_tf)
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
                    reset = True

            # ---------------------------
            # Updating network
            # ---------------------------
            if self.experience.size() > experience_batch+1000:  # update_counter_limit:
                exp = self.experience.replay(experience_batch)
                states_tf = tf.convert_to_tensor(exp[0], dtype='float64')
                actions_tf = tf.convert_to_tensor(exp[1], dtype='float64')
                rewards_tf = tf.convert_to_tensor(exp[2], dtype='float64')
                states_l_tf = tf.convert_to_tensor(exp[3], dtype='float64')
                not_dones_tf = tf.convert_to_tensor(exp[4], dtype='float64')

                with tf.GradientTape(watch_accessed_variables=True, persistent=True) as tape:

                    q_1_current = models['critic-q-1']([states_tf, actions_tf])
                    q_2_current = models['critic-q-2']([states_tf, actions_tf])
                    v_l_current = models['critic-v-t'](states_l_tf)

                    q_target = tf.stop_gradient(rewards_tf + not_dones_tf * self.gamma * v_l_current)
                    q_1_loss = tf.reduce_mean((q_target - q_1_current) ** 2)
                    q_2_loss = tf.reduce_mean((q_target - q_2_current) ** 2)

                    v_current = models['critic-v'](states_tf)
                    actions, policy_loss, sigma = models['actor'](states_tf)
                    q_1_policy = models['critic-q-1']([states_tf, actions_tf])
                    q_2_policy = models['critic-q-2']([states_tf, actions_tf])
                    q_min_policy = tf.minimum(q_1_policy, q_2_policy)

                    v_target = tf.stop_gradient(q_min_policy - self.alpha * policy_loss)
                    v_loss = tf.reduce_mean((v_target - v_current)**2)

                    a_loss = tf.reduce_mean(self.alpha * policy_loss - q_min_policy)

                backward(tape, models['critic-q-1'], q_1_loss)
                backward(tape, models['critic-q-2'], q_2_loss)
                backward(tape, models['critic-v'], v_loss)
                update_net(models['critic-v'], models['critic-v-t'], self.tau)

                backward(tape, models['actor'], a_loss)

                del tape
                #
                #
                #     # ----------------------
                #     # VALUE NETWORK UPDATE
                #     # ----------------------
                #     #   ,, I want to guess how good the set state is. I try to be close to Q for
                #     #   action decided in current policy because it would be the action I would take.
                #     #   Reward is decreased based on entropy as taking less likely decision in a state
                #     #   shouldn't be comparable with taking the most likely decision''
                #     tape_a.watch(models['actor'].trainable_variables)
                #
                #     actions_policy_tf, policy_loss = models['actor'](states_tf)
                #
                #
                #     def critic_value_update(tape):
                #         # target(V) =  Q_theta(s_t,a_t) - log pi_theta(a_t | s_t), a_t ~ pi, s_t ~ D
                #         critic_target_v = q_policy_tf - policy_loss
                #         return critic_target_v
                #
                #     vl = custom_loop(critic_value_update, models['critic-v'], states_tf, tape_v)
                #
                #     # ---------------------
                #     # Q NETWORK UPDATE
                #     # ---------------------
                #     #   ,, I want to guess how reward is tied to doing set action in set state.
                #     #   That is why i target the reward response with replay params and then
                #     #   subtract the reward of state I'm in ''
                #
                #     def critic_q_update(tape):
                #         # target(Q) =  r(s_t,a_t) + gamma * V_psi(s_{t+1})
                #         critic_target_q = rewards_tf + not_dones_tf * self.gamma * models['critic-v-t'](states_l_tf)
                #         return critic_target_q
                #
                #     cl = custom_loop(critic_q_update, models['critic-q'], [states_tf, actions_tf], tape_q)
                #
                #     # ----------------------
                #     # POLICY NETWORK UPDATE
                #     # ----------------------
                #
                #     def actor_update(tape):
                #         # loss(A) = - alpha * log(pi_theta(a_t|s_t) + Q(s_t,a_t)
                #         loss = policy_loss - q_policy_tf
                #         loss_grad = tf.reduce_mean(loss)
                #         return loss_grad
                #
                #     al = custom_loop(actor_update, models['actor'], states_tf, tape_a, target=False)
                #
                print('Loss:\n\tvalue: {}\n\tq1   : {}\n\tq2   : {}\n\tactor (ascent): {}'.format(
                     tf.reduce_mean(v_loss),
                     tf.reduce_mean(q_1_loss),
                     tf.reduce_mean(q_2_loss),
                     tf.reduce_mean(a_loss) #Gradient ascent

                ))
                print('Episode Reward: {}'.format(episode_reward))
                print('Batch sigma: {}'.format(tf.reduce_mean(sigma)))
                print('PolicyLoss sigma: {}'.format(tf.reduce_mean(policy_loss)))

                #
                # update_net(models['critic-v'], models['critic-v-t'], self.tau)

#
# def custom_loop(function, model, model_input, tape, target=True):
#     if target:
#         tape.watch(model.trainable_variables)
#         loop_target = function(tape)
#         loop_loss = model.loss_functions[0](loop_target, model(model_input))
#     else:
#         loop_loss = function(tape)
#     grads = tape.gradient(loop_loss, model.trainable_variables)
#     model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
#     return loop_loss
#


def update_net(model, target, tau):
    len_vars = len(model.weights)
    for i in range(len_vars):
        target.weights[i].assign(tau * model.weights[i] + (1.0 - tau) * target.weights[i])

# def choose_action(mu, sig, with_loss=True):
#
#     sig_clipped = tf.clip_by_value(sig, -20.0, 2.0)
#     exp_sig = tf.exp(sig_clipped)
#     distribution = tfp.distributions.Normal(mu, exp_sig)
#
#     action = distribution.sample()
#     squeezed = tf.tanh(action)
#
#     if not with_loss:
#         return squeezed
#
#     loss_result = distribution.log_prob(action)
#     loss_result_with_tanh_correction = loss_result - tf.math.log(1.0 - squeezed ** 2 + 1e-5)
#     return squeezed, loss_result_with_tanh_correction
#
#
# def sample_action_from_policy(model, states_tf):
#     mu_tf, sig_tf = model(states_tf)
#     return choose_action(mu_tf, sig_tf, with_loss=True)
#
#


def backward(tape, model, loss):
    grads = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(
        zip(grads, model.trainable_variables))
