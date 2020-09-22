import tensorflow as tf
import numpy as np


class SacHuberLoss:
    def __init__(self, gamma, epsilon):
        self.gamma = gamma # 0.99
        self.epsilon = epsilon # np.finfo(np.float32).eps.item()

    #@tf.function
    def sac_huber_loss(self, truth, predicted):
        # --------
        #   # truth = Nx2 rewards + null
        #   # predicted = Nx2 action + critic response
        # --------
        rewards = tf.gather(truth, 0, axis=0)
        gamma = tf.fill(value=self.gamma, dims=tf.shape(rewards))
        gamma = tf.math.cumprod(gamma)

        three_rewards = tf.expand_dims(tf.expand_dims(rewards, axis=0), axis=0)
        three_gamma = tf.expand_dims(tf.expand_dims(gamma, axis=0), axis=0)
        convolution = tf.nn.conv1d(three_rewards, three_gamma, stride=1, padding='VALID')

        # Normalize
        #truth[:][1] = (truth[:][1] - np.mean(truth[:][1])) / (np.std(truth[:][1]) + self.epsilon)

        # # Calculating loss values to update our network
        # history = zip(action_probs_history, critic_value_history, returns)
        # actor_losses = []
        # critic_losses = []
        # for log_prob, value, ret in history:
        #     # At this point in history, the critic estimated that we would get a
        #     # total reward = `value` in the future. We took an action with log probability
        #     # of `log_prob` and ended up recieving a total reward = `ret`.
        #     # The actor must be updated so that it predicts an action that leads to
        #     # high rewards (compared to critic's estimate) with high probability.
        #     diff = ret - value
        #     actor_losses.append(-log_prob * diff)  # actor loss
        #
        #     # The critic must be updated so that it predicts a better estimate of
        #     # the future rewards.
        #     critic_losses.append(
        #         huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
        #     )
        #
        # # Backpropagation
        # loss_value = sum(actor_losses) + sum(critic_losses)

        d=0
        d+=1
        return rewards

        # @tf.function


class LogLoss:
    def __init__(self, delta):
        self.delta = delta

    def log_loss(self, truth, predicted):
        out = tf.clip_by_value(predicted, 1e-8, 1-1e-8)
        prob = truth * tf.math.log(out)

        return tf.math.reduce_sum(-prob * self.delta)
