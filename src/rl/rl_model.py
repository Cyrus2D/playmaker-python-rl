import keras
from keras import layers
import tensorflow as tf
import numpy as np

num_states = 7
print("Size of State Space ->  {}".format(num_states))
num_actions = 4
print("Size of Action Space ->  {}".format(num_actions))

upper_bound = 1
lower_bound = 0

print("Max Value of Action ->  {}".format(upper_bound))
print("Min Value of Action ->  {}".format(lower_bound))

def get_actor():
    # Initialize weights between -3e-3 and 3-e3
    last_init = keras.initializers.RandomUniform(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(256, activation="relu")(inputs)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out)

    # Our upper bound is 2.0 for Pendulum.
    outputs = outputs * upper_bound
    model = keras.Model(inputs, outputs)
    return model


def get_critic():
    # State as input
    state_input = layers.Input(shape=(num_states,))
    state_out = layers.Dense(16, activation="relu")(state_input)
    state_out = layers.Dense(32, activation="relu")(state_out)

    # Action as input
    action_input = layers.Input(shape=(num_actions,))
    action_out = layers.Dense(32, activation="relu")(action_input)

    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(256, activation="relu")(concat)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1)(out)

    # Outputs single value for give state-action
    model = keras.Model([state_input, action_input], outputs)

    return model

class RL_Model:
    def __init__(self) -> None:
        self.memory = [] # Store (state, action, reward, next_state) tuples
        self.batch_size = 32
        self.memory_size = 10000

        self.actor_model = get_actor()
        self.critic_model = get_critic()
        self.target_actor = get_actor()
        self.target_critic = get_critic()
        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())

        self.critic_lr = 0.002
        self.actor_lr = 0.001

        self.critic_optimizer = keras.optimizers.Adam(critic_lr)
        self.actor_optimizer = keras.optimizers.Adam(actor_lr)

        self.gamma = 0.99
        self.tau = 0.005
    
    @tf.function
    def update(
        self,
        state_batch,
        action_batch,
        reward_batch,
        next_state_batch,
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch, training=True)
            y = reward_batch + self.gamma * self.target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = self.critic_model([state_batch, action_batch], training=True)
            critic_loss = keras.ops.mean(keras.ops.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = self.actor_model(state_batch, training=True)
            critic_value = self.critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -keras.ops.mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor_model.trainable_variables)
        )
        
    def policy(self, state):
        sampled_actions = tf.squeeze(self.actor_model(state))
        # noise = np.random.normal(0, 0.1, size=sampled_actions.shape)
        sampled_actions = sampled_actions.numpy() # + noise
        legal_action = np.clip(sampled_actions, lower_bound, upper_bound)
        return [np.squeeze(legal_action)]
    
    def update_memory(self, state, action, reward, next_state):
        pass