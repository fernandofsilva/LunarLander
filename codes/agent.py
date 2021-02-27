import numpy as np
import random
from collections import namedtuple, deque
import tensorflow as tf

from model import QNetwork


class Agent:
    """Interacts with and learns from the environment."""
    def __init__(self,
                 state_size,
                 action_size,
                 buffer_size,
                 batch_size,
                 gamma,
                 tau,
                 lr,
                 update_every):
        """Initialize an Agent object.

        Args:
            state_size: Integer. Dimension of each state
            action_size: Integer. Dimension of each action
            buffer_size: Integer. Replay buffer size
            batch_size: Integer. Mini-batch size
            gamma: Float. Discount factor
            tau: Float. For soft update of target parameters
            lr: Float. Learning rate
            update_every: Integer. How often to update the network
        """
        # Environment parameters
        self.state_size = state_size
        self.action_size = action_size

        # Q-Learning
        self.gamma = gamma

        # Q-Network
        self.model_local = QNetwork(state_size, action_size)
        self.model_target = QNetwork(state_size, action_size)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.loss = tf.keras.losses.MeanSquaredError(name="mse")
        self.tau = tau
        self.update_every = update_every
        self.batch_size = batch_size

        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size)

        # Initialize time step (for updating every update_every steps)
        self.t_step = 0

    def __str__(self):
        pass

    def __repr__(self):
        pass

    def step(self, state, action, reward, next_state, done):
        """Save state on buffer and trigger learn according to update_every

        Args:
            state: The previous state of the environment
            action: Integer. Previous action selected by the agent
            reward: Float. Reward value
            next_state: The current state of the environment
            done: Boolean. Whether the episode is complete
        """
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every update_every time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:

            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Args:
            state: A array like object or list with states
            eps: Float. Random value for epsilon-greedy action selection

        Returns:
            An action selected by the network or by the epsilon-greedy method
        """
        # Reshape state
        state = np.expand_dims(state, 0)

        # Predict action
        action_values = self.model_local(state)

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values)
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.

        Args:
            experiences: Tuple. Content of tuple (s, a, r, s', done)
        """
        states, actions, rewards, next_states, dones = experiences

        with tf.GradientTape(persistent=True) as tape:

            # Get expected Q values from local model
            q_expected = self.model_local(states)

            # Get max predicted Q values (for next states) from target model
            q_targets_next = self.model_target(next_states, training=True)

            # Compute Q targets for current states
            q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))

            # Compute loss
            loss = self.loss(q_expected, q_targets)

        # Minimize the loss
        gradients = tape.gradient(loss, self.model_local.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model_local.trainable_variables))

        # Update target network
        self.soft_update()

    def soft_update(self):
        """Soft update model parameters.

        The model is update using:
            θ_target = τ * θ_local + (1 - τ) * θ_target

        """

        # Instantiate weight list
        new_weights = []

        # Apply soft update
        for weights in self.model_local.get_weights():
            new_weights.append(self.tau * weights + (1.0 - self.tau) * weights)

        # Set new weights
        self.model_target.set_weights(new_weights)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.

        Args:
            action_size: Integer. Dimension of each action
            buffer_size: Integer. Maximum size of buffer
            batch_size: Integer. Size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def __str__(self):
        pass

    def __repr__(self):
        pass

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory.

        Args:
            state: The previous state of the environment
            action: Integer. Previous action selected by the agent
            reward: Float. Reward value
            next_state: The current state of the environment
            done: Boolean. Whether the episode is complete
        """
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.vstack([e.action for e in experiences if e is not None])
        rewards = np.vstack([e.reward for e in experiences if e is not None])
        next_states = np.vstack([e.next_state for e in experiences if e is not None])
        dones = np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)

        return states, actions, rewards, next_states, dones

