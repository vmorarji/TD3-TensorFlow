import numpy as np
import tensorflow as tf
from tensorflow import keras
import datetime as dt

class Actor(keras.Model):
	"""Creates an actor network"""
	def __init__(self, state_dim, action_dim, max_action):

		"""
		Args:
			state_dim: The dimensions of the state the environment will produce. 
				The input for the network.
			action_dim: The dimensions of the actions the environment can take.
				The output for the network.
			max_action: The maximum possible action that the environment can have
				for one particular action. The output is scaled following the 
				tanh activation function.
		"""
		super(Actor, self).__init__()
		self.layer_1 = keras.layers.Dense(state_dim, activation='relu', 
		                                  kernel_initializer=tf.keras.initializers.VarianceScaling(
		                                  	scale=1./3., distribution = 'uniform'))
		self.layer_2 = keras.layers.Dense(400, activation='relu',
		                                 kernel_initializer=tf.keras.initializers.VarianceScaling(
		                                 	scale=1./3., distribution = 'uniform'))
		self.layer_3 = keras.layers.Dense(300, activation='relu',
		                                 kernel_initializer=tf.keras.initializers.VarianceScaling(
		                                 	scale=1./3., distribution = 'uniform'))
		self.layer_4 = keras.layers.Dense(action_dim, activation='tanh',
		                                 kernel_initializer=tf.random_uniform_initializer(
		                                 	minval=-3e-3, maxval=3e-3))
		self.max_action = max_action

	def call(self, obs):
		x = self.layer_1(obs)
		x = self.layer_2(x)
		x = self.layer_3(x)
		x = self.layer_4(x)
		x = x * self.max_action
		return x   


class Critic(keras.Model):
	"""Creates two critic networks"""
	def __init__(self, state_dim, action_dim):
		"""
		Args:
			state_dim: The dimensions of the state the environment will produce. 
				The first input for the network.
			action_dim: The dimensions of the actions the environment can take.
				The second input for the network.
		"""
		super(Critic, self).__init__()
		# The First Critic NN
		self.layer_1 = keras.layers.Dense(state_dim + action_dim, activation='relu',
		                                 kernel_initializer=tf.keras.initializers.VarianceScaling(
		                                 	scale=1./3., distribution = 'uniform'))
		self.layer_2 = keras.layers.Dense(400, activation='relu',
		                                 kernel_initializer=tf.keras.initializers.VarianceScaling(
		                                 	scale=1./3., distribution = 'uniform'))
		self.layer_3 = keras.layers.Dense(300, activation='relu',
		                                 kernel_initializer=tf.keras.initializers.VarianceScaling(
		                                 	scale=1./3., distribution = 'uniform'))
		self.layer_4 = keras.layers.Dense(1, kernel_initializer=tf.random_uniform_initializer(
			minval=-3e-3, maxval=3e-3))
		# The Second Critic NN
		self.layer_5 = keras.layers.Dense(state_dim + action_dim, activation='relu',
		                                 kernel_initializer=tf.keras.initializers.VarianceScaling(
		                                 	scale=1./3., distribution = 'uniform'))     
		self.layer_6 = keras.layers.Dense(400, activation='relu',
		                                 kernel_initializer=tf.keras.initializers.VarianceScaling(
		                                 	scale=1./3., distribution = 'uniform'))
		self.layer_7 = keras.layers.Dense(300, activation='relu',
		                                 kernel_initializer=tf.keras.initializers.VarianceScaling(
		                                 	scale=1./3., distribution = 'uniform'))
		self.layer_8 = keras.layers.Dense(1, kernel_initializer=tf.random_uniform_initializer(
			minval=-3e-3, maxval=3e-3))

	def call(self, obs, actions):
		x0 = tf.concat([obs, actions], 1)
		# forward propagate the first NN
		x1 = self.layer_1(x0)
		x1 = self.layer_2(x1)
		x1 = self.layer_3(x1)
		x1 = self.layer_4(x1)
		# forward propagate the second NN
		x2 = self.layer_5(x0)
		x2 = self.layer_6(x2)
		x2 = self.layer_7(x2)
		x2 = self.layer_8(x2)        
		return x1, x2

	def Q1(self, state, action):
		x0 = tf.concat([state, action], 1)
		x1 = self.layer_1(x0)
		x1 = self.layer_2(x1)
		x1 = self.layer_3(x1)
		x1 = self.layer_4(x1)
		return x1


class TD3(object):
	"""
		Addressing Function Approximation Error in Actor-Critic Methods"
		by Fujimoto et al. arxiv.org/abs/1802.09477 
	"""
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		current_time = None,
		summaries: bool = False,
		gamma = 0.99,
		tau = 0.005,
		noise_std = 0.2,
		noise_clip = 0.5,
		expl_noise = 0.1,
		actor_train_interval = 2,
		actor_lr = 3e-4,
		critic_lr = 3e-4,
		critic_loss_fn = None
	):

		"""
		Args:
			state_dim: The dimensions of the state the environment will produce. 
				This is the input for the Actor network and one of the inputs
				for the Critic network.
			action_dim: The dimensions of the actions the environment can take.
				This is the output for the Actor network and one of the inputs
				for the Critic network.
			max_action: The maximum possible action for the environment. Actions
				will be clipped by this value after noise is added.
			current_time: The date and time to use for folder creation.
			summaries: A bool to gather Tensorboard summaries.
			gamma: The discount factor for future rewards.
			tau: The factor that the target networks are soft updated.
			noise_std: The scale factor to add noise to learning.
			noise_clip: The maximum noise that can be added to actions during 
				learning,
			expl_noise: The scale factor for noise during action selection.
			actor_train_interval: The interval at which the Actor network
				is trained and the target networks are soft updated.
			actor_lr: The learning rate used for SGA of the Actor network.
			critic_lr: The learning rate used for SGD of the Critic network
			critic_loss_fn: The loss function of the Critic network. If none
				tf.keras.losses.Huber() is used.

		"""


		self.actor = Actor(state_dim, action_dim, max_action)
		self.actor_target = Actor(state_dim, action_dim, max_action)
		for t, e in zip(self.actor_target.trainable_variables, self.actor.trainable_variables):
			t.assign(e)
		self.actor_optimizer = keras.optimizers.Adam(lr=actor_lr)


		self.critic = Critic(state_dim, action_dim)
		self.critic_target = Critic(state_dim, action_dim)
		for t, e in zip(self.critic_target.trainable_variables, self.critic.trainable_variables):
			t.assign(e)
		self.critic_optimizer = keras.optimizers.Adam(lr=actor_lr)
		if critic_loss_fn is not None:
			self.critic_loss_fn = critic_loss_fn
		else:
			self.critic_loss_fn = tf.keras.losses.Huber()


		self.action_dim = action_dim
		self.max_action = max_action
		self.gamma = gamma
		self.tau = tau
		self.noise_std = noise_std
		self.noise_clip = noise_clip
		self.expl_noise = expl_noise
		self.actor_train_interval = actor_train_interval
		self.summaries = summaries
		if current_time is None:
			self.current_time = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
		else:
			self.current_time = current_time
		if self.summaries:
			self.train_writer = tf.summary.create_file_writer('./logs/' + self.current_time)



		self.train_it = 0

	def select_action(self, state, noise: bool = False):
		# Action selection by the actor_network.
		state = state.reshape(1, -1)
		action = self.actor.call(state)[0].numpy()
		if noise:
			noise = tf.random.normal(action.shape, mean=0, stddev=self.expl_noise)
			action = tf.clip_by_value(action + noise, -self.max_action, self.max_action)
		return action



	def train(self, replay_buffer, batch_size=128):
		# training of the Actor and Critic networks.
		self.train_it += 1

		# create a sample of transitions
		batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(batch_size)

		# calculate a' and add noise
		next_actions = self.actor_target.call(batch_next_states)

		noise = tf.random.normal(next_actions.shape, mean=0, stddev=self.noise_std)
		noise = tf.clip_by_value(noise, -self.noise_clip, self.noise_clip)
		noisy_next_actions = tf.clip_by_value(next_actions + noise, -self.max_action, self.max_action)

		# calculate the min(Q(s', a')) from the two critic target networks
		target_q1, target_q2 = self.critic_target.call(batch_next_states, noisy_next_actions)
		target_q = tf.minimum(target_q1, target_q2)

		# calculate the target Q(s, a)
		td_targets = tf.stop_gradient(batch_rewards + (1 - batch_dones) * self.gamma * target_q)

		# Use gradient descent on the critic network
		trainable_critic_variables = self.critic.trainable_variables

		with tf.GradientTape(watch_accessed_variables=False) as tape:
			tape.watch(trainable_critic_variables)
			model_q1, model_q2 = self.critic(batch_states, batch_actions)
			critic_loss = (self.critic_loss_fn(td_targets, model_q1) + self.critic_loss_fn(td_targets, model_q2))
		critic_grads = tape.gradient(critic_loss, trainable_critic_variables)
		self.critic_optimizer.apply_gradients(zip(critic_grads, trainable_critic_variables))

		# create tensorboard summaries
		if self.summaries:
			if self.train_it % 100 == 0:
				td_error_1 = td_targets - model_q1
				td_error_2 = td_targets - model_q2
				with self.train_writer.as_default():
					tf.summary.scalar('td_target_mean', tf.reduce_mean(td_targets), step = self.train_it)
					tf.summary.scalar('td_target_max', tf.reduce_max(td_targets), step = self.train_it)
					tf.summary.scalar('td_target_min', tf.reduce_min(td_targets), step = self.train_it)

					tf.summary.scalar('pred_mean_1', tf.reduce_mean(model_q1), step = self.train_it)
					tf.summary.scalar('pred_max_1', tf.reduce_max(model_q1), step = self.train_it)
					tf.summary.scalar('pred_min_1', tf.reduce_min(model_q1), step = self.train_it)

					tf.summary.scalar('pred_mean_2', tf.reduce_mean(model_q2), step = self.train_it)
					tf.summary.scalar('pred_max_2', tf.reduce_max(model_q2), step = self.train_it)
					tf.summary.scalar('pred_min_2', tf.reduce_min(model_q2), step = self.train_it)

					tf.summary.scalar('td_error_mean_1', tf.reduce_mean(td_error_1), step = self.train_it)
					tf.summary.scalar('td_error_mean_abs_1', tf.reduce_mean(tf.abs(td_error_1)), step = self.train_it)
					tf.summary.scalar('td_error_max_1', tf.reduce_max(td_error_1), step = self.train_it)
					tf.summary.scalar('td_error_min_1', tf.reduce_min(td_error_1), step = self.train_it)

					tf.summary.scalar('td_error_mean_2', tf.reduce_mean(td_error_2), step = self.train_it)
					tf.summary.scalar('td_error_mean_abs_2', tf.reduce_mean(tf.abs(td_error_2)), step = self.train_it)
					tf.summary.scalar('td_error_max_2', tf.reduce_max(td_error_2), step = self.train_it)
					tf.summary.scalar('td_error_min_2', tf.reduce_min(td_error_2), step = self.train_it)

					tf.summary.histogram('td_targets_hist', td_targets, step = self.train_it)
					tf.summary.histogram('td_error_hist_1', td_error_1, step = self.train_it)
					tf.summary.histogram('td_error_hist_2', td_error_2, step = self.train_it)
					tf.summary.histogram('pred_hist_1', model_q1, step = self.train_it)
					tf.summary.histogram('pred_hist_2', model_q2, step = self.train_it)



		# Use gradient ascent on the actor network at a set interval
		if self.train_it % self.actor_train_interval == 0:
			trainable_actor_variables = self.actor.trainable_variables

			with tf.GradientTape(watch_accessed_variables=False) as tape:
				tape.watch(trainable_actor_variables)
				actor_loss = -tf.reduce_mean(self.critic.Q1(batch_states, self.actor(batch_states)))
			actor_grads = tape.gradient(actor_loss, trainable_actor_variables)
			self.actor_optimizer.apply_gradients(zip(actor_grads, trainable_actor_variables))

			# update the weights in the critic and actor target models
			for t, e in zip(self.critic_target.trainable_variables, self.critic.trainable_variables):
				t.assign(t * (1 - self.tau) + e * self.tau)

			for t, e in zip(self.actor_target.trainable_variables, self.actor.trainable_variables):
				t.assign(t * (1 - self.tau) + e * self.tau)

			# create tensorboard summaries
			if self.summaries:
				if self.train_it % 100 == 0:
					with self.train_writer.as_default():
						tf.summary.scalar('actor_loss', actor_loss, step = self.train_it)


	def save(self, steps):
		# Save the weights of all the models.
		self.actor.save_weights('./models/{}/actor_{}'.format(self.current_time, steps))
		self.actor_target.save_weights('./models/{}/actor_target_{}'.format(self.current_time, steps))

		self.critic.save_weights('./models/{}/critic_{}'.format(self.current_time, steps))
		self.critic_target.save_weights('./models/{}/critic_target_{}'.format(self.current_time, steps))
