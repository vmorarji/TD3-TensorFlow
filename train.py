import gym
import pybullet_envs
import os
from Model import TD3
from ReplayBuffer import ReplayBuffer
import datetime as dt
import tensorflow as tf


current_time = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
if not os.path.exists('./logs/' + current_time):
	os.makedirs('./logs/' + current_time)

if not os.path.exists('./models/' + current_time):
	os.makedirs('./models/' + current_time)

# initialise the environment
env = gym.make("AntBulletEnv-v0")
# env = wrappers.Monitor(env, save_dir, force = True) 
env.seed(0)
action_dim = env.action_space.shape[0]
state_dim = env.observation_space.shape[0]
max_action = float(env.action_space.high[0])
result_writer = tf.summary.create_file_writer('./logs/' + current_time)


def evaluate_policy(policy, eval_episodes=10):
	# during training the policy will be evaluated without noise
	avg_reward = 0.
	for _ in range(eval_episodes):
		state = env.reset()
		done = False
		while not done:
			action = policy.select_action(state)
			state, reward, done, _ = env.step(action)
			avg_reward += reward
	avg_reward /= eval_episodes
	print ("---------------------------------------")
	print ("Average Reward over the Evaluation Step: %f" % (avg_reward))
	print ("---------------------------------------")
	return avg_reward


# initialise the replay buffer
memory = ReplayBuffer()
# initialise the policy
policy = TD3(state_dim, action_dim, max_action, current_time=current_time, summaries=True)



max_timesteps = 2e6
start_timesteps = 1e4
total_timesteps = 0
eval_freq = 5e3
save_freq = 1e5
eval_counter = 0
episode_num = 0
episode_reward = 0
done = True

while total_timesteps < max_timesteps:

	if done:

		# print the results at the end of the episode
		if total_timesteps != 0:
			print('Episode: {}, Total Timesteps: {}, Episode Reward: {:.2f}'.format(
				episode_num,
				total_timesteps,
				episode_reward
				))
			with result_writer.as_default():
				tf.summary.scalar('total_reward', episode_reward, step = episode_num)

		if eval_counter > eval_freq:
			eval_counter %= eval_freq
			evaluate_policy(policy)

		state = env.reset()

		done = False
		episode_reward = 0
		episode_timesteps = 0
		episode_num += 1

	# the environment will play the initial episodes randomly
	if total_timesteps < start_timesteps:
		action = env.action_space.sample()
	else: # select an action from the actor network with noise
		action = policy.select_action(state, noise=True)

	# the agent plays the action
	next_state, reward, done, info = env.step(action)

	# add to the total episode reward
	episode_reward += reward

	# check if the episode is done
	done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)

	# add to the memory buffer
	memory.add((state, next_state, action, reward, done_bool))

	# update the state, episode timestep and total timestep
	state = next_state
	episode_timesteps += 1
	total_timesteps += 1
	eval_counter += 1

	# train after the first episode
	if total_timesteps > start_timesteps:
		policy.train(memory)
    
    # save the model    
	if total_timesteps % save_freq == 0:
		policy.save(int(total_timesteps / save_freq))