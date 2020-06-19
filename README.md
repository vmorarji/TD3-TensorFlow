# TD3-TensorFlow

### TD3 for OpenAI gym environments using Tensorflow

To address the function approximation error of Actor-Critic methods, [Fujimoto et al.,](https://arxiv.org/abs/1802.09477) proposed the Twin Delayed Deep Deterministic policy gradient method (TD3). 

Using a total of six neural networks, TD3 minimises the approximated Q-value by taking the minimum value from two Critic neural networks and uses this value to optimise the Actor network.

The [original implementation](https://github.com/sfujim/TD3) is written in Pytorch  and is the basis for the Tensorflow version written here.

### Requirements

Gym: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `pip install gym`

Pybullet: 	`pip install pybullet_envs`

The below line in `main.py` has been commented out. To enable recordings of certain episodes simply uncomment line 19.
```python
env = wrappers.Monitor(env, save_dir, force = True)
```
Note that one Windows machines ffmpeg will need to be saved in the same local directory as `main.py`.



### Results

After 1 million timesteps the model will achieve an average total reward of 2200 for "AntBulletEnv-v0".


![Alt Text](https://i.imgur.com/EURQj7q.gif)


The Actor loss has a local minimum around the -60 value where the agent will get stuck in a single position. Using Huber loss as opposed to MSE was instrumental in the agent progressing and achieving higher total rewards.

![Imgur](https://i.imgur.com/HosZ5xN.png)
