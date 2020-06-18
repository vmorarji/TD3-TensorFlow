import numpy as np

class ReplayBuffer(object):
    # The memory to store transitions as the agent plays the environment
    def __init__(self, max_size=1e6):
        """
        Args:
            max_size: The total transitions the agent can store without
                deleting older transitions. 
        """
        self.storage = []
        self.max_size = max_size
        self.ptr = 0
        
    def add(self, transition):
        """
        Store transitions in the memory buffer.
        The Order is state, next_state, actions, reward, done.
        """
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = transition
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(transition)
            
    def sample(self, batch_size):
        """ Retrieve samples from the memory buffer
        Args:
            batch_size: the amount of transitions to be randomly 
                sampled at one time. 
        """
        ind = np.random.randint(0, len(self.storage),  size=batch_size)
        batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = [], [], [], [], []
        for i in ind:
            state, next_state, action, reward, done = self.storage[i]
            batch_states.append(np.array(state, copy=False))
            batch_next_states.append(np.array(next_state, copy=False))
            batch_actions.append(np.array(action, copy=False))
            batch_rewards.append(np.array(reward, copy=False))
            batch_dones.append(np.array(done, copy=False))
        return np.array(batch_states), np.array(batch_next_states), np.array(batch_actions), np.array(batch_rewards).reshape(-1, 1), np.array(batch_dones).reshape(-1, 1)