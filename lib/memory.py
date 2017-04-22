import numpy as np

class Memory(object):
    def __init__(self, state_space, max_length=int(1e6)):
        self.max_length = max_length
        self.top = 0
        self.bottom = 0
        self.size = 0
        self.action = np.zeros(max_length, dtype='uint8')
        self.action[:] = 0
        self.state = np.zeros((max_length,) + state_space, dtype='uint8')
        self.state[:] = 0
        self.reward = np.zeros(max_length, dtype='uint8')
        self.reward[:] = 0
        self.done = np.zeros(max_length, dtype=bool)
        self.done[:] = False
        self.memory_full = False

    def append(self, state, action, reward, done):
        if action is None:
            return
        self.action[self.top] = action
        self.state[self.top] = state
        self.reward[self.top] = reward
        self.done[self.top] = done

        if self.size == self.max_length:
            self.bottom = (self.bottom + 1) % self.max_length
        else:
            self.size += 1
        self.top = (self.top + 1) % self.max_length

    def get_state(self, observation, seq_length):
        index = np.arange(self.top - seq_length + 1, self.top)

        state = np.empty((seq_length,)+self.state.shape[1:], dtype='uint8')
        state[0:seq_length - 1] = self.state.take(index, axis=0, mode='wrap')
        state[-1] = observation
        return state

    def sample(self, seq_length, batch_size=128):
        state = np.zeros((batch_size, seq_length+1) + self.state.shape[1:], dtype='uint8')
        action = np.zeros((batch_size,1), dtype='uint8')
        reward = np.zeros((batch_size,1), dtype='uint8')
        done = np.zeros((batch_size,1), dtype=bool)

        count = 0
        while count < batch_size:
            index = np.random.randint(self.bottom, self.bottom + self.size - seq_length)
            all_indices = np.arange(index, index + seq_length + 1)
            end_index = index + seq_length - 1
            if np.any(self.done.take(all_indices[0:-2], mode='wrap')):
                continue

            state[count] = self.state.take(all_indices, axis=0, mode='wrap')
            action[count] = self.action.take(end_index, mode='wrap')
            reward[count] = self.reward.take(end_index, mode='wrap')
            done[count] = self.done.take(end_index, mode='wrap')
            count += 1

        return state, action, reward, done
