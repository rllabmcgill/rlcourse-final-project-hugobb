import numpy as np

class GridWorld(object):
    def __init__(self, n_agents, grid_size=(5,5), max_length=10000):
        self.grid_size = grid_size
        self.action_space = ['left', 'right', 'up', 'down']
        self.max_length = max_length
        self.n_agents = n_agents
        return

    def reset(self):
        self.landmarks = [(np.random.randint(self.grid_size[0]), np.random.randint(self.grid_size[1]))]
        self.landmarks += [(np.random.randint(self.grid_size[0]), np.random.randint(self.grid_size[1]))]
        self.goal = np.random.randint(2, size=self.n_agents)
        self.steps = 0
        self.observation = []
        observation = []
        for i in range(self.n_agents):
            o = (np.random.randint(self.grid_size[0]), np.random.randint(self.grid_size[1]))
            self.observation.append(o)
        for i in range(self.n_agents):
            observation.append(sum(self.observation, ()) + sum(self.landmarks, ()) + tuple(self.goal))

        return observation

    def step(self, action):
        reward = [0]*self.n_agents
        info = [None]*self.n_agents
        done = [False]*self.n_agents
        observation = []
        for i, a in enumerate(action):
            x, y = self.observation[i]
            if a is None:
                continue

            a = self.action_space[a]
            if a == 'left':
                if y > 0:
                    y = y - 1

            elif a == 'right':
                if y < self.grid_size[1]-1:
                    y = y + 1

            elif a == 'up':
                if x > 0:
                    x = x - 1

            elif a == 'down':
                if x < self.grid_size[0]-1:
                    x = x + 1
            else:
                raise ValueError()

            o = (x,y)
            self.observation[i] = o
            if (x - self.landmarks[self.goal[i]][0])**2 + (y - self.landmarks[self.goal[i]][1])**2 == 0:
                done[i] = True
                reward[i] += 1
            elif self.steps > self.max_length:
                done = [True]*self.n_agents
        for i in range(self.n_agents):
            if action[i] is None:
                observation.append(None)
            else:
                observation.append(sum(self.observation, ()) + sum(self.landmarks, ()) + tuple(self.goal))
        self.steps += 1
        return observation, reward, done, info
