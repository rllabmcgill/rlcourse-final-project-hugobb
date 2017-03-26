import numpy as np
import curses
from ..agent import Agent

COLOR = [curses.COLOR_RED, curses.COLOR_BLUE, curses.COLOR_GREEN]

class MultiAgent(object):
    def __init__(self, n_agents=2, n_landmarks=2, grid_size=(5,5), agent=Agent, rendering=False):
        self.n_landmarks = n_landmarks
        self.p_failure = 0.1
        self.grid_size = grid_size
        self.state_space = self.grid_size*(n_agents + n_landmarks)
        self.action_space = ['left', 'right', 'up', 'down']
        self.pos_init = lambda: (np.random.randint(self.grid_size[0]), np.random.randint(self.grid_size[1]))
        self.goals = [0,1]
        self.agents = []
        for i in range(n_agents):
            a = agent(self.state_space, self.action_space, goal=self.goals[i])
            self.agents.append(a)

        if rendering:
            self.window = curses.initscr()
            curses.noecho()
            curses.cbreak()
            self.window.keypad(1)
            curses.curs_set(0)
            curses.start_color()
            self.pad = curses.newpad(self.grid_size[0]+2, self.grid_size[1]+2)
            self.window.addstr(0,15, 'Press Q to quit.')
            for i in range(3):
                curses.init_pair(i+1, COLOR[i%len(COLOR)], curses.COLOR_BLACK)

    def render(self, timeout=0):
        self.window.timeout(timeout)
        self.pad.clear()
        self.pad.border()
        for i, a in enumerate(self.agents):
            x, y = a.state
            self.pad.addstr(x+1, y+1, 'A', curses.color_pair(i+1))
        for i, l in enumerate(self.landmarks):
            x, y = l
            self.pad.addstr(x+1, y+1, 'R', curses.color_pair(i+1))
        self.pad.refresh(0,0, 0,0, 100,100)

    def reset(self):
        self.landmarks = []
        for i in range(self.n_landmarks):
            x, y = np.random.randint(self.grid_size[0]), np.random.randint(self.grid_size[1])
            self.landmarks.append((x,y))
        for a in self.agents:
            a.done = False
            a.state = self.pos_init()

    def step(self, agent, action):
        reward = 0
        info = None
        if agent.done:
            return reward, agent.done, info

        x,y = agent.state
        if action == 'left':
            if y > 0:
                y = y - 1
        elif action == 'right':
            if y < self.state_space[1]-1:
                y = y + 1
        elif action == 'up':
            if x > 0:
                x = x - 1
        elif action == 'down':
            if x < self.state_space[0]-1:
                x = x + 1
        else:
            raise ValueError()
        agent.state = (x, y)

        if agent.state == self.landmarks[agent.goal]:
            agent.done = True
            reward = 1

        return reward, agent.done, info

    def get_observation(self):
        observation = []
        for a in self.agents:
            observation += a.state
        for l in self.landmarks:
            observation += l
        return tuple(observation)

    def plot_policy(self, q):
        s = ''
        idx = q.argmax(axis=-1)
        nrows, ncols = self.grid_size[0], self.grid_size[1]
        for i in range(nrows):
            for j in range(ncols):
                if idx[i,j] == 0:
                    s += u'\u2190'
                elif idx[i,j] == 1:
                    s += u'\u2192'
                elif idx[i,j] == 2:
                    s += u'\u2191'
                elif idx[i,j] == 3:
                    s += u'\u2193'
                else:
                    raise ValueError()
                s += '\t'
            s += '\n'

        print s
