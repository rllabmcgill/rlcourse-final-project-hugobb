import numpy as np
import curses
from ..agent import Agent

COLOR = [curses.COLOR_RED, curses.COLOR_BLUE, curses.COLOR_GREEN]

class MultiAgent(object):
    def __init__(self, action_space, n_agents, n_landmarks=2, grid_size=(5,5)):
        self.p_failure = 0.1
        self.grid_size = grid_size
        self.action_space = action_space
        self.pos_init = lambda: (np.random.randint(self.grid_size[0]), np.random.randint(self.grid_size[1]))
        self.n_landmarks = n_landmarks
        self.n_agents = n_agents
        self.done = [False]*n_agents

        self.window = None

    def render(self, timeout=0):
        if self.window is None:
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

        self.window.timeout(timeout)
        self.pad.clear()
        self.pad.border()
        for i, s in enumerate(self.states):
            x, y = s
            self.pad.addstr(x+1, y+1, 'A', curses.color_pair(i+1))
        for i, l in enumerate(self.landmarks):
            x, y = l
            self.pad.addstr(x+1, y+1, 'R', curses.color_pair(i+1))
        self.pad.refresh(0,0, 0,0, 100,100)

    def reset(self):
        self.goals = np.random.choice(self.n_landmarks, size=self.n_agents)
        self.landmarks = []
        for i in range(self.n_landmarks):
            x, y = self.pos_init()
            self.landmarks.append((x,y))
        self.states = []
        for i in range(self.n_agents):
            # we don't want the agent to start on a landmark.
            start_pos = self.landmarks[0]
            while start_pos in self.landmarks:
                start_pos = self.pos_init()
            self.states.append(start_pos)
            self.done[i] = False
        observations = self.get_observation()
        return observations

    def step(self, actions):
        reward = [0]*self.n_agents
        info = None
        for i in range(self.n_agents):
            if self.done[i]:
                continue

            action = actions[i]
            x,y = self.states[i]
            if action == 'left':
                if y > 0:
                    y = y - 1
            elif action == 'right':
                if y < self.grid_size[1]-1:
                    y = y + 1
            elif action == 'up':
                if x > 0:
                    x = x - 1
            elif action == 'down':
                if x < self.grid_size[0]-1:
                    x = x + 1
            else:
                raise ValueError()
            self.states[i] = (x,y)

            if self.states[i] == self.landmarks[self.goals[i]]:
                self.done[i] = True
                reward[i] = 1

        observations = self.get_observation()
        return observations, reward, self.done, info


    def get_observation(self):
        observations = []
        for i in range(self.n_agents):
            o = self.states[i] + sum(self.landmarks, ()) + (self.goals[i],)
            observations.append(o)
        return observations


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
