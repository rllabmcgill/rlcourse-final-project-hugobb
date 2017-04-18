import numpy as np

class MultiAgentCoop(object):
    """
    Gridworld where agents get rewards when they make other agents reach goals
    Agents start in any position where there isn't a landmark
    i.e. there can be several agent per grid cell
    Landmarks are placed on distinct grid cells
    At each timestep, each agent takes an action and speak.
    Speech are visible to any other agent at the next timestep

    Every agent sees everyone in this variant (TODO: make it a choice)
    """
    def __init__(self, n_agents, n_landmarks, grid_size, vocab_size):
        """
        n_agents: 
        n_landmarks:
        grid_size:
            tuple (x_size, y_size)
        """
        assert(n_agents > 0 and n_landmarks > 0)
        self.vocab_size = vocab_size
        self.grid_size = grid_size
        self.n_landmarks = n_landmarks
        self.n_agents = n_agents

        # the following values are precomputed to center observations
        self._mean_code_agent = np.mean(range(n_agents))
        self._mean_code_grid = (np.mean(range(self.grid_size[0])), 
                                np.mean(range(self.grid_size[1])))
        self._mean_code_lmark = np.mean(range(n_landmarks))

        # start episode
        self.reset()


    def _pos_init(self):
        return (np.random.randint(self.grid_size[0]),
                np.random.randint(self.grid_size[1]))

    def reset(self):
        """
        create a new episode

        instantiate landmarks, agents, goals
        """
        self.done = [False]*self.n_agents
        # self.agents_coop[3] = (1,2) means that the goal of agent #3 is to make
        # agent #1 go to landmark #2
        # some constraints on goals: writing self.agents_coop[i] = (j,k):
        # * i!=j 
        # * j and k are sampled without replacement
        self.agents_coop = [] 
        self.goals = []

        lmarks = np.random.choice(self.n_landmarks, self.n_agents, replace=False)

        sample_agents = np.asarray(range(self.n_agents))
        # an agent can't be assigned a goal which concerns itself
        while np.any(np.asarray(range(self.n_agents)) == sample_agents):
            sample_agents = np.random.choice(self.n_agents,
                                             size=(self.n_agents,),
                                             replace=False)
        for i in range(self.n_agents):
            self.goals.append((sample_agents[i], lmarks[i])) 
            # old algorithm, didn't work
            #reinsert = False
            #if i in other_agent_possible:
            #    other_agent_possible.remove(i)
            #    reinsert = True
            #target_agent = np.random.choice(other_agent_possible)
            #print "for i, target j", i, ":", target_agent
            #self.goals.append((target_agent, lmarks[i]))
            #other_agent_possible.remove(target_agent)
            #if reinsert:
            #    other_agent_possible.append(i)
        ## all agents have been matched to other agents
        #print other_agent_possible
        #assert(len(other_agent_possible)==0)

        self.landmarks = []
        # it is OK if landmarks sometimes overlap
        for i in range(self.n_landmarks):
            x, y = self._pos_init()
            self.landmarks.append((x,y))

        self.agent_pos = []
        for i in range(self.n_agents):
            # we don't want any agent to start on a landmark
            # but let's assume that several agents can be on a grid cell
            start_pos = self.landmarks[0]
            while start_pos in self.landmarks:
                start_pos = self._pos_init()
            self.agent_pos.append(start_pos)
        observations = [self.get_observation(i) for i in range(self.n_agents)]
        return observations

    def _move(self, i, action):
        """ move agent i by doing action """
        x,y = self.agent_pos[i]
        if action == 0 or action == 'left':
            if x > 0:
                x = x - 1
        elif action == 1 or action == 'right':
            if x < self.grid_size[0]-1:
                x = x + 1
        elif action == 2 or action == 'down':
            if y > 0:
                y = y - 1
        elif action == 3 or action == 'up':
            if y < self.grid_size[1]-1:
                y = y + 1
        else:
            raise ValueError()
        self.agent_pos[i] = (x,y)

    def _get_reward_and_done(self):
        """ compute reward and done flag for all agents """
        done = [False]*self.n_agents
        reward = [0] * self.n_agents

        for i in range(self.n_agents):
            target_agent, target_lmark = self.goals[i]
            if self.agent_pos[target_agent] == self.landmarks[target_lmark]:
                done[i]
                return (1, True)
            else:
                return (0, False)

    def step(self, actions, speeches):
        self.current_speeches = speeches
        reward = []
        # first, all agents move except the ones that are done
        for i in range(self.n_agents):
            if self.done[i]:
                continue
            self._move(i, actions[i])

        # warning: the 2 loops matter
        # it's important to wait for all the moves before checking for rewards
        observations = []
        for i in range(self.n_agents):
            r, d = self._get_reward_and_done(i)
            self.done[i] = d
            reward.append(r)
            observations.append(self.get_observation(i))

        return observations, reward, self.done, None

    def _encode_id(self, i):
        """ encode list containing id of the agent """
        return [i]
        #return [i - self._mean_code_agent]

    def _encode_pos(self, pos):
        """ encode list containing info of position """
        return [pos[0], pos[1]]
        m_0, m_1 = self._mean_code_grid
        code_x = pos[0] - m_0
        code_y = pos[1] - m_1
        return [code_x, code_y]

    def _encode_goal(self, goal): 
        return [goal[0], goal[1]]
        agent, lmark = goal
        code_agent = agent - self._mean_code_agent
        code_lmark = lmark - self._mean_code_lmark
        return [code_agent, code_lmark]

    def observation_space_size(self):
        return len(get_observation(0))

    def action_space_size(self):
        return 4

    def get_observation(self, i):
        """ return a list containing:
        - id of the agent (i)
        - goal: id of the agent j and landmark k 
        - position of the agent itself
        - position of all the landmarks
        """
        # o_i(t) = [i, x_i, c_1..N, m_i, g_i]$
        o = self._encode_id(i)
        o += self._encode_goal(self.goals[i])
        for pos in self.agent_pos + self.landmarks:
            o.extend(self._encode_pos(pos))
        return o
