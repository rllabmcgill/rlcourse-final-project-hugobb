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
        assert(n_agents > 1 and n_landmarks > 0)
        self.vocab_size = vocab_size
        self.grid_size = grid_size
        self.n_landmarks = n_landmarks
        self.n_agents = n_agents

        # the following values are precomputed to center observations
        self._mean_code_agent = np.mean(range(n_agents))
        self._mean_code_grid = (np.mean(range(self.grid_size[0])), 
                                np.mean(range(self.grid_size[1])))
        self._mean_code_lmark = np.mean(range(n_landmarks))

        # the following dicts contains precomputed codes
        self._speech_codes = self._compute_oh_code(vocab_size)
        self._agent_codes = self._compute_oh_code(n_agents)
        self._landmark_codes = self._compute_oh_code(n_landmarks)

        # start episode
        self.reset()

    def _compute_oh_code(self, V):
        """ 
        compute a dict which maps integers from 0..V to one hot encoded vectors
        """
        code = {}
        for e in range(V):
            codeword = np.zeros(V)
            codeword[e] = 1
            code[e] = codeword
        return code

    def _pos_init(self):
        return (np.random.randint(self.grid_size[0]),
                np.random.randint(self.grid_size[1]))

    def reset(self):
        """
        create a new episode

        instantiate landmarks, agents, goals
        """
        self.done = False
        self.reward = False
        self.current_speeches = [0,]*self.n_agents # TODO: change that?
        # following variable indicates whether agent has reached his goal, 
        # i.e. whether he has made ANOTHER agent reach a landmark
        self.already_reached = [False]*self.n_agents

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
        observations = [self._get_observation(i) for i in range(self.n_agents)]
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
        """
        compute reward and done flag for all agents
        The reward is given only once to an agent when he made the other agent
        reach the goal.
        Episode is done when all goals have been reached.
        """
        reward = 0
        for i in range(self.n_agents):
            target_agent, target_lmark = self.goals[i]
            if (self.agent_pos[target_agent] == self.landmarks[target_lmark] and
                not self.already_reached[i]):
                reward += 1
                self.already_reached[i] = True
        done = all(self.already_reached)
        return reward, done

    def step(self, actions, speeches):
        self.current_speeches = [int(s) for s in speeches]

        if not self.done:
            for i in range(self.n_agents):
                self._move(i, actions[i])

        self.reward, self.done = self._get_reward_and_done()
        
        observations = []
        for i in range(self.n_agents):
            observations.append(self._get_observation(i))
        # cooperation: all rewards are the same for every agent
        rewards = [self.reward for _ in range(self.n_agents)]
        done = [self.done for _ in range(self.n_agents)]

        return observations, rewards, done, None

    def _encode_id(self, i):
        """ encode list containing id of the agent """
        return [self._agent_codes[i]]

    def _encode_pos(self, pos):
        """ encode list containing info of position """
        m_0, m_1 = self._mean_code_grid
        code_x = pos[0] - m_0
        code_y = pos[1] - m_1
        return [np.asarray([code_x, code_y])]

    def _encode_goal(self, goal): 
        agent, lmark = goal
        code_agent = self._agent_codes[agent]
        code_lmark = self._landmark_codes[agent]
        return [code_agent, code_lmark]

    def observation_space_size(self):
        return len(self._get_observation(0))

    def action_space_size(self):
        return 4

    def _encode_speech(self, s):
        """ encode s, a token from {0..self.vocab_size-1} """
        return [self._speech_codes[s]]

    def _get_observation(self, i):
        """ return a list containing:
        - id of the agent (i)
        - goal: id of the agent j and landmark k 
        - position of the agent itself
        - position of all the landmarks
        """
        # o_i(t) = [i, x_i, c_1..N, m_i, g_i]$
        o = self._encode_id(i)
        o += self._encode_goal(self.goals[i])
        for pos in [self.agent_pos[i]] + self.landmarks:
            o += self._encode_pos(pos)
        for s in self.current_speeches:
            o += self._encode_speech(s)
        return np.concatenate(o)
    
    def render(self):
        arr = np.chararray(self.grid_size)
        print "Render environment: (agents: a..z, landmarks 1..9, agents over land: A..Z)"
        nothing_char = '.'
        arr[:] = nothing_char
        for i, pos in enumerate(self.landmarks):
            arr[pos[0], pos[1]] = i+1
        for i, pos in enumerate(self.agent_pos):
            if arr[pos[0], pos[1]] == nothing_char:
                arr[pos[0], pos[1]] = chr(97+i)
            else:
                arr[pos[0], pos[1]] = chr(65+i)
        # pretty print array
        for i in arr:
            for j in i:
                print j,
            print ""
        print "communication:", self.current_speeches
        print "reward, done:", self.reward, self.done
