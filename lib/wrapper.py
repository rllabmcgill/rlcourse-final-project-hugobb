import numpy as np
from tqdm import tqdm

class EnvWrapper(object):
    def __init__(self, env, agents, seq_length=4, max_no_op=30,
                preprocess=lambda x: x, init_epsilon=1.,
                epsilon_decay=int(1e6), epsilon_min=0.8, update_frequency=4, batch_size=32):
        self.env = env
        self.num_actions = len(self.env.action_space)
        self.seq_length = seq_length
        self.max_no_op = max_no_op
        self.preprocess = preprocess
        self.epsilon_start = init_epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.epsilon = init_epsilon
        if self.epsilon_decay != 0:
            self.epsilon_rate = ((self.epsilon_start - self.epsilon_min) / self.epsilon_decay)
        else:
            self.epsilon_rate = 0

        self.update_frequency = update_frequency
        self.batch_size = batch_size
        self.agents = agents
        for a in self.agents:
            a.init(self.num_actions, self.seq_length, self.batch_size)

    def run_episode(self, max_steps, eps, mode='train'):
        frame = self.env.reset()
        if self.max_no_op > 0:
            random_actions = np.random.randint(self.max_no_op+1)
            for i in range(random_actions):
                frame, reward, done, info = self.env.step(0)

        loss = np.zeros(2)
        episode_reward = 0
        num_steps = 0
        done_all_agent = [False]*len(self.agents)
        action = []
        for i in range(len(self.agents)):
            action.append(np.random.randint(self.num_actions))
        last_frame = self.preprocess(frame)
        last_action = action
        while True:
            frame, reward, done, info = self.env.step(action)
            num_steps += 1
            episode_reward += np.sum(reward)
            reward = [np.sum(reward)]*len(self.agents)
            if mode == 'init' or mode == 'train':
                for i, agent in enumerate(self.agents):
                    if done[i]:
                        agent.replay_memory.append(last_frame[i], last_action[i], np.clip(reward[i],-1,1), True)
                        done_all_agent[i] = True
                        action[i] = None
            elif mode == 'test':
                for i, agent in enumerate(self.agents):
                    if done[i]:
                        done_all_agent[i] = True
                        action[i] = None
            if all(done_all_agent) or num_steps >= max_steps:
                break

            observation = self.preprocess(frame)
            if mode == 'test':
                for i, agent in enumerate(self.agents):
                    if not done_all_agent[i]:
                        agent.test_memory.append(last_frame[i], last_action[i], np.clip(reward[i],-1,1), False)
                        if num_steps >= self.seq_length:
                            state = agent.test_memory.get_state(observation[i], self.seq_length)
                            action[i] = agent.get_action(state, eps=eps)
                        else:
                            action[i] = np.random.randint(self.num_actions)
            elif mode == 'init':
                for i, agent in enumerate(self.agents):
                    if not done_all_agent[i]:
                        agent.replay_memory.append(last_frame[i], last_action[i], np.clip(reward[i],-1,1), False)
                        if num_steps >= self.seq_length:
                            state = agent.replay_memory.get_state(observation[i], self.seq_length)
                            action[i] = agent.get_action(state, eps=eps)
                        else:
                            action[i] = np.random.randint(self.num_actions)

            elif mode == 'train':
                self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_rate)
                for i, agent in enumerate(self.agents):
                    if not done_all_agent[i]:
                        agent.replay_memory.append(last_frame[i], last_action[i], np.clip(reward[i],-1,1), False)
                        if num_steps >= self.seq_length:
                            state = agent.replay_memory.get_state(observation[i], self.seq_length)
                            action[i] = agent.get_action(state, eps=self.epsilon)
                        else:
                            action[i] = np.random.randint(self.num_actions)

                        if num_steps % self.update_frequency == 0:
                            s, a, r, d = agent.replay_memory.sample(self.seq_length, self.batch_size)
                            loss[i] += agent.train(s, a, r, d)

            elif mode == 'render':
                self.env.render()
                self.test_memory.append(last_frame, last_action, np.clip(reward[i],-1,1), False)
                if num_steps >= self.seq_length:
                    state = self.test_memory.get_state(observation, self.seq_length)
                    action = self.agent.get_action(state, eps=eps)
                else:
                    action = np.random.randint(self.num_actions)

            elif mode == 'baseline':
                self.test_memory.append(last_frame, last_action, np.clip(reward[i],-1,1), False)
                if num_steps >= self.seq_length:
                    state = self.test_memory.get_state(observation, self.seq_length)
                    action = self.agent.get_action(state, eps=eps)
                else:
                    action = np.random.randint(self.num_actions)
            else:
                raise ValueError('Wrong mode, choose between init|train|test')

            last_frame = observation
            last_action = action

        return episode_reward, num_steps, loss

    def run_epoch(self, epoch_length, epsilon=None, mode='train'):
        num_episodes = 0
        steps_left = epoch_length
        pbar = tqdm(total=epoch_length)
        reward_per_episode = []
        steps_per_episode = []
        loss_per_episode = np.zeros(2)
        while steps_left > 0:
            episode_reward, num_steps, loss = self.run_episode(steps_left, eps=epsilon, mode=mode)
            loss_per_episode += loss
            num_episodes += 1
            reward_per_episode.append(episode_reward)
            steps_per_episode.append(num_steps)
            steps_left -= num_steps
            pbar.update(num_steps)
        pbar.close()

        if mode == 'train':
            return num_episodes, loss_per_episode/num_episodes, self.epsilon
        elif mode == 'test' or mode == 'baseline' or mode == 'render' or mode == 'init':
            return (num_episodes, np.mean(steps_per_episode), np.max(steps_per_episode),
                    np.sum(reward_per_episode), np.mean(reward_per_episode), np.max(reward_per_episode))
