from lib.env import GridWorld
from lib.wrapper import EnvWrapper
from lib.agent import DeepQAgent
import numpy as np
import os
import argparse
from nn import MLP

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--save', help='The path of the directory where to save results.',
                        type=str, default='results/multiagent_mdp_mlp')
    args = parser.parse_args()

    path = args.save

    replay_start_size = 25000
    train_epoch_length = 5000
    test_epoch_length = 5000
    n_epochs = 100
    n_agents = 2
    n_landmarks = 2
    state_space = (3*n_agents+2*n_landmarks,)
    mem_size = 25000
    norm = 4.0
    update_frequency = 20000

    agent1 = DeepQAgent(MLP(), state_space=state_space, update_frequency=update_frequency, norm=norm, memory_size=mem_size)
    agent2 = DeepQAgent(MLP(), state_space=state_space, update_frequency=update_frequency, norm=norm, memory_size=mem_size)

    env = GridWorld(2, max_length=50)
    env_wrapper = EnvWrapper(env, [agent1, agent2], epsilon_decay=int(5e4), epsilon_min=0.1)

    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(os.path.join(path,'params')):
        os.makedirs(os.path.join(path,'params'))

    results_file = open(os.path.join(path, 'results.csv'), 'w')
    results_file.write(\
        'epoch,num_episodes,mean_length,max_length,total_reward,max_reward,mean_reward\n')
    results_file.flush()

    learning_file = open(os.path.join(path, 'learning.csv'), 'w')
    learning_file.write('epoch,num_episodes,mean_loss,epsilon\n')
    learning_file.flush()

    results = env_wrapper.run_epoch(replay_start_size, mode='init', epsilon=1.)
    print "baseline: num episodes: %d, mean length: %d, max length: %d, total reward: %d, mean_reward: %.4f, max_reward: %d"%(results)
    for epoch in range(n_epochs):
        num_episodes, mean_loss, epsilon = env_wrapper.run_epoch(train_epoch_length, mode='train')
        print "epoch: %d,\tnum episodes: %d,\tepsilon: %.2f"%(
                epoch,num_episodes,epsilon)

        results = env_wrapper.run_epoch(test_epoch_length, mode='test', epsilon=0.05)
        print "epoch: %d, num episodes: %d, mean length: %d, max length: %.d, total reward: %d, mean_reward: %.4f, max_reward: %d"%(
                (epoch,)+results)

        out = "{},{},{},{},{},{},{}\n".format(epoch, results[0], results[1], results[2], results[3], results[4], results[5])
        results_file.write(out)
        results_file.flush()

        out = "{},{},{}\n".format(epoch, num_episodes, epsilon)
        learning_file.write(out)
        learning_file.flush()

        for a in env_wrapper.agents:
            a.save(os.path.join(path,'params/epoch_%d'%(epoch)))
