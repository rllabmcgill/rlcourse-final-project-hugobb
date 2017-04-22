import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
plt.style.use('ggplot')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='The path to the directory containing the results.', type=str, default='results/')
    parser.add_argument('-o', '--output', help='The path where to save the plots.', type=str, default='report/')
    args = parser.parse_args()

    path = args.input
    output_path = args.output

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    mlp_results = np.loadtxt(os.path.join(path,'multiagent_mdp_mlp/results.csv'), delimiter=',', skiprows=1, usecols=(2,5))
    recurrent_results = np.loadtxt(os.path.join(path,'multiagent_mdp_recurrent/results.csv'), delimiter=',', skiprows=1, usecols=(2,5))

    plt.figure(1, figsize=(8,4))
    plt.subplot(121)
    plt.plot(mlp_results[:,0])
    plt.plot(recurrent_results[:,0])
    plt.title('Average length of an episode')
    plt.legend(['mlp', 'recurrent'])
    plt.xlabel('number of epochs')
    plt.ylabel('Length of episode')
    plt.subplot(122)
    plt.plot(mlp_results[:,1])
    plt.plot(recurrent_results[:,1])
    plt.title('Average reward per episode')
    plt.legend(['mlp', 'recurrent'])
    plt.xlabel('number of epochs')
    plt.ylabel('Reward')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path,'multiagent_mdp_results.png'))
