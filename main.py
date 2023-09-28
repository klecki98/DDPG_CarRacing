import gymnasium as gym
import numpy as np
from agent import Agent
from read_data import reformat_observation
import os

if __name__ == '__main__':
    '''
    Sensitivity to:
    alpha = 1e-4, 1e-3, 1e-5
    beta = 1e-3, 1e-2, 1e-4
    tau = 1e-3, 1e-2, 1e-4
    batch_size = 64, 128, 32
    hidden_dims = [400,300], [40, 30], [400]
    '''
    hidden_dims = [400, 300]
    env = gym.make("CarRacing-v2", domain_randomize=False)#'LunarLanderContinuous-v2')
    agent = Agent(alpha=0.0001, beta=0.001,
                    input_dims=20, tau=0.001,
                    batch_size=64, hidden_dims=hidden_dims,
                    n_actions=env.action_space.shape[0])
    # agent = Agent(alpha=alpha, beta=beta,
    #                 input_dims=20, tau=tau,
    #                 batch_size=batch_size, hidden_dims=hidden_dims,
    #                 n_actions=env.action_space.shape[0])
    n_games = 1000
    filename = 'CarRacing-v2_alpha_' + str(agent.alpha) + '_beta_' + \
               str(agent.beta) + '_tau_' + str(agent.tau) + '_batch_' + str(agent.batch_size) + \
               '_hidden_' + str(hidden_dims) + '_test_case_' + str(len(os.listdir('results'))+1)
    result_file = 'results/' + filename + '.txt'

    best_score = env.reward_range[0]
    score_history = []
    f = open(result_file, "w")
    f.close()

    for i in range(n_games):
        observation, info = env.reset()
        observation = reformat_observation(observation)
        done = False
        score = 0
        agent.noise.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done1, done2, info = env.step(action)
            observation_ = reformat_observation(observation_)
            done = done1 or done2
            if observation_[-4] == 0.0:
                reward -= 20
                done = True
            agent.save_transition(observation, action, reward, observation_, done)
            agent.learn()
            score += reward
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if len(score_history)>100:
                agent.save_networks()
                print(filename)
        f = open(result_file,"a")
        f.write(str(score)+"\n")
        f.close()
        print('episode ', i, 'score %.1f' % score,
                'average score %.1f' % avg_score)