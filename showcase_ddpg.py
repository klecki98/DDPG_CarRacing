from agent import Agent
import gymnasium as gym
from read_data import reformat_observation

def visualize(actor, env_id):
    from time import sleep
    for _ in range(5):
        if env_id == "CarRacing-v2":
            env = gym.make(env_id, render_mode='human', domain_randomize=False)
            observation, _ = env.reset()
            observation = reformat_observation(observation)
        else:
            env = gym.make(env_id, render_mode='human')
            observation, _ = env.reset()
        env.render()
        for i in range(1_000):
            action = actor.choose_action_no_noise(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            done = truncated or terminated
            if env_id == "CarRacing-v2":
                observation = reformat_observation(observation)
            # print(observation[0])
            # env.render()
            # sleep(0.002)
            if done:
                print("Episode finished after {} timesteps".format(i + 1))
                env.close()
                break


if __name__ == '__main__':
    env_id = 'CarRacing-v2'
    env = gym.make(env_id)
    agent = Agent(alpha=0.0001, beta=0.001,
                    input_dims=20, tau=0.001,
                    batch_size=64, hidden_dims=[400,300],
                    n_actions=env.action_space.shape[0])
    agent.load_networks()
    visualize(agent, env_id)
