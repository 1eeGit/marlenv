import marlenv
import gym

import matplotlib.pyplot as plt


def visualize_env(fig):
    pass


if __name__ == '__main__':
    # Plot env
    fig = plt
    env = gym.make("Snake-v1", num_snakes=1, width=30, height=30, disable_env_checker=True)
    #env = marlenv.wrappers.SingleAgent(env)
    #env = gym.make("Snake-v1", num_snakes=2, disable_env_checker=True)
    obs = env.reset()
    done = False
    episode_reward = 0.0
    while not done:
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        """
        Default: 'ascii' --> image in command line
                 'rgb_array' --> (grid_size, grid_size, 3) array
                 'gif' --> doesn't render, then saves .gif video (still unclear where)
                 
        Recommended: don't render during training, then 'rgb_array'. We will provide a helper function to visualize
                     quite soon!
        """
        env.render(mode="rgb_array")
    print(episode_reward)
