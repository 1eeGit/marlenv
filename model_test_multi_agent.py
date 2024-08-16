import marlenv
import marlenv.wrappers as wrappers
import gym
import numpy as np

import pygame
from utils import BasicVideoRecorder, visualize_env



import torch 
import DQN_Linear_agent
import DQN_Linear_model
from DQN_Linear_model import Linear_QNet, QTrainer
from DQN_Linear_agent import DQNAgent


snake_1 = Linear_QNet(input_dim=3200)
snake_2 = Linear_QNet(input_dim=3200)
snake_1.load_state_dict(torch.load('/home/lee/Downloads/dqn_model_new.pth'))
snake_2.load_state_dict(torch.load('/home/lee/Downloads/dqn_model_new.pth'))

actions = {
    'd': 1,
    'a': 2
}

RECORD = False


def handle_inputs():
    """Control the agent using a/d keys"""
    action = 0
    close_request = False
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            try:
                if event.unicode in actions.keys():
                    action = int(actions[event.unicode])
                    break
                elif event.unicode == '\x1b':
                    close_request = True
            except KeyError:
                print("invalid input")

    return action, close_request


if __name__ == '__main__':
    # Plot env
    screen = pygame.display.set_mode((600, 600))

    env = gym.make("Snake-v1", num_snakes=2, width=30, height=30)
    env = wrappers.SingleMultiAgent(env)
    obs = env.reset()
    done = [False, False]
    episode_reward = np.array([0.0, 0.0])
    while not all(done):
        # handle_inputs() is needed ONLY IF you want to play the game with keyboard
        # action_1, close_request = handle_inputs()
        action_1 = snake_1.get_action(snake_1, obs[0])
        action_2 = snake_2.get_action(snake_2, obs[1])
        # if close_request:
            # break
        obs, reward, done, _ = env.step([action_1, action_2])
        episode_reward += np.array(reward)
        # Render
        current_frame = env.render(mode="rgb_array")
        visualize_env(screen, frame=current_frame)

    print(episode_reward)
