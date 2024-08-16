import numpy as np

import marlenv
import marlenv.wrappers as wrappers
import gym

import pygame

from utils import BasicVideoRecorder, visualize_env


actions = {
    'd': 1,
    'a': 2
}

RECORD = True


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
    if RECORD:
        recorder = BasicVideoRecorder(data_dir="trajectories")
    else:
        recorder = None

    env = gym.make("Snake-v1", num_snakes=1, width=30, height=30)
    env = wrappers.SingleAgent(env)
    obs = env.reset()
    done = False
    episode_reward = 0.0
    while not done:
        # handle_inputs() is needed ONLY IF you want to play the game with keyboard
        action, close_request = handle_inputs()
        if close_request:
            break
        obs, reward, done, _ = env.step(action)
        episode_reward += reward
        # Render
        current_frame = env.render(mode="rgb_array")
        visualize_env(screen, frame=current_frame)
        if RECORD:
            """
            By default, recorder will save the observation from the environment. If you want to use the RGB image,
            you should recorder.put(<image_var_name>, action) instead. 
            NOTE! In order to get the RGB image you will need to use mode='rgb_array' in env.render()!
            """
            recorder.put(obs, action)

    if RECORD:
        recorder.store_and_reset()

    print(episode_reward)
