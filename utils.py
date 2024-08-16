import os
import time

import numpy as np

import pygame


class BasicVideoRecorder:
    def __init__(self, data_dir):
        self.frames = []
        self.actions = []
        self.timestep = 0
        self.trajectory_offset = 0
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def put(self, obs, act):
        self.frames.append(obs)
        self.actions.append(act)
        self.timestep += 1

    def store_and_reset(self):
        np.savez_compressed(f"{self.data_dir}/example_{self.trajectory_offset}.npz", obs=np.array(self.frames), act=np.array(self.actions), num_steps=self.timestep)
        self.frames = []
        self.actions = []
        self.timestep = 0
        self.trajectory_offset += 1


def visualize_env(screen, frame, upscale_factor=20, fps=10):
    frame = np.repeat(frame, upscale_factor, axis=0)
    frame = np.repeat(frame, upscale_factor, axis=1)
    screen.blit(pygame.surfarray.make_surface(frame), (0, 0))
    pygame.display.update()
    time.sleep(1 / fps)
