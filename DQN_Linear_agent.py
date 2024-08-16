### modified from: https://github.com/patrickloeber/snake-ai-pytorch

import torch
import random
import numpy as np
from collections import deque
from DQN_Linear_model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.state_size = np.prod(env.observation_space.shape)
        self.action_size = 2 # 0,1,2
        self.memory = deque(maxlen=100_000)
        self.gamma = 0.9  
        self.epsilon = 0
        self.learning_rate = 0.001
        self.n_games = 0
        self.record = 0

        self.model = Linear_QNet(input_dim=self.state_size)
        self.trainer = QTrainer(self.model, lr=self.learning_rate, gamma=self.gamma)
        self.scores = []
        self.mean_scores = []

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        if random.randint(0, 200) < self.epsilon:
            action = random.randint(0, 2)
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            action = torch.argmax(prediction).item()

        return action




    def remember(self, state, action, reward, next_state, done):
        """Store the experience in memory."""
        self.memory.append((state, action, reward, next_state, done))

    def train_short_memory(self, state, action, reward, next_state, done):
        state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float).unsqueeze(0)
        action_tensor = torch.tensor([[action]])
        reward_tensor = torch.tensor([[reward]])
        
        self.trainer.train_step(state_tensor, action_tensor, reward_tensor, next_state_tensor, done)

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float).unsqueeze(0)

            target = reward
            if not done:
                target = (reward + self.gamma * torch.max(self.model(next_state_tensor)).item())

            action_tensor = torch.tensor([[action]])
            reward_tensor = torch.tensor([[target]])

            self.trainer.train_step(state_tensor, action_tensor, reward_tensor, next_state_tensor, done)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, num_episodes=100):
        for e in range(num_episodes):
            state = self.env.reset()
            state = state.flatten()
            done = False
            total_score = 0

            while not done:
                action = self.get_action(state)
                next_state, reward, done_list, _ = self.env.step([action])
                next_state = next_state.flatten()

                if isinstance(reward, list):
                    reward = sum(reward)  
                
                self.remember(state, action, reward, next_state, done)
                self.train_short_memory(state, action, reward, next_state, done)
                
                state = next_state
                total_score += reward
            self.n_games += 1
            self.replay(32)
            self.scores.append(total_score)
            self.mean_scores.append(np.mean(self.scores[-100:]))
            plot(self.scores, self.mean_scores)


        # self.model.save('dqn_model.pth')




if __name__ == "__main__":
    import marlenv
    import gym
    from DQN_Linear_agent import DQNAgent 
    import matplotlib.pyplot as plt
    
    custom_rew = {
        'fruit': 2.0,
        'kill': 5.0,
        'lose': -10.0,
        'win': 10.0,
        'time': 0.5
        }


    env = gym.make('Snake-v1', num_fruits=4, num_snakes=1, reward_dict=custom_rew, disable_env_checker=True)
    # env = gym.make('Snake-v1')

    agent = DQNAgent(env)
    agent.train(num_episodes=500)
    print(agent.record)