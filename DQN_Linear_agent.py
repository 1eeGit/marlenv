### modified from: https://github.com/patrickloeber/snake-ai-pytorch

import torch
import random
import numpy as np
from collections import deque
from DQN_Linear_model import Linear_QNet, QTrainer
from helper import plot
import marlenv
import gym
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("GPU is available:", torch.cuda.is_available())

MAX_MEMORY = 100_000
BATCH_SIZE = 50 # 32, 100, 50
LR = 0.0005 # 0.001: lowering the learning rate

class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.state_size = np.prod(env.observation_space.shape)
        self.action_size = 2 # 0,1,2
        self.memory = deque(maxlen=MAX_MEMORY)
        self.gamma = 0.9  

        self.epsilon = 1.0
        self.epsilon_decay = 0.995  # 0.995 : raising the epsilon decay 
        self.epsilon_min = 0.5 # 0.1, 0.05

        self.learning_rate = LR
        self.n_games = 0
        self.record = 0

        

        self.model = Linear_QNet(input_dim=self.state_size).to(device)
        self.trainer = QTrainer(self.model, lr=self.learning_rate, gamma=self.gamma)
        self.scores = []
        self.mean_scores = []

    def get_action(self, state):
        # print(f"Current epsilon: {self.epsilon}")   
        if random.uniform(0, 1) < self.epsilon:
            action = random.randint(0, self.action_size - 1)
            # print(f"Exploring: Chose random action {action}")   
        else:
            state0 = torch.tensor(state, dtype=torch.float).to(device)
            prediction = self.model(state0)
            action = torch.argmax(prediction).item()
            # print(f"Exploiting: Chose action {action} based on prediction")   

        return action




    def remember(self, state, action, reward, next_state, done):
        """Store the experience in memory."""
        self.memory.append((state, action, reward, next_state, done))

    def train_short_memory(self, state, action, reward, next_state, done):
        state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(device)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float).unsqueeze(0).to(device)
        action_tensor = torch.tensor([[action]])
        reward_tensor = torch.tensor([[reward]])
        
        action_tensor = torch.tensor([[action]]).to(device)  
        reward_tensor = torch.tensor([[reward]]).to(device)

        self.trainer.train_step(state_tensor, action_tensor, reward_tensor, next_state_tensor, done)

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(device)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float).unsqueeze(0).to(device)

            target = reward
            if not done:
                target = (reward + self.gamma * torch.max(self.model(next_state_tensor)).item())

            action_tensor = torch.tensor([[action]]).to(device)
            reward_tensor = torch.tensor([[target]]).to(device)

            self.trainer.train_step(state_tensor, action_tensor, reward_tensor, next_state_tensor, done)
        

    def train(self, num_episodes=100):
        for e in range(num_episodes):
            # state = self.env.reset()
            state = self.env.reset()
            state = state.flatten()
            done = False
            total_score = 0
            steps = 0
            
            # print(f"Starting Episode: {e}")   

            while not done:
                action = self.get_action(state)
                # print(f"Done: {done}")  
                next_state, reward, done_list, _ = self.env.step([action])
                next_state = next_state.flatten()

                steps += 1
                if steps > 80:  #10, 50 , 100, 1000
                    done = True

                if isinstance(reward, list):
                    reward = sum(reward)  
                
                self.remember(state, action, reward, next_state, done)
                self.train_short_memory(state, action, reward, next_state, done)
                
                state = next_state
                total_score += reward
                if total_score > self.record:
                    self.record = total_score
                # print(f"Episode: {e}, Score: {total_score}, Action: {action}")   

            self.n_games += 1
            self.replay(BATCH_SIZE)   # ranmonly choose sample from saved memory
            self.scores.append(total_score)
            self.mean_scores.append(np.mean(self.scores[-100:]))
            plot(self.scores, self.mean_scores)
            
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                self.epsilon = max(self.epsilon_min, self.epsilon)
            # print(f"Epsilon after decay: {self.epsilon}")   
        
        self.model.save('dqn_model_min0.5.pth')

if __name__ == "__main__":


    custom_rew = {
        'fruit': 5.0,
        'kill': 5.0,
        'lose': -10.0,
        'win': 10.0,
        'time': 1.0,
        }

    ### if you want to add multiple snakes, need to change action into list: orlese
    ### AssertionError: File "/home/unix/marlenv/marlenv/envs/snake_env.py", line 199, in step
    ### assert len(actions) == self.num_snakes
    env = gym.make('Snake-v1', num_fruits=4, num_snakes=1, reward_dict=custom_rew, disable_env_checker=True)
    # env = gym.make('Snake-v1')

    agent = DQNAgent(env)
    # agent.train(num_episodes=1000)   # with current parameters, 500 iterations will reach 60+ score
    for episode in range(500):
        agent.train(num_episodes=1)
        plot(agent.scores, agent.mean_scores, show_final=False)

    hyperparameters = (
        f"Epsilon: {agent.epsilon_min} - {agent.epsilon_decay}\n"
        f"Gamma: {agent.gamma}\n"
        f"Learning Rate: {agent.learning_rate}\n"
        f"Batch Size: {BATCH_SIZE}\n"
        f"Max Memory: {MAX_MEMORY}\n"
        f"Reward Structure: {custom_rew}"
    )
    plot(agent.scores, agent.mean_scores, hyperparameters=hyperparameters, save_path=f'training_plot_min0.5.png')

    print(agent.record)