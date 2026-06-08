
import torch
import torch.nn as nn
import random
from collections import deque
import torch.optim as optim
import numpy as np
import os

class DQN_Agent(nn.Module):
    def __init__(self, state=10, action=4):
        super(DQN_Agent, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action)
        )
        self.target_network = nn.Sequential(
            nn.Linear(state, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action)
        )
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval()
        
        self.epsilon = 1.0
        self.epsilon_reduce = 0.9995
        self.epsilon_min = 0.05
        self.memory = deque(maxlen=50000)
        self.gamma = 0.99
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.0005)
        self.criterion = nn.MSELoss()

    def forward(self, state):
        return self.network(state)
    
    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 3) 
        else:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.network(state_t)
            return torch.argmax(q_values).item()
    
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < 128:
            return
            
        batch = random.sample(self.memory, 128)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.LongTensor(actions).unsqueeze(1)
        rewards_t = torch.FloatTensor(rewards)
        next_states_t = torch.FloatTensor(np.array(next_states))
        dones_t = torch.FloatTensor(dones)
        
        current_q = self.network(states_t).gather(1, actions_t).squeeze(1)
        
        with torch.no_grad():
            max_next_q = self.target_network(next_states_t).max(1)[0]
            target_q = rewards_t + (1 - dones_t) * self.gamma * max_next_q
            
        loss = self.criterion(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def update_target_network(self):
        self.target_network.load_state_dict(self.network.state_dict())

    def save(self, filename):
        torch.save(self.network.state_dict(), filename)

    def load(self, filename):
        if os.path.exists(filename):
            self.network.load_state_dict(torch.load(filename))
            self.target_network.load_state_dict(self.network.state_dict())
            self.network.eval()
            return True
        return False