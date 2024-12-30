#!/usr/bin/env python
# coding: utf-8

# Importing Libraries

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
from mini_pong import MiniPongEnv


# Define the DQN (Deep Q-Network)

# In[3]:


# This neural network will be used to approximate the Q-values for each action.
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        # Input layer: 3 inputs (y, dx, dz), hidden layer of 32 units
        self.fc1 = nn.Linear(3, 32)
        # Second hidden layer with 32 units
        self.fc2 = nn.Linear(32, 32)
        # Output layer: Q-values for each of the 3 possible actions (left, right, do nothing)
        self.fc3 = nn.Linear(32, 3)
        
    def forward(self, x):
        # Forward pass through the network with ReLU activations
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Output Q-values for each action


# Initialize the Environment and Hyperparameters

# In[4]:


# Set up the MiniPong environment for Level 3 with a normalized state
env = MiniPongEnv(level=3, size=5, normalise=True)

# Initialize the replay memory with a maximum size of 2000 transitions
memory = deque(maxlen=2000)

# Hyperparameters
gamma = 0.99          # Discount factor to prioritize future rewards
epsilon = 1.0         # Initial exploration rate (start with maximum exploration)
epsilon_min = 0.1     # Minimum exploration rate
epsilon_decay = 0.99  # Decay rate for epsilon to gradually reduce exploration
episodes = 500        # Number of training episodes
batch_size = 64       # Number of samples in each training batch
max_steps = 200       # Maximum steps per episode to control runtime


# Define Helper Functions

# In[5]:


# Function to store experiences in replay memory
def remember(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

# Function to replay experiences and train the DQN
def replay(batch_size, dqn, optimizer, loss_fn):
    # Ensure enough samples are available in memory for replay
    if len(memory) < batch_size:
        return
    # Sample a mini-batch from memory
    minibatch = random.sample(memory, batch_size)
    
    for state, action, reward, next_state, done in minibatch:
        # Set target reward; if not done, add future reward prediction
        target = reward
        if not done:
            target += gamma * torch.max(dqn(torch.FloatTensor(next_state)))
        
        # Predicted Q-values
        q_values = dqn(torch.FloatTensor(state))
        target_f = q_values.clone()
        target_f[action] = target  # Update Q-value for the taken action
        
        # Train the network
        optimizer.zero_grad()
        loss = loss_fn(q_values, target_f)
        loss.backward()
        optimizer.step()


# Initialize DQN and Optimizer

# In[6]:


dqn = DQN()  # Create an instance of the DQN
optimizer = optim.Adam(dqn.parameters(), lr=0.001)  # Adam optimizer with learning rate 0.001
loss_fn = nn.MSELoss()  # Mean Squared Error loss function


# Training Loop

# In[7]:


# Track rewards for each episode to plot learning progress
rewards = []

for episode in range(episodes):
    # Reset the environment at the start of each episode
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0  # Step counter

    # Run the episode until completion or until max_steps
    while not done and steps < max_steps:
        # Epsilon-greedy action selection
        if random.uniform(0, 1) < epsilon:
            action = random.choice([0, 1, 2])  # Random action (exploration)
        else:
            action = torch.argmax(dqn(torch.FloatTensor(state))).item()  # Select action with max Q-value (exploitation)

        # Take action in the environment and observe the result
        next_state, reward, done, _ = env.step(action)
        remember(state, action, reward, next_state, done)  # Store transition in memory

        # Update the state and total reward
        state = next_state
        total_reward += reward
        steps += 1

        # Train the DQN using replay
        replay(batch_size, dqn, optimizer, loss_fn)

    # Decay epsilon after each episode
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    rewards.append(total_reward)  # Store total reward for this episode

    # Print progress every 10 episodes
    if episode % 10 == 0:
        print(f'Episode {episode}, Total Reward: {total_reward}')


# Plot Training Rewards

# In[8]:


plt.figure(figsize=(12, 6))
plt.plot(rewards, color='blue')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Training Reward Plot for DQN')
plt.grid(True)
plt.show()


# Evaluation: Testing the Trained Agent

# In[9]:


# Set epsilon to 0 for testing (no exploration)
epsilon = 0.0
test_episodes = 50  # Number of test episodes
test_rewards = []  # List to store test rewards

for episode in range(test_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0

    # Run the episode until completion or until max_steps
    while not done and steps < max_steps:
        # Select the best action based on learned Q-values
        action = torch.argmax(dqn(torch.FloatTensor(state))).item()
        next_state, reward, done, _ = env.step(action)
        
        # Update the state and accumulate reward
        state = next_state
        total_reward += reward
        steps += 1

    test_rewards.append(total_reward)  # Store total reward for this episode

# Calculate and print the test performance
test_average = np.mean(test_rewards)
test_std_dev = np.std(test_rewards)
print(f"Test Average Reward: {test_average}, Test Standard Deviation: {test_std_dev}")

