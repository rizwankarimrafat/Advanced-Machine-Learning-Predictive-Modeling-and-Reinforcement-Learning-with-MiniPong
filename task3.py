#!/usr/bin/env python
# coding: utf-8

# Create a Simple Python Program to Play MiniPong Without RL

# In[26]:


from mini_pong import MiniPongEnv

#initialize environment
env = MiniPongEnv(level=1, size=5, normalise=True)

#reset environment to start
state = env.reset()

done = False
total_reward = 0

while not done:
    dz = state  #state contains dz in level 1
    
    #basic policy: move left, right, or do nothing based on dz
    if dz < 0:
        action = 1  # move left
    elif dz > 0:
        action = 2  # move right
    else:
        action = 0  # do nothing

    #take action and get next state and reward
    state, reward, done, _ = env.step(action)
    total_reward += reward

print("Total reward:", total_reward)


# In[3]:


import numpy as np
import random
import matplotlib.pyplot as plt
from mini_pong import MiniPongEnv
import time

#initialize the MiniPong environment
env = MiniPongEnv(level=1, size=5, normalise=True)

#discretization parameters
num_states = 21  # Discretizing the dz space into 21 states (-1 to 1, or -13 to 13 if normalise=False)
state_bins = np.linspace(-1, 1, num_states)

#Q-learning parameters
action_size = env.action_space.n  # 3 actions: [move left, move right, do nothing]
q_table = np.zeros((num_states, action_size))  # Q-table initialization
learning_rate = 0.1  # Alpha
gamma = 0.95  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_min = 0.1  # Minimum epsilon
epsilon_decay = 0.995  # Epsilon decay rate
episodes = 500  # Number of episodes
max_steps = 200  # Max steps per episode

#function to discretize the continuous state space into discrete bins
def discretize_state(dz):
    return np.digitize(dz, state_bins) - 1  # Return index of the bin for dz

#epsilon-greedy action selection
def choose_action(state, epsilon):
    if np.random.rand() <= epsilon:
        return random.randrange(action_size)  # Random action (exploration)
    return np.argmax(q_table[state])  # Action with highest Q-value (exploitation)

#Q-learning algorithm with TD-learning update
def q_learning_update(state, action, reward, next_state, done):
    best_next_action = np.argmax(q_table[next_state])
    td_target = reward + gamma * q_table[next_state, best_next_action] * (1 - done)
    td_error = td_target - q_table[state, action]
    q_table[state, action] += learning_rate * td_error  # Q-value update
    
#start the timer
start_time = time.time()

#training loop
rewards_per_episode = []

for episode in range(episodes):
    state = env.reset()
    state = discretize_state(state)  # Discretize the initial state
    total_reward = 0
    
    for step in range(max_steps):
        action = choose_action(state, epsilon)
        next_state, reward, done, _ = env.step(action)
        next_state = discretize_state(next_state)  # Discretize next state
        
        # Q-learning update (TD learning)
        q_learning_update(state, action, reward, next_state, done)
        
        state = next_state
        total_reward += reward
        
        if done:
            break
    
    #decay epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    
    #store total reward for this episode
    rewards_per_episode.append(total_reward)
    print(f"Episode {episode + 1}/{episodes} | Total Reward: {total_reward} | Epsilon: {epsilon:.2f}")

#plot training rewards
plt.figure(figsize=(12, 6))
plt.plot(rewards_per_episode, color='blue')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Training Rewards per Episode')
plt.show()

#calculate total run time
total_run_time = time.time() - start_time
print(f"Total model run time: {total_run_time:.2f} seconds")

#close environment
env.close()


# In[28]:


#set epsilon to 0 for testing (no exploration)
epsilon = 0.0

#initialize variables to store test rewards
test_rewards = []
test_episodes = 50

#testing loop
for episode in range(test_episodes):
    state = env.reset()
    state = discretize_state(state)  # Discretize state
    total_reward = 0
    
    done = False
    while not done:
        action = choose_action(state, epsilon)  # Always exploit the learned Q-values
        next_state, reward, done, _ = env.step(action)
        next_state = discretize_state(next_state)
        
        state = next_state
        total_reward += reward
        
        if done:
            test_rewards.append(total_reward)
            break

#calculate the test average and standard deviation
test_average = np.mean(test_rewards)
test_std_deviation = np.std(test_rewards)

print(f"Test Average Reward: {test_average}")
print(f"Test Standard Deviation: {test_std_deviation}")

