# Advanced-Machine-Learning-Predictive-Modeling-and-Reinforcement-Learning-with-MiniPong

This project focuses on applying advanced machine learning techniques, including Convolutional Neural Networks (CNNs), Convolutional Autoencoders, and Reinforcement Learning (RL), to tackle various predictive modeling and decision-making tasks. The primary dataset and environment used for this project are derived from MiniPong, a simplified gaming simulation for testing agent performance.

# Key Highlights

1. **Predictive Modeling with CNNs:** Two CNN architectures were designed to predict object coordinates (x, y, and z) in a 3D space. The first model predicted only the x-coordinate, while the second employed a multi-branch CNN for simultaneous prediction of x, y, and z. Both models showed significant improvement in loss and mean absolute error (MAE) during training, achieving an overall accuracy of 89.35% for the multi-output model.

2. **Image Reconstruction with Autoencoders:** A convolutional autoencoder was implemented to reconstruct images from the MiniPong dataset. The autoencoder achieved a final test loss of 0.0009, indicating minimal reconstruction error. The encoder-decoder architecture effectively compressed and restored the images, with notable success in retaining core features of the original data.

# Reinforcement Learning Agents:

1. **Tabular Q-Learning:** A Q-learning agent was implemented for MiniPong's Level 1 environment. By discretizing states and applying an epsilon-greedy policy, the agent improved its cumulative rewards over 500 episodes, achieving an average test reward of 161.2.
2. **Deep Q-Network (DQN):** For Level 3, a DQN agent was trained using a neural network with two hidden layers. Despite learning to achieve moderate rewards, the agent faced challenges in achieving consistent high performance, with an average test reward of 30.42.

# Insights and Challenges

The project highlights the versatility of machine learning techniques in solving diverse tasks. The CNNs excelled in predictive modeling, while the autoencoder demonstrated efficient dimensionality reduction and reconstruction. In RL, the tabular Q-learning approach showed promising results, but the DQN agent faced stability issues in learning due to the task's complexity and limited training episodes. These challenges suggest opportunities for further optimization, such as architectural improvements and extended training durations.
