#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 19:50:25 2024

@author: oliver
"""

import numpy as np
from mini_pong import MiniPongEnv  
from matplotlib import pyplot as plt

size = 5
level = 0  # Set level to 0 to get pixel representation

# Create an instance of the MiniPong environment
pong = MiniPongEnv(level=level, size=size)

# Create the dataset dimensions based on the environment properties
rows = pong.xymax * pong.xymax * size  # Total number of samples
columns = pong.size * pong.size        # Flattened size of the pixel array

# Prepare arrays to store pixel data and labels
allpix = np.zeros((rows, columns), dtype=int)
alllabels = np.zeros((rows, 3), dtype=int)
j = 0

# Generate the dataset with all possible (x, y, p) combinations
for x in range(1, pong.xymax + 1):
    for y in range(1, pong.xymax + 1):
        for p in range(pong.zmax + 1):
            # Manually set the state of the environment
            pong.s1 = (x, y, p, 0, 0)
            # Get the pixel representation from the environment
            pixels = pong._to_pixels(pong.s1, binary=False).flatten(order='C')
            allpix[j, :] = pixels
            alllabels[j, :] = [x, y, p]
            j += 1

# Save the dataset in a random order
np.random.seed(10)
neworder = np.random.choice(rows, size=rows, replace=False)
allpix = allpix[neworder, :]
alllabels = alllabels[neworder, :]
np.savetxt('allpix.csv', allpix, delimiter=',', fmt='%d')
np.savetxt('alllabels.csv', alllabels, delimiter=',', fmt='%d')

# Save a few example images from the dataset
for i in range(4):
    fig, ax = plt.subplots()
    ax.axis("off")
    plt.imshow(allpix[10 + i, :].reshape(15, 15), origin='lower', cmap=plt.cm.coolwarm)
    plt.savefig(f'proto{i+1}.png', dpi=300)
    plt.show(block=False)
    plt.pause(2)  
    plt.close(fig)
    
# Number of training and test samples
ntrain = round(0.8 * rows)
ntest = rows - ntrain

# Split the dataset into training and testing sets
trainingpix = allpix[:ntrain, :]
traininglabels = alllabels[:ntrain, :]
testingpix = allpix[ntrain:, :]
testinglabels = alllabels[ntrain:, :]

# Save training and testing data
np.savetxt('trainingpix.csv', trainingpix, delimiter=',', fmt='%d')
np.savetxt('traininglabels.csv', traininglabels, delimiter=',', fmt='%d')
np.savetxt('testingpix.csv', testingpix, delimiter=',', fmt='%d')
np.savetxt('testinglabels.csv', testinglabels, delimiter=',', fmt='%d')
