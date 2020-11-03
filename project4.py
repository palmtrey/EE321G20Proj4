# Title: EE 321 Project 4
# Purpose: To complete project 4 of Dr. Mahesh Banavar's EE 321: Systems and Signal Processing project 4
# Developers: Cameron Palmer, Shawn Boyd, Siddesh Sood
# Last Modified: November 3rd, 2020

import numpy as np
import matplotlib.pyplot as plt

# Counts the number of ones and negative ones in a vector
def countones(input):

    numones = 0
    numnegones = 0

    for i in range(0, np.size(input), 1):
        if input[i] == 1:
            numones = numones + 1
        elif input[i] == -1:
            numnegones = numnegones + 1

    percentageofones = numones / (numones + numnegones)

    print(str(percentageofones * 100) + "%")

    return numones, numnegones


generator = np.random.default_rng()

#np.random.normal(1, 1, N)
N = 1000  # Number of symbols to be transmitted
A = 1  # We can adjust this as needed. This is directly related to the transmit power.
x = A * 2 * (generator.normal(size=N) > 0) - 1 # Generates +A and -A; both equally likely

y = np.arange(0, N, 1)

print(countones(x))
#print(x)

plt.figure(0)
plt.scatter(y, x, 0.5)
#plt.show()




