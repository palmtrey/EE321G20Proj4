# Title: EE 321 Project 4
# Purpose: To complete project 4 of Dr. Mahesh Banavar's EE 321: Systems and Signal Processing project 4
# Developers: Cameron Palmer, Shawn Boyd, Siddesh Sood
# Last Modified: November 3rd, 2020

import numpy as np
import matplotlib.pyplot as plt
import test_functions as tf

# Counts the number of ones and negative ones in a vector


generator = np.random.default_rng()

N = 1000  # Number of symbols to be transmitted
A = 1  # We can adjust this as needed. This is directly related to the transmit power.
x = A * 2 * (generator.normal(size=N) > 0) - 1 # Generates +A and -A; both equally likely

y = np.arange(0, N, 1)

# Show the distribution of positive numbers and negative ones among our array
tf.countones(x)

plt.figure(0)
plt.scatter(y, x, 0.5)
plt.title("Original Data Sent")
plt.xlabel("Bit #")
plt.ylabel("Bit value")
plt.show()


var_n = 0.01  # Noise variance
n = generator.normal(size=N) * np.sqrt(var_n)  # Zero-mean Gaussian noise with variance var_n
r = x + n  # Received signal

print(r)



