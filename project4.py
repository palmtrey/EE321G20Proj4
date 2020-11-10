# Title: EE 321 Project 4
# Purpose: To complete project 4 of Dr. Mahesh Banavar's EE 321: Systems and Signal Processing project 4
# Developers: Cameron Palmer, Shawn Boyd, Siddesh Sood
# Last Modified: November 8th, 2020

import numpy as np
import matplotlib.pyplot as plt
import test_functions as tf

# Constants
N = 10000  # Number of symbols to be transmitted
SNRLOW = -50  # Lower bound to vary SNR
SNRHIGH = 50  # Upper bound to vary SNR
SNRINC = 1  # Increment for SNR
var_n = 0.01   # Noise variance

error_probability = np.zeros(len(range(SNRLOW, SNRHIGH, SNRINC)))

k = 0

for j in range(SNRLOW, SNRHIGH, SNRINC):

    # Create a random number generator object
    generator = np.random.default_rng()

    # Create some data variables
    A = np.sqrt(np.abs(j * var_n))  # We can adjust this as needed. This is directly related to the transmit power.
    snr = A ** 2 / var_n
    x = A * 2 * (generator.normal(size=N) > 0) - 1  # Generates +A and -A; both equally likely
    n = generator.normal(size=N) * np.sqrt(var_n)  # Zero-mean Gaussian noise with variance var_n
    r = x + n  # Received signal

    # Receiver calculating interpreted signal
    interpreted_data = np.zeros(N)
    i = 0
    while True:
        if r[i] > 0:
            interpreted_data[i] = A * 2 - 1
        else:
            interpreted_data[i] = -1

        i = i + 1

        if i >= len(r):
            break

    # Finding number of errors in the interpreted data
    num_errors = 0
    i = 0
    while True:
        if interpreted_data[i] != x[i]:
            num_errors = num_errors + 1

        i = i + 1

        if i >= len(r):
            break

    #print("Number of errors: " + str(num_errors))

    error_probability[k] = num_errors/N

    #print("SNR is " + str(j) + "Db. r is " + str(r) + "\n\n\n")

    k = k + 1

print(error_probability)

'''
plt.figure(4)
plt.scatter(range(SNRLOW, SNRHIGH, SNRINC), np.log10(error_probability))
plt.xlabel("SNR")
plt.ylabel("Log of Probability of error")
plt.show()




# Numeration constant for bits for graphing
Y = np.arange(0, N, 1)

# Plotting the data sent
plt.figure(0)
plt.scatter(Y, x, 0.5)
plt.title("Original Data Sent")
plt.xlabel("Bit #")
plt.ylabel("Bit value")
#plt.show()


# Plotting received data
plt.figure(1)
plt.scatter(Y, r, 0.5)
plt.title("Data Received")
plt.xlabel("Bit #")
plt.ylabel("Bit value")
#plt.show()

# Plotting interpreted data
plt.figure(2)
plt.scatter(Y, interpreted_data, 0.5)
plt.title("Data interpreted")
plt.xlabel("Bit #, SNR = " + str(snr))
plt.ylabel("Bit value")
#plt.show()
'''

