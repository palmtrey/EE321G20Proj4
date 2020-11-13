# Title: EE 321 Project 4 - Trial 3
# Purpose: To complete project 4 of Dr. Mahesh Banavar's EE 321: Systems and Signal Processing project 4
# Developers: Cameron Palmer, Shawn Boyd, Siddesh Sood
# Last Modified: November 13th, 2020

import numpy as np
import matplotlib.pyplot as plt
import random as ran

# Constants
N = 1000000  # Number of symbols to be transmitted
SNRLOW = -20  # Lower bound to vary SNR
SNRHIGH = 20  # Upper bound to vary SNR
SNRINC = 1  # Increment for SNR
var_n = 0.1   # Noise variance
error_probability = np.zeros(len(np.arange(SNRLOW, SNRHIGH, SNRINC)))

k = 0

for j in np.arange(SNRLOW, SNRHIGH, SNRINC):

    # Skip j = 0
    if j == 0:
        j = j + SNRINC

    print("Cycle: SNR = " + str(np.round(j, 2)))

    A = np.sqrt(np.abs(j * var_n))  # This value is directly related to the power of the system, and will be modified to find a graph of error vs SNR
    val = A/(2**(1/2))  # The first and second value of the symbols

    # Defining symbols to use
    t = [val, -val, -val, val]  # t is x-axis
    r = [val, val, -val, -val]  # r is y-axis

    # Decision boundaries don't need to be calculated for this, as it can be clearly seen from the graph
    # of the four symbols that the decision boundaries are simply the x and y axes, making the decision
    # algorithm fairly simple (the four traditional quadrants are used)

    # Generates a number between 0 and 3, each corresponding to one point
    x = np.zeros(N)
    for i in np.arange(0, N):
        x[i] = ran.randint(0, 3)

    # Plotting the generated data
    gen_data_x = np.zeros(N)
    gen_data_y = np.zeros(N)

    # Assign each value generated in x[] to a symbol
    for i in np.arange(0, N):
        if x[i] == 0.:
            gen_data_x[i] = val  # Symbol in quadrant 1
            gen_data_y[i] = val
        elif x[i] == 1.:
            gen_data_x[i] = -val  # Symbol in quadrant 2
            gen_data_y[i] = val
        elif x[i] == 2.:
            gen_data_x[i] = -val  # Symbol in quadrant 3
            gen_data_y[i] = -val
        elif x[i] == 3.:
            gen_data_x[i] = val  # Symbol in quadrant 4
            gen_data_y[i] = -val

    # Add some noise
    generator = np.random.default_rng()

    n_x = generator.normal(size=N) * np.sqrt(var_n)  # Zero-mean Gaussian noise with variance var_n
    n_y = generator.normal(size=N) * np.sqrt(var_n)

    # Received signal
    r_x = gen_data_x + n_x
    r_y = gen_data_y + n_y

    # Receiver de-noising signal to the best of its ability
    rec_data_x = np.zeros(N)
    rec_data_y = np.zeros(N)

    # Decision algorithm
    ROUND_VAL = 2

    # Determine which area each point falls into based on the algorithm
    for i in np.arange(0, N):
        val_x = np.round(r_x[i], ROUND_VAL)
        val_y = np.round(r_y[i], ROUND_VAL)

        if val_x > 0 and val_y > 0:
            rec_data_x[i] = val  # Quadrant 1
            rec_data_y[i] = val
        elif val_x < 0 and val_y > 0:
            rec_data_x[i] = -val  # Quadrant 2
            rec_data_y[i] = val
        elif val_x < 0 and val_y < 0:
            rec_data_x[i] = -val  # Quadrant 3
            rec_data_y[i] = -val
        elif val_x > 0 and val_y < 0:
            rec_data_x[i] = val  # Quadrant 4
            rec_data_y[i] = -val

    # Finding number of errors in the interpreted data
    num_errors = 0
    i = 0  # Iterator

    # If the x and y coordinates of the received data do not exactly match the
    # original generated data (before noise), increase the number of errors
    # by 1.
    while True:
        if rec_data_x[i] != gen_data_x[i] and rec_data_y[i] != gen_data_y[i]:
            num_errors = num_errors + 1

        i = i + 1

        if i >= len(rec_data_x):
            break

    error_probability[k] = num_errors/N

    k = k + 1

plt.figure(6)
plt.semilogy(np.arange(SNRLOW, SNRHIGH, SNRINC), error_probability)
plt.title("Trial 3")
plt.xlabel("SNR")
plt.ylabel("Log of Probability of error")
plt.show()
