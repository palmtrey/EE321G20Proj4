# Title: EE 321 Project 4: Multi-State Modulation
# Purpose: To complete project 4 of Dr. Mahesh Banavar's EE 321: Systems and Signal Processing project 4
# Developers: Cameron Palmer, Shawn Boyd, Siddesh Sood
# Last Modified: November 10th, 2020

import numpy as np
import matplotlib.pyplot as plt
import random as ran

# Constants
N = 100000  # Number of symbols to be transmitted
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

    A = np.sqrt(np.abs(j * var_n))  # THis value is directly related to the power of the system, and will be modified to find a graph of error vs SNR
    first_val = A  # The first value of the symbols (in a scheme (A, 0), (0, A), (-A, 0), (0, -A), this will be A)
    second_val = 0  # The second value of the symbols (in a scheme (A, 0), (0, A), (-A, 0), (0, -A), this will be 0)

    # Defining symbols to use
    t = [first_val, second_val, -first_val, second_val]  # t is x-axis
    r = [second_val, first_val, second_val, -first_val]  # r is y-axis


    # Deriving and sketching decision boundaries

    # First, calculate (dashed) lines connecting the points on the outside (in a "diamond" shape)

    # 1: Line from (-A, 0) to (0, +A)
    m1 = (first_val - second_val)/(second_val + first_val)  # Calculate m (slope)
    b1 = second_val - m1*(-first_val)  # Calculate b (intercept)

    x1 = np.arange(-first_val, second_val + 0.1, 0.1)
    y1 = m1*x1 + b1  # Calculate a discrete set of points along the line to actually plot it

    # 2: Line from (0, +A) to (+A, 0)
    m2 = (second_val - first_val)/(first_val - second_val)
    b2 = first_val - m2*second_val

    x2 = np.arange(0, first_val + 0.1, 0.1)
    y2 = m2*x2 + b2

    # 3: Line from (+A, 0) to (0, -A)
    m3 = (-first_val - second_val)/(second_val - first_val)
    b3 = second_val - m3*first_val

    x3 = np.arange(0, first_val + 0.1, 0.1)
    y3 = m3*x3 + b3

    # 4: Line from (0, -A) to (-A, 0)
    m4 = (second_val + first_val)/(-first_val - second_val)
    b4 = -first_val - m4*second_val

    x4 = np.arange(-first_val, 0 + 0.1, 0.1)
    y4 = m4*x4 + b4


    # Second, calculate the perpendicular lines bisecting the dashed lines

    PERP_RES = 0.01  # The resolution of the perpendicular lines. Crank this way down when doing final simulations

    # 1: Line bisecting dashed line #1, (-A, 0) to (0, +A)
    mp1 = -m1
    t1 = (-first_val + second_val)/2
    r1 = (second_val + first_val)/2
    bp1 = r1 - mp1*t1

    xp1 = np.arange(-first_val - var_n*10, first_val + PERP_RES + var_n*10, PERP_RES)
    yp1 = mp1*xp1 + bp1

    # 2: Line bisecting dashed line #2, (0, +A) to (+A, 0)
    mp2 = -m2
    t2 = (second_val + first_val)/2
    r2 = (first_val + second_val)/2
    bp2 = r2 - mp2*t2

    xp2 = np.arange(-first_val - var_n*10, first_val + PERP_RES + var_n*10, PERP_RES)
    yp2 = mp2*xp2 + bp2

    #plt.figure(0)
    #plt.scatter(t, r)
    #plt.plot(x1, y1, "--")
    #plt.plot(x2, y2, "--")
    #plt.plot(x3, y3, "--")
    #plt.plot(x4, y4, "--")

    #plt.plot(xp1, yp1)
    #plt.plot(xp2, yp2)
    #plt.show()


    # Actually start calculations now

    generator = np.random.default_rng()  # Create a random number generator object

    # Create some data variables
    # snr =

    # Generates a number between 0 and 3, each corresponding to one point
    x = np.zeros(N)
    for i in np.arange(0, N):
        x[i] = ran.randint(0, 3)

    # Plotting the generated data
    gen_data_x = np.zeros(N)
    gen_data_y = np.zeros(N)

    # Assign each value generated in x to a point
    for i in np.arange(0, N):
        if x[i] == 0.:
            gen_data_x[i] = first_val
            gen_data_y[i] = second_val
        elif x[i] == 1.:
            gen_data_x[i] = second_val
            gen_data_y[i] = first_val
        elif x[i] == 2.:
            gen_data_x[i] = -first_val
            gen_data_y[i] = second_val
        elif x[i] == 3.:
            gen_data_x[i] = second_val
            gen_data_y[i] = -first_val

    #plt.figure(2)
    #plt.scatter(gen_data_x, gen_data_y, 0.5)
    #plt.show()


    # Noise this shit up
    generator = np.random.default_rng()

    n_x = generator.normal(size=N) * np.sqrt(var_n)  # Zero-mean Gaussian noise with variance var_n
    n_y = generator.normal(size=N) * np.sqrt(var_n)

    # Received signal
    r_x = gen_data_x + n_x
    r_y = gen_data_y + n_y

    #plt.figure(3)
    #plt.scatter(r_x, r_y, 0.5)
    #plt.show()


    # Receiver de-noising signal to the best of its ability
    rec_data_x = np.zeros(N)
    rec_data_y = np.zeros(N)

    # Decision algorithm
    ROUND_VAL = 2
    #print(xp1)

    xp1_float = np.zeros(len(xp1))  # Initialize arrays to store xp1, xp2, yp1, and yp2, as rounded floats
    xp2_float = np.zeros(len(xp2))
    yp1_float = np.zeros(len(yp1))
    yp2_float = np.zeros(len(yp2))

    # Round xp1, xp2, yp1, yp2, and store the rounded values in xp1_float, etc
    for i in np.arange(0, len(xp1)):
        xp1_float[i] = np.round(float(xp1[i]), ROUND_VAL)
        xp2_float[i] = np.round(float(xp2[i]), ROUND_VAL)

    for i in np.arange(0, len(yp1)):
        yp1_float[i] = np.round(float(yp1[i]), ROUND_VAL)
        yp2_float[i] = np.round(float(yp2[i]), ROUND_VAL)


    # Determine which area each point falls into based on the algorithm
    for i in np.arange(0, N):
        val_x = np.round(r_x[i], ROUND_VAL)
        val_y = np.round(r_y[i], ROUND_VAL)

        # Calculate indices of xp1_float...yp2_float where the values of val_x and val_y respectively are located
        index_xp1 = (((xp1_float == np.round(val_x, ROUND_VAL)).nonzero())[0])  # Find the index where the point's x-value can be found in the perpendicular line p1
        index_xp2 = (((xp2_float == np.round(val_x, ROUND_VAL)).nonzero())[0])  # Find the index where the point's x-value can be found in the perpendicular line p2
        index_yp1 = (((yp1_float == np.round(val_y, ROUND_VAL)).nonzero())[0])  # Find the index where the point's y-value can be found in the perpendicular line p1
        index_yp2 = (((yp2_float == np.round(val_y, ROUND_VAL)).nonzero())[0])  # Find the index where the point's y-value can be found in the perpendicular line p2

        # This function determines if the indices calculated above exist and are valid to index.
        # If this function returns False, the indices do not exist (the point is outside the
        # range that the perpendicular boundary lines have been calculated)
        # This should never happen, however, in testing, this was often a problem, and for safety,
        # this function will remain implemented.
        def indicesExist():
            if index_xp1.size == 1 and index_xp2.size == 1 and index_yp1.size == 1 and index_yp2.size == 1:
                return True
            else:
                return False

        if indicesExist() and (val_x > xp1_float[index_yp1[0]]) and (val_x < xp2_float[index_yp2[0]]) and (val_y > yp1_float[index_xp1[0]]) and (val_y > yp2_float[index_xp2[0]]):
            rec_data_x[i] = second_val
            rec_data_y[i] = first_val
        elif indicesExist() and (val_x > xp2_float[index_yp2[0]]) and (val_x > xp1_float[index_yp1[0]]) and (val_y > yp1_float[index_xp1[0]]) and (val_y < yp2_float[index_xp2[0]]):
            rec_data_x[i] = first_val
            rec_data_y[i] = second_val
        elif indicesExist() and (val_x > xp2_float[index_yp2[0]]) and val_x < xp1_float[index_yp1[0]] and (val_y < yp2_float[index_xp2[0]]) and (val_y < yp2_float[index_xp2[0]]):
            rec_data_x[i] = second_val
            rec_data_y[i] = -first_val
        elif indicesExist() and (val_x < xp1_float[index_yp1[0]]) and (val_x < xp2_float[index_yp2[0]]) and (val_y > yp2_float[index_xp2[0]]) and (val_y < yp1_float[index_xp1[0]]):
            rec_data_x[i] = -first_val
            rec_data_y[i] = second_val

    # Graph the received, de-noised data
    #plt.figure(4)
    #plt.scatter(rec_data_x, rec_data_y)
    #plt.show()


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
#plt.plot(range(SNRLOW, SNRHIGH, SNRINC), error_probability)
plt.semilogy(np.arange(SNRLOW, SNRHIGH, SNRINC), error_probability)
plt.title("Trial 2")
plt.xlabel("SNR")
plt.ylabel("Log of Probability of error")
plt.show()