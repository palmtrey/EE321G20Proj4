# Title: EE 321 Project 4: Multi-State Modulation
# Purpose: To complete project 4 of Dr. Mahesh Banavar's EE 321: Systems and Signal Processing project 4
# Developers: Cameron Palmer, Shawn Boyd, Siddesh Sood
# Last Modified: November 10th, 2020

import numpy as np
import matplotlib.pyplot as plt
import random as ran

# Constants
N = 1000  # Number of symbols to be transmitted
SNRLOW = -50  # Lower bound to vary SNR
SNRHIGH = 50  # Upper bound to vary SNR
SNRINC = 1  # Increment for SNR
var_n = 0.01   # Noise variance

error_probability = np.zeros(len(range(SNRLOW, SNRHIGH, SNRINC)))

A = 1
first_val = A
second_val = 0

# Defining symbols to use
t = [first_val, second_val, -first_val, second_val]  # t is x-axis
r = [second_val, first_val, second_val, -first_val]  # r is y-axis

# Deriving and sketching decision boundaries

# First, calculate (dashed) lines connecting the points on the outside (in a "diamond" shape)

# 1: Line from (-A, 0) to (0, +A)
m1 = (first_val - second_val)/(second_val + first_val)
b1 = second_val - m1*(-first_val)

x1 = np.arange(-first_val, second_val + 0.1, 0.1)
y1 = m1*x1 + b1

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

#4: Line from (0, -A) to (-A, 0)
m4 = (second_val + first_val)/(-first_val - second_val)
b4 = -first_val - m4*second_val

x4 = np.arange(-first_val, 0 + 0.1, 0.1)
y4 = m4*x4 + b4


# Second, calculate the perpendicular lines bisecting the dashed lines

#1: Line bisecting dashed line #1, (-A, 0) to (0, +A)
mp1 = -m1
t1 = (-first_val + second_val)/2
r1 = (second_val + first_val)/2
bp1 = r1 - mp1*t1

xp1 = np.arange(-first_val, first_val + 0.1, 0.1)
yp1 = mp1*xp1 + bp1

#2: Line bisecting dashed line #2, (0, +A) to (+A, 0)
mp2 = -m2
t2 = (second_val + first_val)/2
r2 = (first_val + second_val)/2
bp2 = r2 - mp2*t2

xp2 = np.arange(-first_val, first_val + 0.1, 0.1)
yp2 = mp2*xp2 + bp2

plt.figure(0)
plt.scatter(t, r)
plt.plot(x1, y1, "--")
plt.plot(x2, y2, "--")
plt.plot(x3, y3, "--")
plt.plot(x4, y4, "--")

plt.plot(xp1, yp1)
plt.plot(xp2, yp2)
#plt.show()


# Actually start calculations now

generator = np.random.default_rng()  # Create a random number generator object

# Create some data variables
# snr =

# Generates a number between 0 and 3, each corresponding to one point
x = np.zeros(N)
for i in np.arange(0, N):
    x[i] = ran.randint(0, 3)

print(x)

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

plt.figure(2)
plt.scatter(gen_data_x, gen_data_y, 0.5)
#plt.show()


# Noise this shit up
generator = np.random.default_rng()

n_x = generator.normal(size=N) * np.sqrt(var_n)  # Zero-mean Gaussian noise with variance var_n
n_y = generator.normal(size=N) * np.sqrt(var_n)

# Received signal
r_x = gen_data_x + n_x
r_y = gen_data_y + n_y

plt.figure(3)
plt.scatter(r_x, r_y, 0.5)
plt.show()

'''
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
'''

