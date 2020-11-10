# Title: EE 321 Project 4: Multi-State Modulation
# Purpose: To complete project 4 of Dr. Mahesh Banavar's EE 321: Systems and Signal Processing project 4
# Developers: Cameron Palmer, Shawn Boyd, Siddesh Sood
# Last Modified: November 10th, 2020

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

A = 1
first_val = A
second_val = 0

# Defining symbols to use
t = [first_val, second_val, -first_val, second_val]
r = [second_val, first_val, second_val, -first_val]

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
plt.show()




