
import numpy as np

def countpositives(input):

    num_positives = 0
    num_negatives = 0

    for i in range(0, np.size(input), 1):
        if input[i] > 0:
            num_positives = num_positives + 1
        elif input[i] < 0:
            num_negatives = num_negatives + 1

    print("Percentage of positive #s in the array: " + str(num_positives / (num_positives + num_negatives) * 100) + "%\n")
