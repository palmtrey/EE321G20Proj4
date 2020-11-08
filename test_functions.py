
import numpy as np

def countones(input):

    numones = 0
    numnegones = 0

    for i in range(0, np.size(input), 1):
        if input[i] > 0:
            numones = numones + 1
        elif input[i] == -1:
            numnegones = numnegones + 1

    percentageofones = numones / (numones + numnegones)

    print("Percentage of positive #s in the array: " + str(percentageofones * 100) + "%\n")

    return numones, numnegones
