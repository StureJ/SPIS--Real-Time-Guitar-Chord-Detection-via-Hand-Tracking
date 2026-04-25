import numpy as np

def compute_distances(fingertips):

    #Calculates the distances between every pair of fingertips in 3D space (x,y,z).
    fingers = list(fingertips.values())

    distances = []

    for i in range(len(fingers)):
        for j in range(i + 1, len(fingers)):
            a = np.array(fingers[i])
            b = np.array(fingers[j])

            d = np.linalg.norm(a - b)
            distances.append(d)

    return distances