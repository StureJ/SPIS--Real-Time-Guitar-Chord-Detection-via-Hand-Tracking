import numpy as np

def window_to_features(window):
    # Converts a time series of feature data into a single summary vector
    # by computing the average and variation (standard deviation) of each feature over time.

    window = np.array(window)

    #mean over time (gesture shape)
    mean_features = np.mean(window, axis=0)

    #variability over time (stability / motion)
    std_features = np.std(window, axis=0)

    #combine into one vector
    features = np.concatenate([mean_features, std_features])

    return features