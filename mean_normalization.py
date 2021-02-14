import numpy as np

# Subtracts mean rating of every movie from the corresponding row
# so that each movie has a rating of zero on average
# takes care of the special case when an user hasn't rated any movies


def mean_normalization(Y, R):

    (n_m, n_u) = np.shape(Y)
    # Will contain Mean Normalized Rating
    Y_norm = np.zeros(np.shape(Y))
    # Will contain avg. rating of movies
    Y_mean = np.zeros(n_m)

    for i in range(n_m):
        indices = (R[i] == 1)  # For which rating is available
        ratings = Y[i][indices]
        Y_mean[i] = np.mean(ratings)
        Y_norm[i][indices] = Y[i][indices] - Y_mean[i]

    return (Y_norm, Y_mean)
