import numpy as np


def cost_fn_value(parameters, Y, R, num_users, num_movies, num_features, lambd):
    """
    Calculates the collaborative filtering cost function value
    for a specific setting of parameters
    """

    # Retreiving X and Theta from parameters
    X = np.resize(parameters[:num_movies*num_features],
                  (num_movies, num_features))
    Theta = np.resize(
        parameters[num_movies*num_features:], (num_users, num_features))

    # Predicted ratings of all the movies by all the users
    rating_predictions = np.dot(X, np.transpose(Theta))

    # Updating rating_predictions to only contain values for which actual rating is available
    updated_rating_predictions = np.multiply(rating_predictions, R)

    # Error associated with all the predictions
    error = Y - updated_rating_predictions

    # Squared Error
    squared_error = error**2

    # Sum of squared error for all the predictions
    sum_sq_error = np.sum(squared_error)

    # Cost function value without regularization
    J = sum_sq_error/2

    # Regularized Cost Function Value
    J += ((np.sum(Theta**2) + np.sum(X**2))*lambd)/2

    return J
