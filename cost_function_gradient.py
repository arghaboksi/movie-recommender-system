import numpy as np


def cost_fn_gradients(parameters, Y, R, num_users, num_movies, num_features, lambd):
    """
    Calculates gradients of the collaborative filtering cost function
    w.r.t the parameters
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
    error = updated_rating_predictions - Y

    # Gradients w.r.t the X's
    X_grad = np.transpose(np.dot(np.transpose(Theta), np.transpose(error)))

    # Gradients w.r.t the Theta's
    Theta_grad = np.transpose(np.dot(np.transpose(X), error))

    # Gradients with Regularization
    X_grad += (lambd*X)
    Theta_grad += (lambd*Theta)

    # Unrolling X_grad and Theta_grad into gradients
    X_grad_vec = np.resize(X_grad, (num_movies*num_features, ))
    Theta_grad_vec = np.resize(Theta_grad, (num_users*num_features, ))

    # Contains all the gradients
    gradients = np.concatenate((X_grad_vec, Theta_grad_vec))

    return gradients
