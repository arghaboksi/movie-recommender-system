import numpy as np

# Calculates collaborative filtering cost function value and gradients for a
# particular setting of parameters(movie features and user preferences)


def cost_gradient(Theta, X, Y, R, num_users, num_movies, lambd):

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

    # Gradients w.r.t the X's
    X_grad = np.transpose(np.dot(np.transpose(Theta), np.transpose(error)))

    # Gradients w.r.t the Theta's
    Theta_grad = np.transpose(np.dot(np.transpose(X), error))

    # Regularized Cost Function Value
    J += ((np.sum(Theta**2) + np.sum(X**2))*lambd)/2

    # Gradients with Regularization
    X_grad += (lambd*X)

    Theta_grad += (lambd*Theta)
