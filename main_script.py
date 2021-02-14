from scipy.io import loadmat
import numpy as np
from scipy.optimize import minimize
from load_movie import load_movie
from mean_normalization import mean_normalization
from cost_function import cost_fn_value
from cost_function_gradient import cost_fn_gradients

# Loading the movie rating dataset
annots = loadmat('movies.mat')  # returns a python dictionary
print(annots.keys())
print(type(annots['Y']), type(annots['R']))  # both Y and R are numpy arrays
Y = annots['Y']  # contains the ratings provided by users
# R(i, j) = 1 if user j gave a rating to movie i, and R(i, j) = 0 otherwise
R = annots['R']
print(np.shape(Y))  # num_movies x num_users
print(np.shape(R))
(num_movies, num_users) = np.shape(Y)

# Average rating of a particular movie
movie_id = 1
movie_ratings = Y[movie_id-1]
# no. of users who have rated the movie
no_users_rated = np.count_nonzero(R[movie_id-1] == 1)
print("Average Rating of movie", movie_id, ": ",
      np.sum(movie_ratings)/no_users_rated)


# Testing Cost Function and its gradient evaluation
# annots = loadmat('test_params.mat')
# print(annots.keys())
# X = annots['X']
# Theta = annots['Theta']
# num_users = 4
# num_movies = 5
# num_features = 3
# X = X[:num_movies, :num_features]
# print(X)
# Theta = Theta[:num_users, : num_features]
# print(Theta)
# Y = Y[:num_movies, :num_users]
# R = R[:num_movies, :num_users]
# X_vec = np.resize(X, (num_movies*num_features, ))
# Theta_vec = np.resize(Theta, (num_users*num_features, ))
# parameters = np.concatenate((X_vec, Theta_vec))
# J = cost_fn_value(parameters, Y, R, num_users, num_movies, num_features, 0)
# print(J)  # should be around 22.224
# grad = cost_fn_gradients(parameters, Y, R, num_users,
#                          num_movies, num_features, 0)
# print(grad)
# J = cost_fn_value(parameters, Y, R, num_users, num_movies, num_features, 1.5)
# print(J)  # should be around 31.34
# grad = cost_fn_gradients(parameters, Y, R, num_users,
#                          num_movies, num_features, 1.5)
# print(grad)

# # Testing Mean Normalization
# (Y_norm, Y_mean) = mean_normalization(Y, R)
# print(Y_norm)
# print(Y_mean)

# Enter Your own recommendations
movie_list = load_movie()

# Initialize new user ratings to all zeros
my_ratings = np.zeros(num_movies+1)

# New User Ratings
my_ratings[216] = 4  # When Harry Met Sally
my_ratings[313] = 5  # Titanic
my_ratings[69] = 5  # Forrest Gump
my_ratings[955] = 4  # Before Sunrise
my_ratings[173] = 5  # The Princess Bride
my_ratings[739] = 5  # Pretty Woman
my_ratings[88] = 5  # Sleepless in Seattle
my_ratings[70] = 4  # Four Weddings and a Funeral

print("New User Ratings : ")

for i in range(1, num_movies+1):
    if(my_ratings[i] != 0):
        print("Movie Name : ", movie_list[str(i)],
              ", Rating Provided : ", my_ratings[i])

# Adding my ratings to the matrices Y and R
Y = np.insert(Y, num_users, my_ratings[1:], axis=1)
user_r = np.where(my_ratings > 0, 1, my_ratings)
R = np.insert(R, num_users, user_r[1:], axis=1)
print(np.shape(Y))
print(np.shape(R))
num_users += 1

# Mean Normalization
(Y_norm, Y_mean) = mean_normalization(Y, R)

# No. of Movie Features to be learned
num_features = 10

# Random Initialization of X and Theta
X = np.random.randn(num_movies, num_features)
Theta = np.random.randn(num_users, num_features)

print(np.shape(X))
print(np.shape(Theta))

# Unrolling all the parameters into a vector
X_vec = np.resize(X, (num_movies*num_features, ))
Theta_vec = np.resize(Theta, (num_users*num_features, ))
initial_parameters = np.concatenate((X_vec, Theta_vec))

# Regularization Parameter
lambd = 10

# Optimization
res = minimize(cost_fn_value, initial_parameters, args=(Y_norm, R, num_users, num_movies, num_features, lambd), method='BFGS',
               jac=cost_fn_gradients, options={'disp': True})

print(np.shape(res.x), 16820+9440)

# Unfolding the learned parameters into X and Theta
learned_parameters = res.x
X = np.reshape(
    learned_parameters[:num_movies*num_features], (num_movies, num_features))
Theta = np.reshape(
    learned_parameters[num_movies*num_features:], (num_users, num_features))

print("Recommender System learning completed!")

# Recommendations for new user

# Predictions Matrix
predictions = np.dot(X, np.transpose(Theta))
# My Predictions
my_predictions = predictions[:, num_users-1] + Y_mean
# Top 10 Recommendations indices
rec_mov_indices = np.argsort(my_predictions)[-1:-11:-1]

print('Top 10 Recommendations for You : ')

for i in rec_mov_indices:
      print(movie_list[str(i+1)])


