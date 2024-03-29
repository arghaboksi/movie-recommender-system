# movie-recommendation-system
A movie recommendation system based on the collaborative filtering learning algorithm.

## Dataset :
MovieLens 100k Dataset from GroupLens Research.

## Files : 
1. movies.mat - contains all the movie ratings provided by the users.
2. movie_ids.txt - contains movie names along with their corresponding ids. 
3. main_script.py - runs the main collaborative filtering algorithm on the dataset by calling several helper functions.
4. cost_function.py - calculates the collaborative filtering cost function value for a specific setting of parameters.
5. cost_function_gradient.py - calculates gradients of the collaborative filtering cost function w.r.t the parameters.
6. load_movie.py - processes movie_ids.txt and returns a dictionary containing ids and movie names as key value pairs.
7. mean_normalization.py - performs mean normalization on the dataset before running the optimization algorithm.

## Parameters of the model : 
The task is to learn two matrices X and Theta, so that M = X(Theta)' approximates Y(a matrix containing all the actual user ratings). 

1. X - a matrix containing movie features, The i-th row of X corresponds to the feature vector x(i) for the i-th movie.
2. Theta - a matrix containing user preferences, the j-th row of Theta corresponds to the parameter vector Theta(j), for the j-th user.

## Hyperparameters : 
1. num_features - no. of latent features to be learned for each of the movies.
2. lambda - regularization parameter.

## Optimization Algorithm Used :
Conjugate Gradient

## Packages Used : 
1. NumPy : For Matrix Manipulations.
2. SciPy : For Optimization.

## Performance : 
Hyperparameters values used - 
num_features = 10, lambda = 0.2

"New User Ratings" are the ratings given by a new user for some of the movies, here I have rated "Drama/Romance/Comedy" movies highly but did not really like few of the "Crime" movies.

![](Screenshots/New_User_Ratings.PNG)

"Top 10 Recommendations for you" are the 10 movies that the algorithm is recommending for me to watch next. 

![](Screenshots/Recommendations.PNG)



Here are the genres of the recommended movies -

1) Comedy/Drama
2) Comedy/Drama
3) Comedy/Drama
4) Romance/Drama
5) Adventure/Comedy/Drama
6) Drama/Comedy
7) Comedy/War
8) Mystery/Thriller
9) Satire/Comedy/Drama
10) Mystery/Thriller
