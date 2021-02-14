# Processes movie_ids.txt and returns a dictionary

def load_movie():

    # A dictionary containing movie_ids and corresponding names
    movie_dict = {}

    with open("movie_ids.txt", "r") as a_file:
        for line in a_file:
            [movie_id, movie_name] = line.split(" ", 1)
            movie_dict[movie_id] = movie_name[:-1]

    return movie_dict
