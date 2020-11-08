import pandas as pd
# preprocess the raw data (create mappings etc)

rating = pd.read_csv("large_files/movielens-20m-dataset/rating.csv")

# make it zero based
rating.userId = rating.userId-1

unique_movie_ids = set(rating.movieId.values)
movie2idx = {}
count = 0
for movieId in unique_movie_ids:
    movie2idx[movieId] = count
    count += 1

rating['movie_idx'] = rating.apply(lambda row: movie2idx[row.movieId], axis=1)
rating = rating.drop(columns=['timestamp'])

rating.to_csv("large_files/movielens-20m-dataset/edited_rating.csv")