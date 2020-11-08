import pandas as pd
import numpy as np
from collections import Counter

rating_edited = pd.read_csv("large_files/movielens-20m-dataset/edited_rating.csv")
print("Original dataframe size: ", len(rating_edited))

N = rating_edited.userId.max() + 1
M = rating_edited.movieId.max() + 1

user_ids_count = Counter(rating_edited.userId)
movie_ids_count = Counter(rating_edited.movie_idx)

n = 10000 # number of users we would like to keep
m = 2000 # number of movies we would like to keep

user_ids = [u for u, c in user_ids_count.most_common(n)]
movie_ids = [m for m, c in movie_ids_count.most_common(m)]

rating_small = rating_edited[rating_edited.userId.isin(user_ids) & rating_edited.movie_idx.isin(movie_ids)].copy()

new_user_id_map = {}
i = 0
for old in user_ids:
    new_user_id_map[old] = i
    i += 1
print("i: ", i)

new_movie_id_map = {}
j = 0
for old in movie_ids:
    new_movie_id_map[old] = j
    j += 1
print("j: ", j)

print("setting new ids")
rating_small.loc[:, 'userId'] = rating_small.apply(lambda row: new_user_id_map[row.userId], axis=1)
rating_small.loc[:, 'movie_idx'] = rating_small.apply(lambda row: new_movie_id_map[row.movie_idx], axis=1)
print("max user id: ", rating_small.userId.max())
print("max movie id: ", rating_small.movie_idx.max())

print("small dataframe size: ", len(rating_small))
rating_small.to_csv("large_files/movielens-20m-dataset/u10k_m2k_rating.csv")