import pandas as pd

rating = pd.read_csv("large_files/movielens-20m-dataset/rating.csv")

nunique_movies = rating.movieId.nunique()
nunique_users = rating.userId.nunique()
print(f"movies {nunique_movies}\nusers {nunique_users}")