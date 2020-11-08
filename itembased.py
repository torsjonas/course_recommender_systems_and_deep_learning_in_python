import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from datetime import datetime
from sortedcontainers import SortedList
import os

if not os.path.exists('jonas/item-item-recs/user2movie.pickle') or \
   not os.path.exists('jonas/item-item-recs/movie2user.pickle') or \
   not os.path.exists('jonas/item-item-recs/usermovie2rating.pickle') or \
   not os.path.exists('jonas/item-item-recs/usermovie2rating_test.pickle'):
   import preprocess2dict


with open('jonas/item-item-recs/user2movie.pickle', 'rb') as f:
  user2movie = pickle.load(f)

with open('jonas/item-item-recs/movie2user.pickle', 'rb') as f:
  movie2user = pickle.load(f)

with open('jonas/item-item-recs/usermovie2rating.pickle', 'rb') as f:
  usermovie2rating = pickle.load(f)

with open('jonas/item-item-recs/usermovie2rating_test.pickle', 'rb') as f:
  usermovie2rating_test = pickle.load(f)

N = np.max(list(user2movie.keys())) + 1

# may see different users in train and test
m1 = np.max(list(movie2user.keys()))
m2 = np.max([m for (u,m) in usermovie2rating_test.items()])
M = max(m1, m2) + 1
print("N: ", N, "M: ", M)

K_NEAREST_NEIGHBORS = 20
NUM_COMMON_LIMIT = 5
rating_averages = {}
user_deviations = {}
neighbors = {}

#item -> j and jprime
#users that rated j -> usersj and usersjprime

for j, usersj in movie2user.items():
    j_ratings = [usermovie2rating[userId, j] for userId in usersj]
    j_avg_rating = np.mean(j_ratings)
    rating_averages[j] = j_avg_rating

print("Calculated average ratings")

for j in range(M):
    print("Calculating weights for movie: ", j)
    if not j in neighbors.keys():
        neighbors[j] = SortedList()

    usersj = movie2user[j]

    for jprime in range(M):
        if (j == jprime):
            continue

        usersjprime = movie2user[jprime]
        users_who_rated_both = list((set(usersj) & set(usersjprime)))
        if len(users_who_rated_both) < NUM_COMMON_LIMIT:
            continue

        sum_products = 0
        sum_squares_j = 0
        sum_squares_jprime = 0
        for user in users_who_rated_both:
            ij_deviation = usermovie2rating[user, j] - rating_averages[j]
            ijprime_deviation = usermovie2rating[user, jprime] - rating_averages[jprime]
            sum_products = sum_products + ij_deviation * ijprime_deviation
            sum_squares_j = sum_squares_j + np.square(ij_deviation)
            sum_squares_jprime = sum_squares_jprime + np.square(ijprime_deviation)

            # keep all user deviations for later use in prediction
            user_deviations[(user, jprime)] = usermovie2rating[user, jprime] - rating_averages[jprime]

        weight_jjprime = sum_products / (np.sqrt(sum_squares_j) * np.sqrt(sum_squares_jprime))

        # SortedList sorts ascendingly, we want it the other way so add the negative value
        neighbors[j].add((-weight_jjprime, jprime))
        if len(neighbors[j]) > K_NEAREST_NEIGHBORS:
            del neighbors[j][-1]

def predict(movie, user):
    numerator = 0
    denominator = 0
    for weight, neighbor_movie in neighbors[movie]:
        try:
            deviation = -(user_deviations[(user, neighbor_movie)])
            numerator = numerator + weight * deviation
            denominator = denominator + abs(weight)
        except KeyError:
            # user has not rated this neighbor movie, just skip it
            pass

    if denominator == 0:
        return rating_averages[movie]

    return numerator / denominator + rating_averages[movie]

train_predictions = []
train_targets = []
for (u,m), target in usermovie2rating.items():
    train_predictions.append(predict(m, u))
    train_targets.append(target)

test_predictions = []
test_targets = []
for (u,m), target in usermovie2rating_test.items():
    test_predictions.append(predict(m, u))
    test_targets.append(target)

# calculate accuracy
def mse(p, t):
  p = np.array(p)
  t = np.array(t)
  return np.mean((p - t)**2)

print('train mse:', mse(train_predictions, train_targets))
print('test mse:', mse(test_predictions, test_targets))