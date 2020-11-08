import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

smallRating = pd.read_csv("large_files/movielens-20m-dataset/small_rating.csv")
# Basic correlogram
# "userId","movieId","rating","timestamp"
data = smallRating.drop(['userId','timestamp'], axis=1)
sns.pairplot(data)
plt.show()

# print(smallRating.columns)