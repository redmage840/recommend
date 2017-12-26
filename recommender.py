# This is the new test line for the new branch called 'refine'
# Load external data file into "df"
# Ensure proper header/index/column labeling as "userID", "gameID", "rating"
# Call predict_rating(userID, gameID)
# That's it! You can return the top eight suggested gameID with get_top_suggestions(userID)

# Try averaging predictions with average gameID rating
import numpy
import pandas

# header = ['userID', 'gameID', 'rating']
header_test = ['userID', 'gameID']
df1 = pandas.read_csv('inputs/boardgame-elite-users.csv').rename(columns = {'Compiled from boardgamegeek.com by Matt Borthwick':'userID'})
df2 = pandas.read_csv('inputs/boardgame-frequent-users.csv').rename(columns = {'Compiled from boardgamegeek.com by Matt Borthwick':'userID'})
frames = [df1, df2]
df = pandas.concat(frames)

my_text = numpy.loadtxt('inputs/boardgame-elite-users.csv', delimiter = ',', skiprows = 1, dtype = numpy.float16)
userIDs = numpy.unique(my_text[:,0])
gameIDs = numpy.unique(my_text[:,1])
ratings = numpy.unique(my_text[:,2])

# print(my_text)

df_test_matrix = numpy.loadtxt('inputs/boardgame-users-test.csv', delimiter = ',')
all_data_ptable = pandas.pivot_table(df, index='userID', columns='gameID', values='rating', fill_value=0)
ratings_array = numpy.array(all_data_ptable)

# from scipy.sparse import issparse
# print(issparse(all_matrix))

from sklearn.metrics.pairwise import pairwise_distances
user_similarity = pairwise_distances(ratings_array, metric='cosine')

# Takes a 2d array (199, 402) and a 2d array (199, 199), returns a 2d array (199, 402) of the predicted rating
# for each userID-gameID pair
def derive_prediction_matrix(ratings_arr, user_simil):
    mean_user_rating = ratings_arr.mean(axis=1)
    ratings_minus_mean = (ratings_arr - mean_user_rating[:, numpy.newaxis])
    # am I adding mean_user_rating back twice? this still doesn't make sense below
    pred = mean_user_rating[:, numpy.newaxis] + user_simil.dot(ratings_minus_mean) / numpy.count_nonzero(ratings_arr.T, axis = 1)
    # Add back mean_user_rating
    pred = pred + mean_user_rating[:, numpy.newaxis]
    return pred

user_prediction_matrix = derive_prediction_matrix(ratings_array, user_similarity)

# Helper functions, round to a precision
def round_to(n, precision):
    correction = 0.5 if n >= 0 else -0.5
    return int( n/precision+correction ) * precision
def round_to_p5(n):
    return round_to(n, 0.5)
# Takes a float, ceilings it at 10.0, returns float
def max_ten(some_float):
    return 10.0 if some_float > 10 else some_float

# Takes 2 ints, returns a float rounded to nearest half (1.0, 1.5, 2.0)
# change this so I don't need to reference ptable
def predict_rating(userID, gameID):
    user_location = numpy.where(all_data_ptable.index.values == userID)
    game_location = numpy.where(all_data_ptable.columns == gameID)
    return max_ten(round_to_p5(user_prediction_matrix[user_location, game_location]))
print(all_data_ptable.index.values)

# Takes an int and returns a list of 8 ints (gameIDs)
# def get_top_suggestions(userID):
#     rating_game_tuples = []
#     for gameID in df[:,1]:
#         if all_data_ptable.loc[(userID, gameID)] < .1:
#             rating = predict_rating(userID, gameID)
#             rating_game_tuples.append((rating, gameID))
#     # return gameID list of highest 8 predicted ratings
#     return [x[1] for x in sorted(rating_game_tuples, reverse = True)[:9]]

my_file = open("users.csv","w+")

for elem in df_test_matrix:
    if elem[0] in all_data_ptable.index.values:
        if 6 < predict_rating(elem[0], elem[1]) < 9:
            my_file.write(str(elem[0]) + ',' + str(elem[1]) + ',' + str(predict_rating(elem[0], elem[1])) + '\n')
my_file.close()

#**********************************************************************************************
# EVALUATION, RMSE
# from math import sqrt
# def rmse(predictions, targets):
#     return numpy.sqrt(((predictions - targets) ** 2).mean())