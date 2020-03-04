import pandas as pd
from surprise import Reader,Dataset, accuracy, dump, SVDpp
from surprise import prediction_algorithms, KNNWithZScore
import surprise
from surprise.model_selection import cross_validate, GridSearchCV
import numpy as np
from fuzzywuzzy import fuzz
from math import isnan
from operator import itemgetter
from collections import defaultdict
import pickle
import scipy
#from funk_svd import SVD
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
# visualization imports
import seaborn as sns
import matplotlib.pyplot as plt

def fuzzy_matching(fav_movie, movies_id_dc, *args):
    """
    return the closest match via fuzzy ratio. If no match found, return None

    Parameters
    ----------
    mapper: dict, map movie title name to index of the movie in data

    fav_movie: str, name of user input movie

    verbose: bool, print log if True

    Return
    ------
    index of the closest match
    """
    verbose = True
    matches = []
    # get match
    for key in movies_id_dc:
        ratio = fuzz.ratio(str(key).lower(), fav_movie.lower())
        if ratio >= 60:
            matches.append((key, movies_id_dc[key], ratio))
    # sort
    matches = sorted(matches, key=lambda x: x[2])[::-1]
    if not matches:
        print('Oops! No match is found')
        return
    if verbose:
        print('Found possible matches in our database: {0}\n'.format([x[0] for x in matches]))
    return matches[0][1]


def make_recommendation(data, usr_pref, user_data, user_id):
    """
    return top n similar movie recommendations based on user's input movie


    Parameters
    ----------
    model_knn: sklearn model, knn model

    data: movie-user matrix

    mapper: dict, map movie title name to index of the movie in data

    fav_movie: str, name of user input movie

    n_recommendations: int, top n recommendations

    Return
    ------
    list of top n similar movie recommendations
    """
    # get input movie index
    #data = data.sample(frac=0.1, random_state=7)
    #d = {'userId': [user_id], 'movieId': [idx], 'rating': [5.0]}
    df2 = usr_pref
    #pd.DataFrame(d)
    new_user_data = pd.read_csv('new user ratings.csv')
    #df2 = df2[['userId', 'movieId']]
    #df2 = df2.set_index("userId", inplace=True)
    #print(df2.head())
    data = data.append(df2)
    data.reset_index(inplace=True, drop=True)
    #print(data.tail())

    # populating test data for user predictions

    #user_data.set_index("userId", inplace=True)

    reader = Reader(rating_scale=(1, 5))
    # reader2 = Reader()
    data_dr = Dataset.load_from_df(data[['userId', 'movieId', 'rating']], reader)
    model = dump.load('Complete SVD v1.12')
    algo = model[1][1]
    user_dr = Dataset.load_from_df(new_user_data[['userId', 'movieId', 'rating']], reader)
    #algo = surprise.SVD(n_factors = 10, n_epochs=5, lr_all=0.004, reg_all=0.04, verbose=True)
    #algo = KNNWithZScore(k=40, min_k=1, sim_options={'name': 'pearson_baseline', 'user_based': False}, verbose=True,)
    algo.fit(data_dr.build_full_trainset())
    predictions = []
    #user_ar = user_data.to_numpy()
    uids = user_data['userId'].to_numpy()
    iids = user_data['movieId'].to_numpy()
    i=0
    while i < len(uids):
        predictions.append(algo.predict(uid=uids[i], iid=iids[i]))
        i+=1
    return predictions

def get_top_n(predictions, id_movies_dc, *args):
    '''Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    '''

    # i=0
    # results = []
    # while i <len(predictions):
    #     results.append((predictions[i][1], predictions[i][2]))
    #
    # sorted_est = sorted(results, key=lambda tup: tup[0])
    # result = sorted_est[:n]
    # return result


    # First map the predictions to each user.
    top_n = []
    for prediction in predictions:
        top_n.append((id_movies_dc[int(prediction.iid)], prediction.est))
    #top_n.sort(reverse=True)



    # # Then sort the predictions for each user and retrieve the k highest ones.
    # top_n[uid].sort(key=lambda x: x[1], reverse=True)
    # top_n[uid] = top_n[uid][:n]
    return top_n

    # # First map the predictions to each user.
    # top_n = defaultdict(list)
    # uid = predictions[0][0]
    #
    # for iid, true_r, est, _ in predictions:
    #     top_n.append((iid, est))
    #
    # # Then sort the predictions for each user and retrieve the k highest ones.
    # for est in top_n.items():
    #     est.sort(key=lambda x: x[1], reverse=True)
    #     top_n[uid] = user_ratings[:n]
    #
    # return top_n




main_path ='/Users/Avijit/Documents/GitHub/MyDataSciencePortfolio/movie_recommender/Movie SysRec/ml-latest/'
movies = pd.read_csv(main_path+'movies.csv')
data = pd.read_pickle("filtered dataset.pkl")
movies.drop_duplicates(subset =['title'], inplace=True)
#print(movies.title.head())
movies = movies[movies.movieId.isin(list(data.movieId))]
movies = movies.drop(['genres'], axis=1)
movies.dropna(inplace=True)
#data.reset_index(inplace=True, drop=True)
# print(data.head())
# print(data.shape)

#movie_id_dc = pd.Series(movies.movieId,index=movies.title).to_dict()
id_movies_dc = movies.set_index('movieId').T.to_dict('list')
movies_id_dc = movies.set_index('title').T.to_dict('list')
#movies_id_dc = pd.Series(movies['movieId'], index=movies['title']).to_dict()

#id_movies_dc = pd.Series(movies.title,index=movies['movieId']).to_dict()
#movies_id_dc = {id_movies_dc[k] : k for k in id_movies_dc}
#movies_id_dc = {v: k for k, v in id_movies_dc.items()}
#print(id_movies_dc[318])
#movies_id_dc = {k: id_movies_dc[k] for k in id_movies_dc if not isnan(k)}
# print(id_movies_dc)
#print([id_movies_dc[64997]])
#print(movies_id_dc['Jumanji (1995)'])

print("Enter the number of movies you will rate")
nofrates = int(input())

usr_movies =[]
usr_rates = []
usr_iids = []
i=0

while i < nofrates:
    print("Enter the title of movie {} below".format(i+1))
    fav_movie = str(input())
    usr_movies.append(fav_movie)
    print('You selected:', fav_movie)
    idi = fuzzy_matching(fav_movie, movies_id_dc=movies_id_dc)
    while idi == None:
        print("Please enter a different movie")
        fav_movie = str(input())
        idi = fuzzy_matching(fav_movie, movies_id_dc=movies_id_dc)
    idx = idi[0]
    usr_iids.append(idx)
    print("Enter your rating (range:1-5, 0.5 increments) for movie {} below".format(i+1))
    rating = str(input())
    usr_rates.append(rating)
    i+=1

print("Enter the total number of movie recommendations you want")
n = int(input())

preference = {'userId': (data['userId'].max()+1), 'movieId': usr_iids , "rating": usr_rates}

usr_pref = pd.DataFrame(preference)
#print(idx)
# inference
print('Finding the best recommendations based on your taste using machine learning')
print('......\n')
d = {'movieId': []}
user_data = pd.DataFrame(d)
user_data['movieId'] = movies['movieId']
user_id = data['userId'].max()+1
user_data['userId'] = data['userId'].max()+1
user_data = user_data[['userId', 'movieId']]
#user_data.set_index("userId", inplace=True)
# print(user_data.head())


predictions = make_recommendation(data=data, usr_pref= usr_pref, user_data=user_data,user_id=user_id)
#print(predictions[0:10])

result2 = pd.DataFrame(data=predictions)
titles = []
for movieID in result2['iid'].values:
    titles.append(str(id_movies_dc[movieID]))

# helping less popular movies rise up the list by penalizing the estimated rating of a subset of the most popular ones
mov_rating_f = pd.DataFrame(data.groupby('movieId').size(), columns=['count'])
popularity_thres = 3000
popular_movies_i = list(set(mov_rating_f.query('count >= @popularity_thres').index))
pop_movies = data[data.movieId.isin(popular_movies_i)]
pmovids = pop_movies['movieId']

penalty = 0.3
p=0
for id in result2.iid.values:
    if result2.iid[p] in pmovids.values:
        result2.est.values[p] = result2.est.values[p] - penalty
    p+=1

result2['Titles'] = titles
result2.sort_values(by=['est'], inplace=True, ascending=False)
result2=result2[['Titles', 'est']]
result2 = result2.rename(columns={"est": "Your Predicted Rating"})
result2.to_csv('Movie Recommendations.csv')
print(result2.head(n))


# result2 = result2['iid', 'est']
# result2.sort_values(by=['est'])

# result2['Titles'] = id_movies_dc[result2['iid']]
# result = get_top_n(predictions, id_movies_dc=id_movies_dc)
#
# a = pd.DataFrame(result, columns =['movie', 'score'])
#
# def sorted(a):
#     return a.sort_values(by=['score'],  ascending=False)
#
# a['score'].apply(pd.to_numeric)
# a.drop_duplicates(subset="movie", keep='first', inplace=True)
# a=sorted(a)
# #a.sort_values(by=['score'])
# print(a.head(n))
# a.to_csv('recommended movies.csv')
# for t in result:
#     print(id_movies_dc[t[0]], t[1])


#for uid, user_ratings in top_n.items():
    #print(uid, [id_movies_dc[iid] for (iid, _) in user_ratings])


