# importing the relevant libraries
import pandas as pd
import pickle

# load data
main_path ='FILEPATH to folder containing MovieLens Dataset'
movies = pd.read_csv(main_path+'movies.csv')
ratings = pd.read_csv(main_path+'ratings.csv')

# getting rid of the timestamp data column
ratings = ratings.drop(['timestamp'],axis=1)

# getting the number of times each movie was rated
mov_rating_f = pd.DataFrame(ratings.groupby('movieId').size(), columns=['count'])
mov_rating_f.describe()

#getting rid of movies that were not frequently rated
popularity_thres = 22000
popular_movies = list(set(mov_rating_f.query('count >= @popularity_thres').index))
pop_ratings = ratings[ratings.movieId.isin(popular_movies)]
usr_f = pd.DataFrame(pop_ratings.groupby('userId').size(), columns=['count'])

# getting rid of users that were not frequently rating
popularity_thres1 = 150
freq_users = list(set(usr_f.query('count >= @popularity_thres1').index))
active_usr = ratings[ratings.userId.isin(freq_users)]

# storing the number of unique movies and users
uq_movies = active_usr['movieId'].nunique()
uq_users = active_usr['userId'].nunique()

# outputting the processed dataframe to a pickle file so that it can be loaded later
active_usr.to_pickle("filtered dataset.pkl")
