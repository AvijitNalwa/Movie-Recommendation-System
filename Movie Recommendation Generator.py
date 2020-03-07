import pandas as pd
from surprise import Reader,Dataset, accuracy, dump, SVDpp
from surprise import prediction_algorithms
import surprise
import numpy as np
from fuzzywuzzy import fuzz
from math import isnan
import pickle


# function to find movies in the database based on user selection
def movie_matching(fav_movie, movies_id_dc, *args):
    verbose = True
    matches = []
    # get match by comparing database movies with user input string
    for key in movies_id_dc:
        ratio = fuzz.ratio(str(key).lower(), fav_movie.lower())
        if ratio >= 60:
            matches.append((key, movies_id_dc[key], ratio))
    # sort matches based on how similar they are
    matches = sorted(matches, key=lambda x: x[2])[::-1]
    if not matches:
        print('Oops! No match is found')
        return
    if verbose:
        print('Found possible matches in our database: {0}\n'.format([x[0] for x in matches]))
    return matches[0][1]


def find_recommendations(data, usr_pref, user_data, user_id):
    # adding the user preferences dataframe to our train data
    df2 = usr_pref
    data = data.append(df2)
    data.reset_index(inplace=True, drop=True)

    # making movielens data compatible with the model by creating a reader object
    reader = Reader(rating_scale=(1, 5))
    data_dr = Dataset.load_from_df(data[['userId', 'movieId', 'rating']], reader)

    # defining our model based on the parameters that yielded the lowest rmse while having the most efficient running time
    # notice that we use n_factors=10 for two reasons, 1: it drastically reduces the amount of time to train the model.
    # 2: it still results in high model accuracy, and reduces the dimension of the subspace being worked with in the SVD, providing
    # an efficient approximation [we get the most significant factors explaining a majority of user behaviour]
    algo = surprise.SVD(n_factors=10, n_epochs=10, lr_all=0.03, reg_all=0.04, verbose=True)

    # training the model on the dataset that now includes the users preferences
    algo.fit(data_dr.build_full_trainset())

    # list to hold the models predictions
    predictions = []

    # getting arrays that can be traversed of the user id, movie (item) id,
    # these are used to apply the predict function
    uids = user_data['userId'].to_numpy()
    iids = user_data['movieId'].to_numpy()

    # counting variable
    i=0

    # loop to add predictions for each uid, movieid pair to the predictions list defined above
    while i < len(uids):
        predictions.append(algo.predict(uid=uids[i], iid=iids[i]))
        i+=1

    # making the function return the list containing our predictions
    return predictions

# loading the movie data that will allow us to create our title, movieID dictionaries
main_path ='FILEPATH to the folder containing the Movielens data'
movies = pd.read_csv(main_path+'movies.csv')

# loading our pickle file containing the dataset resulting from our processing steps in the data prep python file
data = pd.read_pickle("filtered dataset.pkl")

# making sure we do not have any duplicate movie title entries in the movies data so that we can create a one to one dictionary
movies.drop_duplicates(subset =['title'], inplace=True)

# reducing the movies dataframe to one that only contains movies that are also in our main rating data file
movies = movies[movies.movieId.isin(list(data.movieId))]

# getting rid of the genres column from the movies dataframe as it is not going to be used directly in this model
movies = movies.drop(['genres'], axis=1)

# drop any entries that have a nan value
movies.dropna(inplace=True)

# creating an id -> title dictionary for the movies by manipulating the dataframe (change index, then transpose)
id_movies_dc = movies.set_index('movieId').T.to_dict('list')

# creating a title -> id dictionary for the movies by manipulating the dataframe (change index, then transpose)
movies_id_dc = movies.set_index('title').T.to_dict('list')

# User prompt to decide upper limit for preference detecting data
print("Enter the number of movies you will rate")
nofrates = int(input())

# list variables to store user inputs that will be used to populate preference dataframe
usr_movies =[]
usr_rates = []
usr_iids = []

# counting variable for the while loop
i=0

while i < nofrates:
    print("Enter the title of movie {} below".format(i+1))
    fav_movie = str(input())
    # add movie to the list
    usr_movies.append(fav_movie)
    print('You selected:', fav_movie)

    # use the matching function to find the closest movie to the user input in the database
    idi = movie_matching(fav_movie, movies_id_dc=movies_id_dc)
    # loop to handle error if there is no match
    while idi == None:
        # ask user again, hopefully a match is found
        print("Please enter a different movie")
        fav_movie = str(input())
        # call matching function again to find a match, this repeats till a match occurs
        idi = movie_matching(fav_movie, movies_id_dc=movies_id_dc)
    # creating a variable to store the movieID for the user entered movie, our dict that maps titles to ids gives a one element list
    idx = idi[0]
    # add the movieID to the user entered movieID list
    usr_iids.append(idx)

    # Accept rating for the chosen movie
    print("Enter your rating (range:1-5, 0.5 increments) for movie {} below".format(i+1))
    rating = str(input())

    # add the rating to the user entered movie ratings list
    usr_rates.append(rating)
    # increment the counting variable so that the the input while loop breaks as desired
    i+=1

# accept the number of recommendations that should be printed
print("Enter the total number of movie recommendations you want")
n = int(input())

# creating a preference dictionary containg data titles and the lists that were populated above
preference = {'userId': (data['userId'].max()+1), 'movieId': usr_iids , "rating": usr_rates}

# creating a pandas dataframe and populating it with data from the preferences list above, keys are column names, values are column data
usr_pref = pd.DataFrame(preference)

print('Finding the best recommendations based on your taste using machine learning')
print('..........\n')

# creating a dictionary to help create the user_data dataframe
d = {'movieId': []}
# creating the user_data dataframe that is used to generate predicitons
user_data = pd.DataFrame(d)

# list to hold the test ids
test_ids = []

# counter variable to traverse
k=0

# simplifying notation
all_ids = movies['movieId'].to_numpy()

# loop to add movie ids to the test list only if they were not entered by the user
while k < len(movies['movieId'].to_numpy()):
    if movies.movieId.to_numpy()[k] not in usr_iids:
        test_ids.append(all_ids[k])
    k+=1

# populating the testing dataframe with movieIds from our dataset
user_data['movieId'] = test_ids

# creating a new, unique user by finding a userId that is not used yet
user_id = data['userId'].max()+1
# setting the userId column of the test dataset to the new unique users uid
user_data['userId'] = data['userId'].max()+1
# reordering test dataframe columns
user_data = user_data[['userId', 'movieId']]

# calling the recommendation function and storing the list of predictions in a variable
predictions = find_recommendations(data=data, usr_pref= usr_pref, user_data=user_data,user_id=user_id)

# creating a dataframe and populating it with the models predicitons data
result2 = pd.DataFrame(data=predictions)

# creating a list variable to hold our recommended movie's titles
titles = []

# adding the movie title of each recommended movie to our list variable defined above
for movieID in result2['iid'].values:
    # we use our movieID -> movie title dictionary and use the movieID as the key value, then convert the result to a string and append
    titles.append(str(id_movies_dc[movieID]))

# ** helping less popular movies rise up the list by penalizing the estimated rating of a subset of the most popular ones **
# getting the rating frequency for each movie in the form of a dataframe
mov_rating_f = pd.DataFrame(data.groupby('movieId').size(), columns=['count'])

# setting a threshold for ratign frequency (as a measure of popularity), we used this number after using the .describe() function on the frequency dataframe above
popularity_thres = 3000

# getting a list of the indexs of the popular movies
popular_movies_i = list(set(mov_rating_f.query('count >= @popularity_thres').index))

# creating a DataFrame containing only the popular movies using the index list
pop_movies = data[data.movieId.isin(popular_movies_i)]

# simplifying notation by storing the popular movie ids column in a variable
pmovids = pop_movies['movieId']

# **** PENALTY, we set the rating penalty we want to impose on very popular movies so that relatively less popular, yet tasteful movies can rise to the top ***
# play around with the penalty value to shift the number of popular movies in the top recommendations
penalty = 0.3

# counting variable to traverse the arays of estimated ratings and movie (item) ids from the dataframe containing recommendations
p=0

# we loop through every movieid
for id in result2.iid.values:
    # if the movie id is in the array of popular movie ids
    if result2.iid[p] in pmovids.values:
        # apply the penalty to the corresponding rating estimation (prediction)
        result2.est.values[p] = result2.est.values[p] - penalty
    # otherwise leave unpopular movies unaffected by the penalty
    # increment the counting variable to continue traversing arrays
    p+=1

# adding the Titles column to the results2 dataframe, adding the recommendation titles as values of the column
result2['Titles'] = titles
# sort the recommendations by the predicted rating to make sure the best recommendations are on top
result2.sort_values(by=['est'], inplace=True, ascending=False)
# reducing the results2 dataframe to just the title and estimated ratings column
result2=result2[['Titles', 'est']]
# renaming the 'est' column to a more intuitive name
result2 = result2.rename(columns={"est": "Your Predicted Rating"})
# writing the final, sorted & titled recommendations to a csv file
result2.to_csv('Movie Recommendations.csv')

# print out the top n results by using the head function on the dataframe
print(result2.head(n))
