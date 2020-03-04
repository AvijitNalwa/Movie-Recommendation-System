# Movie Recommendation System
A machine learning model that detects your taste in movies and suggests the ones you would like the most.

# Method
I use Collaborative Filtering via Singular Value Decomposition (SVD) implemented well by the Surprise (http://surpriselib.com/) python library.  

# Data
The Dataset: MovieLens Full Dataset (https://grouplens.org/datasets/movielens/latest/)
 
Numbers:
 
27,000,000 ratings by 280,000 users for 58,000 movies 
 
I use the ratings, and movies files. Though this original dataset is huge, I add restrictions to reach a much smaller size containing more frequently rated movies, and more frequently rating users. Along with this I preprocess the data and save it to a pickle file for ease of use through the project. (Find the restriction & processing in the Data Pre-Processing.py file)

# Model
 I use the surprise implementation of the SVD & SGD (Stochastic Gradient Descent) model based on Simon Funk's Netflix Prize approach. I also built a model using SVD++(pp) which accounts for implicit ratings (based on the assumption that rating/not rating a movie is an indication of some preference). However, after testing both approaches I found that the more traditional SVD approach yielded more efficient and better generalizing results. 

# General Structure
 The main file: Movie Recommendation Generator.py allows a user to rate as many/few>1 movies as they wish, and then provides them with the n (of their choosing) best movie recommendations. It also produces a .csv file containing sorted data of predicted titles and ratings.

# References
 I referred/used certain aspects of the following resources:
 
https://grouplens.org/datasets/movielens/latest/
 
http://surpriselib.com/

https://stackoverflow.com/

https://github.com/AvijitNalwa/MyDataSciencePortfolio/tree/master/movie_recommender



