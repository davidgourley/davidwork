#Description: I'm going to create a movie recommender system for people. It will use collaborative filtering which recommends items based on similarities between 
# user preferences or an item's characteristics. Uses libraries numpy, pandas, matplotlib, and seaborn. Preprocess data by removing unneeded columns & merging relevant 
# dataframes to create dataframe that has user ratings for each movie. Visualize data to show distribution of ratings, relationship between number of ratings/average 
# rating, and correlation between movies. Create a user-movie rating matrix (each movie = column. user ratings = matrix values). Create recommendations based on 
# correlations between movies.  Highest correlations to a given movie/set of movies rated by a user are given as recommendations.

#Libraries used: Uses numpy, pandas, matplotlib, seaborn.

#What will it do?: Remove unneeded columns and then merge relevant dataframes to create datagrame that has user ratings for each movie.

#Uses both ML model and data visualization. ML model = machine learning model that suggests movies based on user preferences using collaborative filtering. 
# Data visualization = shows distribution of ratings, reltationship between number of ratings/average rating, and then correlation between movies.

#Goal: In the end, I want to be able to recommend movies to someone based off of their preferences. In my free time, I've been trying to develop a project 
# where users can input ratings for TV shows. While these are different, creating this movie recommender project makes me think I should incorporate it into 
# my own personal project (the tv show ratings project) and maybe apply it to TV shows - or even add movies to it. What I'm going to develop in this project 
# is create a visual representation for movies highly correlated to their highest rated movies and make recommendations based on that.

#Collaborative Filtering = The process used to calculate similarities between
#   the buying trends of various users or similarities between products

#User-Based Collaborative Filtering is dependent on user choices. 
# if two users, X and Y, like products A
# and B and there is another user Z who likes product A, 
# the product B will be recommended to user Z. 

#Item-based Collaborative Filtering eliminates user dependency, and
# even if user choices change over time, the properties of products
# remain unchanged.

#IMPORT LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#IMPORT DATASET
movie_ids_titles = pd.read_csv("C:/Users/david/Downloads/movies.csv")
print(movie_ids_titles.head())

movie_ids_ratings = pd.read_csv("C:/Users/david/Downloads/ratings.csv")
print(movie_ids_ratings.head())

#SHAPE
movie_ids_ratings.shape
print("Shape: ", movie_ids_ratings.shape)

#DATA PREPROCESSING
#Remove genres column from movie_ids_titles dataframe
print("Movies without genre column:")
movie_ids_titles.drop("genres", inplace = True, axis = 1)
print(movie_ids_titles.head()) 

#Remove timestamp column from movie_ids_ratings dataframe
print("Movies without timestamp column:")
movie_ids_ratings.drop("timestamp", inplace = True, axis = 1)
print(movie_ids_ratings.head())

#Merge movie_ids_titles and movie_ids_ratings
print("Merging of movie_ids_ratings and movie_ids_titles:")
merged_movie_df = pd.merge(movie_ids_ratings, movie_ids_titles, on='movieId')
print(merged_movie_df.head())

#DATA VISUALIZATION
#Group the dataset by title & describe
print("Dataset grouped by title & describe:")
describe_dataset = merged_movie_df.groupby('title').describe()
print(describe_dataset)

#Extract mean of ratings grouped by title:
print("Extract mean of ratings grouped by title:")
mean_ratings = merged_movie_df.groupby('title')['rating'].mean().head()
print(mean_ratings)

#Sort movie titles by descending order of average user ratings
print("Movie titles by descending order of average user ratings:")
sorted_ratings = merged_movie_df.groupby('title')['rating'].mean().sort_values(ascending=False).head()
print(sorted_ratings)

#NEW DATAFRAME
#Create new dataframe that shows title, mean rating, and rating counts
print("Dataframe that shows title, mean rating, and rating counts:")
movie_rating_mean_count = pd.DataFrame(columns=['rating_mean', 'rating_count']) 
movie_rating_mean_count["rating_mean"] = merged_movie_df.groupby('title')['rating'].mean()
movie_rating_mean_count["rating_count"] = merged_movie_df.groupby('title')['rating'].count()
print(movie_rating_mean_count.head())

#Plot a histogram to view average ratings distribution
plt.figure(figsize=(10,8))
sns.set_style("darkgrid")
movie_rating_mean_count['rating_mean'].hist(bins=30, color= "blue")
plt.xlabel('Average Rating')
plt.ylabel('Movies')
plt.title('Distribution of Average Ratings')
plt.show()

#Plot distribution of rating counts
plt.figure(figsize=(10,8))
sns.set_style("darkgrid")
movie_rating_mean_count['rating_count'].hist(bins=33,color = "red") 
plt.xlabel('Rating Count')
plt.ylabel('Movies')
plt.title('Distribution of Rating Counts')
plt.show()

#Scatterplot to show relationship between mean ratings and rating counts of a movie
plt.figure(figsize=(10,8))
sns.set_style("darkgrid")
sns.regplot(x="rating_mean", y="rating_count", data=movie_rating_mean_count, color = "green")
plt.xlabel('Average Rating')
plt.ylabel('Rating Count')
plt.title('Relationship between Mean Ratings and Rating Counts')
plt.show()

#Top 5 highest number of ratings:
print("Top 5 highest number of ratings:")
top_5_highest_ratings = movie_rating_mean_count.sort_values("rating_count", ascending=False).head()
print(top_5_highest_ratings)

#------------------------------------------------

#Item-based Collaborative Filtering
#We'll use the avg ratings as the common characteristic of the
# collaborative filtering of movies

#Create a dataframe where each movie is represented by a column
# and rows contain user ratings for movies. Use pivot_table() function
# of a Pandmas dataframe.
print("New dataframe w/columns representing movies and rows for user ratings:")
user_movie_rating_matrix = merged_movie_df.pivot_table(index='userId', columns='title', values='rating')
print(user_movie_rating_matrix)

#Plot shape of dataframe:
print("Shape of the Dataframe: ", user_movie_rating_matrix.shape)

#Find Recommendations Based on a Single Movie:
# example: Pulp Fiction (1994)
print("Find recommendations based on a single movie (Pulp Fiction):")
pulp_fiction_ratings = user_movie_rating_matrix["Pulp Fiction (1994)"]
#Find correlation between user ratings of all movies &
# user ratings for Pulp Fiction.
print("Find correlation between user ratings of all movies & user ratings for Pulp Fiction")
pulp_fiction_correlations = pd.DataFrame(user_movie_rating_matrix.corrwith(pulp_fiction_ratings), columns =["pf_corr"]) 
print(pulp_fiction_correlations.sort_values("pf_corr", ascending=False).head(5))
#Correlation isn't accurate. Highest correlation = Movies w/5 star ratings
#Add rating counts to try and improve issue:
print("Add rating counts to try and improve issue:")
pulp_fiction_correlations = pulp_fiction_correlations.join(movie_rating_mean_count["rating_count"])
print(pulp_fiction_correlations.head())
#Null ratings = there are movies rated by users who didn't rate Pulp Fiction
#Remove all movies w/Null correlation w/Pulp Fiction
print("#Remove all movies w/Null correlation w/Pulp Fiction")
pulp_fiction_correlations.dropna(inplace = True) 
print(pulp_fiction_correlations.sort_values("pf_corr",ascending=False).head(5))
#Find movies w/rating counts of 50+ & having highest correlation w/Pulp Fiction:
print("Find movies w/rating counts of 50+ & having highest correlation w/Pulp Fiction:")
pulp_fiction_correlations_50 = pulp_fiction_correlations[pulp_fiction_correlations['rating_count']>50]
print(pulp_fiction_correlations_50.sort_values('pf_corr',ascending=False).head())

#---------------------------------------------------------------

#Find Recommendations Based on Multiple Movies:
#Create a dataframe which has a correlation between all movies in
# dataset in the form of a matrix.
print("Dataframe which has a correlation between all movies in dataset in the form of a matrix:")
all_movie_correlations = user_movie_rating_matrix.corr(method = 'pearson', min_periods = 50)
print(all_movie_correlations.head())

#Create new dataframe that contains fictional ratings given by a user to 3 movies:
print("New datarame that contains fictional ratings given by a user:")
movie_data = [['Forrest Gump (1994)', 4.0], ['Fight Club (1999)', 3.5], ['Interstellar (2014)', 4.0]]
test_movies = pd.DataFrame(movie_data, columns = ['Movie_Name', 'Movie_Rating'])
print(test_movies.head())

#To get name & ratings of a movie from test_movie dataframe, print this:
#print(test_movies['Movie_Name'][0])
#print(test_movies['Movie_Rating'][0])

#Obtain correlation values for movies related to Forrest Gump from
# all_movie_correlations dataframe
print("Obtain correlation values:")
print(all_movie_correlations['Forrest Gump (1994)'].dropna())

#Next, we'll iterate through the 3 movies in the test_movie
# dataframe, find correlated movies, and then multiply the
# correlation of all the correlated movies w/ratings of the
# input movie. The correlated movies, along with the weighted
# correlation (calculated by multiplying the actual correlation
# with the ratings of the movies in the test_movie dataframe),
# are appended to an empty series named recommended_movies.
print("Iterate through dataframe and give correlation values:")
recommended_movies = pd.Series()
for i in range(0, 2):
    movie = all_movie_correlations[test_movies['Movie_Name'][i]].dropna()
    movie = movie.map(lambda movie_corr: movie_corr *test_movies['Movie_Rating'][i])
    recommended_movies = recommended_movies._append(movie)
print(recommended_movies)

#Final recommendation = Sort movies in descending order of weighted correlation:
print("Top 10 Recommended Movies from Forrest Gump:")
recommended_movies.sort_values(inplace = True, ascending = False)
print(recommended_movies.head(10))

#You can see from the above output that Forrest Gump and Fight Club
# have the highest correlation with themselves. Hence, they are 
# recommended. The movie Interstellar doesn’t appear on the list because 
# it might not have passed the minimum 50 ratings thresholds.
# The remaining movies are the movies recommended by our
# recommender system to a user who watched Forrest Gump, Fight Club,
# and Interstellar.









