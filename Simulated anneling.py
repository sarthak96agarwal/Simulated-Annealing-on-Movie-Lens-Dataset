import pandas as pd
import numpy as np
import math
#100004 ratings and 671 users across 9125 movies.

def cost(movie_feat_vec, user_feat_vec, l1, l2, matrix, is_rated, no_movies, no_users, no_feat):
	new_cost=np.sum(np.sum(np.square(matrix-np.multiply(np.dot(movie_feat_vec,
	np.transpose(user_feat_vec)), is_rated))))+np.sum(np.sum(np.square(movie_feat_vec)))*l1 + np.sum(np.sum(np.square(user_feat_vec)))*l2
	return new_cost/100000


print("Reading the ratings file...")
Rating=pd.read_csv('ratings.csv')
no_users = Rating['userId'].max()
no_movies = Rating['movieId'].max()
matrix = np.zeros(shape = (no_movies, no_users))
for i in range(0,len(Rating)):
	matrix[Rating['movieId'].iloc[i]-1, Rating['userId'].iloc[i]-1] = Rating['rating'].iloc[i]
print("Normalizing the rating matrix...")


##normalizing the rating matrix
is_rated = matrix!=0
mean_rating = np.zeros(shape=(matrix.shape[0], 1))
norm_rating = np.zeros(shape=(matrix.shape))
for i in range(0, no_movies):
	t=is_rated[i,]==1
	indexes = [j for j, x in enumerate(t) if x]
	if not indexes:
		continue
	mean_rating[i]=np.mean(matrix[i,indexes])
	norm_rating[i,indexes] = matrix[i,indexes] - mean_rating[i];

no_feat = 3
s = np.random.normal(0, 1, no_movies*no_feat)
movie_feat_vec=np.reshape(s, (no_movies,no_feat))
t = np.random.normal(0, 1, no_users*no_feat)
user_feat_vec=np.reshape(t, (no_users,no_feat))
l1=np.random.normal(0, 0.1, 1)
l2=np.random.normal(0, 0.1, 1)
##applying simulated anneling
print ("Applying Simulated Annealing....")

initial_cost=cost(movie_feat_vec, user_feat_vec, l1, l2, norm_rating, is_rated, no_movies, no_users, no_feat)
no_iter=10
steps=0.1
for k in range(0, no_iter):
	print("Iteration :",k)
	temp=pow(1-steps, k)
	s=np.random.normal(0,1,no_movies*no_feat)
	t_movie=np.reshape(s, (no_movies,no_feat))
	t=np.random.normal(0,1,no_users*no_feat)
	t_user=np.reshape(t, (no_users,no_feat))
	t_l1=np.random.normal(0,1,1)
	t_l2=np.random.normal(0,1,1)
	new_cost=cost(t_movie, t_user, t_l1, t_l2, norm_rating, is_rated, no_movies, no_users, no_feat)
	print ("New cost=",new_cost, " Old cost=", initial_cost)
	if new_cost < initial_cost or np.random.uniform(1,0,1) < np.exp(-(new_cost-initial_cost)/temp):
		initial_cost=new_cost
		movie_feat_vec=t_movie	 
		user_feat_vec=t_user
		l1=t_l1
		l2=t_l2
predictions = np.dot(movie_feat_vec, np.transpose(user_feat_vec))
for i in range(0, no_movies):
	predictions[i,:] = predictions[i,:] + mean_rating[i]
MSE = np.sum(np.square(np.multiply(matrix, is_rated) - np.multiply(predictions, is_rated)))/100000
print ("MSE=", MSE)
RMSE = math.sqrt(MSE)
print ("RMSE=", RMSE)