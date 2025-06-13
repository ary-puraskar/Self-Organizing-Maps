#Self Organizing Map

#Importing the necessary libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the dataset
df=pd.read_csv("Credit_Card_Applications.csv")

#Creating two sets

X=df.iloc[:,:-1].values
Y=df.iloc[:,-1].values

#Feature Scaling

from sklearn.preprocessing import MinMaxScaler
Mms= MinMaxScaler(feature_range=(0,1))
X=Mms.fit_transform(X)

#Training the SOM
from minisom import MiniSom #Importing Minisom class from the minisom python library
som=MiniSom(x=10, y=10, input_len=15, sigma=1.0,learning_rate=0.5) #Creating object, sigma is the neighborhood radius
som.random_weights_init(X) #Weight initialization
som.train_random(data=X, num_iteration=100)

#Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show 
bone()
pcolor(som.distance_map().T) #T means transpose
colorbar()
markers=['o','s']
colors= ['r','g']
for i,x in enumerate(X): # i means index of customers and x be the different vectors of the customer
    w=som.winner(x)
    plot(w[0]+0.5,
         w[1]+0.5,
         markers[Y[i]], 
         markeredgecolor=colors[Y[i]], 
         markerfacecolor='None', 
         markersize=10, 
         markeredgewidth=2)
show()

#Finding the frauds

mappings= som.win_map(X)
frauds= np.concatenate((mappings[(1,3)],mappings[(3,5)]), axis=0)
frauds=Mms.inverse_transform(frauds)