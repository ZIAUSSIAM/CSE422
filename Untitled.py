#!/usr/bin/env python
# coding: utf-8

# In[53]:


import pandas as pd
import numpy as np
import matplotlib as mat

#read the data from the csv
df = pd.read_csv('../../Desktop/top50.csv', encoding='ISO-8859-1')

df.head(10)


# In[54]:


#the data rows and colomns
df.shape


# In[55]:


#describtion of each atribute
df.describe()


# In[56]:


df.values


# In[58]:


#checked if there is any null data 
df.isnull().sum()


# In[59]:


#created a array consisting of values in genre atribute
g = df['Genre']

df


# In[60]:


# all the unqie catagories of genre attribute
g.unique()


# In[61]:


from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()

# Apply the encoding to the "Accessible" column
df['Genre_enc'] = enc.fit_transform(g)

#number of unique attribute
g.unique().shape


# In[62]:


# Compare the two columns
df[['Genre', 'Genre_enc']]


# In[63]:


#show the new df consisting of encoded variable
df


# In[65]:


#new rows and colomns of the data frame
df.shape


# In[66]:


#Feature selection
import seaborn as sns

#Correlation matrix
music_corr = df.corr()
music_corr


# In[67]:


sns.heatmap(music_corr, cmap = 'YlGnBu')


# In[69]:


#dropped unrequired atributes
df = df.drop(['Artist.Name', 'Loudness..dB..','Valence.','Acousticness..','Unnamed: 0','Danceability','Liveness'], axis = 1)
#new Data Frame
df


# In[70]:


df.shape


# In[ ]:




