#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
# from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings


# In[3]:


#prepare the data
df = pd.read_csv('CAE (1).csv')
df.head()


# In[4]:


df.set_index('Journal Name', inplace = True)
df.head()


# In[5]:


#count the cosine similarity
vector = CountVectorizer()
count_matrix = vector.fit_transform(df['Journal Absract'].values.astype('U'))
# print(count_matrix)
#measure the cosine similarity
cosine_simi = cosine_similarity(count_matrix,count_matrix)
# cosine_simi
# creating a Series for the movie titles so they are associated to an ordered numerical
# list I will use later to match the indexes
indices = pd.Series(df.index)
indices[:5]


# In[7]:


#now define the recommended function
def recommended(Journal_Name , cosine_simi=cosine_simi):
    recommended_journel = []
    # gettin the index of the journel that matches the journel name
    idx = indices[indices == Journal_Name].index[0]
    # creating a Series with the similarity scores in descending order
    score_series = pd.Series(cosine_simi[idx]).sort_values(ascending = False)
    # getting the indexes of the 10 most similar movies
    top_10_indexes = list(score_series.iloc[1:11].index)
    # populating the list with the titles of the best 10 matching movies
    for i in top_10_indexes:
        recommended_journel.append(list(df.index)[i])
        
    return recommended_journel


# In[9]:


recommended('ACM COMPUTING SURVEYS')


# In[ ]:




