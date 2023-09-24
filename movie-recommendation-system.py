#!/usr/bin/env python
# coding: utf-8

# # Knowledge based Movie Recomendation System

# In[1]:


get_ipython().system('python --version')


# In[2]:


import numpy as np 
import pandas as pd

import warnings
warnings.filterwarnings('ignore') 


# In[3]:


data = pd.read_csv("movies.csv")
data


# ## Analysing Data

# In[4]:


print(data.shape)


# In[5]:


data.info()


# In[6]:


data.isnull().sum()


# In[7]:


data.duplicated().sum()


# ## Cleaning Data

# In[8]:


# dropping unnecesary title 
df = data.drop(["production_companies", "popularity", "budget", "revenue", "status", "recommendations", "runtime", "vote_average", "backdrop_path", "tagline"], axis=1)


# In[9]:


df.drop_duplicates(inplace=True)


# In[10]:


df.title.duplicated().sum()


# In[11]:


df[["title", "release_date"]].duplicated().sum()


# In[12]:


df.drop_duplicates(subset=["title","release_date"], inplace=True)


# In[13]:


df = df[df.vote_count >= 350].reset_index()


# In[14]:


df.isnull().sum()


# In[15]:


df.fillna("", inplace = True)


# In[16]:


index = df[(df.genres == "") & (df.overview == "")].index
df.drop(index, inplace=True)


# In[17]:


df.genres = df.genres.apply(lambda x: " ".join(x.split("-")))
df.keywords = df.keywords.apply(lambda x: " ".join(x.split("-")))
df.credits = df.credits.apply(lambda x: " ".join(x.replace(" ", "").split("-")[:5]))


# In[18]:


# making tags consisting of overview, genre, credits, keywords and original language
df["tags"] =df.overview + " "+ df.genres + " "  +df.credits + " " +df.keywords + " " + df.original_language


# In[19]:


# making new framework with important features
new_df = df[["id", "title", "tags", 'poster_path']]


# In[20]:


new_df.tags = new_df.tags.apply(lambda x:x.lower())


# In[21]:


new_df.head()


# ### Stremming
# 
# Stemming is the process of reducing words to their base or root form.

# In[22]:


import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[23]:


def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    
    return " ".join(y)    


# In[24]:


new_df["tags"] = new_df["tags"].apply(stem)


# ### Text Vectorization
# 
# Text vectorization is the process of converting text data into numerical vectors so that they can be used as input for machine learning models.

# In[25]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(stop_words="english",max_features=2000)


# In[26]:


vectors = cv.fit_transform(new_df["tags"]).toarray()


# In[27]:


cv.get_feature_names()[80:85]


# ## ML Modeling
# 
# Cosine Similarity is a process of creating a machine learning model that utilizes the cosine similarity metric to determine the similarity between two pieces of text.

# In[28]:


from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)


# In[29]:


similarity.shape


# ## Testing

# In[30]:


def recommend(movies):
    movie_index = new_df[new_df.title == movies].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key= lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(new_df.iloc[i[0]].title)


# In[31]:


recommend("Toy Story")


# In[32]:


recommend("The Conjuring")


# __Observation__ : The function is returning 5 similar movies.
