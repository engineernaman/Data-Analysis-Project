#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
from textblob import TextBlob
import seaborn as sns
import pandas as pd
import numpy as np
from wordcloud import WordCloud, STOPWORDS
import plotly.graph_objs as go
from plotly.offline import iplot
import re


# In[2]:


file1 = pd.read_csv(r"C:\Users\soumy\OneDrive\Desktop\pyprog\Data Analytics\1-Youtube Text Data Analysis\GBcomments.csv", error_bad_lines=False)
file1.head()


# In[3]:


polarity = []
for i in file1["comment_text"]:
    try:
        polarity.append(TextBlob(i).sentiment.polarity)
    except:
        polarity.append(0)
        
file1["polarity"]=polarity
file1.head(10)


# In[4]:


pos_cmt = file1[file1["polarity"] == 1]
pos_cmt.head()


# In[6]:


cmt_str = (" ".join(pos_cmt["comment_text"]))
len(cmt_str)


# In[7]:


pos_cloud = WordCloud(width=1000, height=500, stopwords=set(STOPWORDS)).generate(cmt_str)
plt.figure(figsize = (15,5))
plt.axis("off")
plt.imshow(pos_cloud)


# In[8]:


neg_cmt = file1[file1["polarity"] == -1]
neg_cmt.head()


# In[9]:


neg_cmt_str = (" ".join(neg_cmt["comment_text"]))
len(neg_cmt_str)


# In[10]:


neg_cloud = WordCloud(width=1000, height=500, stopwords=set(STOPWORDS)).generate(neg_cmt_str)
plt.figure(figsize = (15,5))
plt.axis("off")
plt.imshow(neg_cloud)


# In[11]:


#Second File Analysis
#__________________________________


# In[12]:


file2 = pd.read_csv(r"C:\Users\soumy\OneDrive\Desktop\pyprog\Data Analytics\1-Youtube Text Data Analysis\GBvideos.csv", error_bad_lines=False)
file2.head()


# In[13]:


tags_str = (" ".join(file2["tags"]))


# In[14]:


tags = re.sub('[^a-zA-z]', ' ', tags_str)
tags = re.sub(' +', ' ', tags)


# In[16]:


tags_cloud = WordCloud(width = 1000, height = 500, stopwords=set(STOPWORDS)).generate(tags)
plt.figure(figsize = (15,5))
plt.axis('off')
plt.imshow(tags_cloud)


# In[17]:


sns.regplot(data=file2, x='views', y='likes')
plt.title("Regression plot for vies vs likes")


# In[18]:


sns.regplot(data=file2, x='views', y='dislikes')
plt.title("Regression plot for vies vs dislikes")


# In[19]:


df_corr = file2[['views','likes','dislikes']]
sns.heatmap(df_corr.corr(),annot=True)


# In[20]:


#Third Part Emoji Analysis
#___________________________________________________________


# In[21]:


import emoji


# In[23]:


file1.dropna(axis = 0, subset = ["comment_text"], inplace = True)
file1["comment_text"].isna().sum()


# In[24]:


str=''
for i in file1["comment_text"]:
    list = [c for c in i if c in emoji.UNICODE_EMOJI]
    for ele in list:
        str += ele


# In[25]:


res_emoji = {i:str.count(i) for i in set(str)}
res_emoji


# In[26]:


res_emoji = {k:v for k,v in sorted(res_emoji.items(), key=lambda item:item[1])}
res_emoji


# In[28]:


keys = [*res_emoji.keys()]
values = [*res_emoji.values()]


# In[29]:


df = pd.DataFrame({'chars':keys[-20:], 'num':values[-20:]})


# In[30]:


trace = go.Bar(x=df['chars'], y=df['num'])
iplot([trace])


# In[32]:


#Completed


# In[ ]:




