#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


videos = pd.read_csv(r"C:\Users\soumy\OneDrive\Desktop\pyprog\Data Analytics\1-Youtube Text Data Analysis\UScomments.csv", error_bad_lines=False)
videos.head()


# In[3]:


get_ipython().system('pip install textblob')


# In[4]:


from textblob import TextBlob


# In[5]:


TextBlob("Logan Paul it's yo big day ‼️‼️‼️").sentiment.polarity


# In[6]:


polarity = []
for i in videos["comment_text"]:
    try:
        polarity.append(TextBlob(i).sentiment.polarity)
    except:
        polarity.append(0)


# In[7]:


videos['polarity'] = polarity


# In[8]:


videos.head(20)


# In[9]:


comments_positive = videos[videos["polarity"] == 1]
comments_positive.head(10)


# In[10]:


get_ipython().system('pip install wordcloud')


# In[11]:


from wordcloud import WordCloud, STOPWORDS


# In[12]:


total_comments = (" ".join(comments_positive["comment_text"]))
len(total_comments)


# In[13]:


wordcloud = WordCloud(width=1000 , height=500, stopwords=set(STOPWORDS)).generate(total_comments)
plt.figure(figsize=(15,5))
plt.imshow(wordcloud)
plt.axis('off')


# In[14]:


comments_neg = videos[videos["polarity"] == -1]
comments_neg.head(10)


# In[15]:


total_comms = (" ".join(comments_neg["comment_text"]))
len(total_comms)


# In[16]:


negwordcloud = WordCloud(width=1000 , height=500, stopwords=set(STOPWORDS)).generate(total_comms)
plt.figure(figsize=(15,5))
plt.imshow(negwordcloud)
plt.axis('off')


# In[17]:


#Next Part Begins
#____________________________________________________________________


# In[18]:


videos2 = pd.read_csv(r"C:\Users\soumy\OneDrive\Desktop\pyprog\Data Analytics\1-Youtube Text Data Analysis\USvideos.csv", error_bad_lines=False)
videos2.head()


# In[19]:


tags_complete = (" ".join(videos2["tags"]))


# In[20]:


videos2["tags"][0]


# In[21]:


import re


# In[22]:


tags = re.sub('[^a-zA-Z]',' ', tags_complete)


# In[23]:


tags


# In[24]:


tags = re.sub(' +', ' ', tags)


# In[25]:


tags


# In[26]:


tagswordcloud = WordCloud(width=1000 , height=500, stopwords=set(STOPWORDS)).generate(tags)
plt.figure(figsize=(15,5))
plt.imshow(tagswordcloud)
plt.axis('off')


# In[27]:


sns.regplot(data=videos2, x='views', y='likes')
plt.title("Regression plot for vies vs likes")


# In[28]:


sns.regplot(data=videos2, x='views', y='dislikes')
plt.title("Regression plot for vies vs dislikes")


# In[29]:


df_corr = videos2[['views','likes','dislikes']]
sns.heatmap(df_corr.corr(),annot=True)


# In[30]:


#Next Part Begins 
#________________________________________________________


# In[31]:


get_ipython().system('pip install emoji')


# In[40]:


videos.dropna(axis=0
,subset=['comment_text'], inplace=True)
videos["comment_text"].isna().sum()


# In[41]:


len(videos["comment_text"])


# In[42]:


videos["comment_text"][4]
import emoji


# In[43]:


str=''
for i in videos["comment_text"]:
    list = [c for c in i if c in emoji.UNICODE_EMOJI]
    for ele in list:
        str += ele


# In[44]:


print(str)


# In[45]:


len(set(str))


# In[46]:


res = {i:str.count(i) for i in set(str)}
print(res)


# In[53]:


res = {k:v for k,v in sorted(res.items(), key=lambda item:item[1])}
res


# In[54]:


keys = [*res.keys()]
values = [*res.values()]


# In[56]:


df = pd.DataFrame({'chars':keys[-20:], 'num':values[-20:]})


# In[59]:


get_ipython().system('pip install plotly')


# In[60]:


import plotly.graph_objs as go
from plotly.offline import iplot


# In[61]:


trace = go.Bar(x=df['chars'], y=df['num'])
iplot([trace])


# In[ ]:




