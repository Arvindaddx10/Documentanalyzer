#!/usr/bin/env python
# coding: utf-8

# In[72]:


#Opening a file
doc=open("F:\\old\\Addx\\ADDX\\twitter\\doc123.txt")
#read the document and print its content
doctxt=doc.read()
print(doctxt)


# In[73]:


#Normalise the text
from string import punctuation

#remove numeric digits
txt=''.join(c for c in doctxt if not c.isdigit())

#remove puncutation and make lower case
txt=''.join(c for c in txt if c not in punctuation).lower()

#print the normalized form
print(txt)


# In[75]:


#Get the frequency distribution
import nltk
import pandas as pd
from nltk.probability import FreqDist
nltk.download("punkt")

# Tokenize the text into indivifual words
words=nltk.tokenize.word_tokenize(txt)

# get the frequency distribution of the word into a Data frame
fdist=FreqDist(words)
count_frame=pd.DataFrame(fdist,index=[0]).T
count_frame.columns=["count"]
print(count_frame)


# In[76]:


# plot the Distribution as a chart
get_ipython().run_line_magic('matplotlib', 'inline')


# In[77]:


import matplotlib.pyplot as plt

# Sort the Data frame by frequency
counts= count_frame.sort_values('count',ascending=False)

#Display the top 60 Words as a bar plot
fig=plt.figure(figsize=(16,9))
ax=fig.gca()
counts['count'][:60].plot(kind='bar',ax=ax)
ax.set_title("Frequency of the most common word")
ax.set_ylabel("Frequency of word")
ax.set_xlabel("word")
plt.show()


# In[78]:


# remove stop words
nltk.download('stopwords')
from nltk.corpus import stopwords

#filter out the stop words
txt =' '.join([word for word in txt.split() if word not in (stopwords.words('english'))])
print(txt)

#get the frequency distribution of remaining words
words=nltk.tokenize.word_tokenize(txt)
fdist=FreqDist(words)
count_frame=pd.DataFrame(fdist,index=[0]).T
count_frame.columns=["count"]
print(count_frame)

#plot the frequency of top 60 words
counts= count_frame.sort_values('count',ascending=False)
fig=plt.figure(figsize=(16,9))
ax=fig.gca()
counts['count'][:60].plot(kind='bar',ax=ax)
ax.set_title("Frequency of the most common word")
ax.set_ylabel("Frequency of word")
ax.set_xlabel("word")
plt.show()


# In[ ]:




