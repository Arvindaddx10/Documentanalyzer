#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Opening a file
doc=open("F:\\old\\Addx\\ADDX\\twitter\\doc123.txt")
#read the document and print its content
doctxt=doc.read()
print(doctxt)

#Normalise the text
from string import punctuation

#remove numeric digits
txt=''.join(c for c in doctxt if not c.isdigit())

#remove puncutation and make lower case
txt=''.join(c for c in txt if c not in punctuation).lower()

#print the normalized form
print(txt)

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


# plot the Distribution as a chart
get_ipython().run_line_magic('matplotlib', 'inline')


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


# remove stop words
nltk.download('stopwords')
from nltk.corpus import stopwords

#filter out the stop words
txt =' '.join([word for word in txt.split() if word not in (stopwords.words('english'))])


#get the frequency distribution of remaining words
words=nltk.tokenize.word_tokenize(txt)
fdist=FreqDist(words)
count_frame=pd.DataFrame(fdist,index=[0]).T
count_frame.columns=["count"]

#plot the frequency of top 60 words
counts= count_frame.sort_values('count',ascending=False)
fig=plt.figure(figsize=(16,9))
ax=fig.gca()
counts['count'][:60].plot(kind='bar',ax=ax)
ax.set_title("Frequency of the most common word")
ax.set_ylabel("Frequency of word")
ax.set_xlabel("word")
plt.show()


print(doctxt)
print("------------------------------------------------------")

#get a second document , normalize it and remove stop words
#Opening a file
doc2=open("F:\\old\\Addx\\ADDX\\twitter\\doc1234.txt")
#read the document and print its content
doc2txt=doc2.read()
print(doc2txt)
from string import punctuation
txt2=''.join(c for c in doc2txt if not c.isdigit())

#remove puncutation and make lower case
txt2=''.join(c for c in txt2 if c not in punctuation).lower()
txt2 =' '.join([word for word in txt2.split() if word not in (stopwords.words('english'))])

# and a third
print("--------------------------------------------------------")
#get a second document , normalize it and remove stop words
#Opening a file
doc3=open("F:\\old\\Addx\\ADDX\\twitter\\doc12345.txt")
#read the document and print its content
doc3txt=doc3.read()
print(doc3txt)
from string import punctuation
txt3=''.join(c for c in doc3txt if not c.isdigit())

#remove puncutation and make lower case
txt3=''.join(c for c in txt3 if c not in punctuation).lower()
txt3 =' '.join([word for word in txt3.split() if word not in (stopwords.words('english'))])


# install textblob and define functions for TF-IDF
get_ipython().system('pip install -U textblob')
import math
from textblob import TextBlob as tb

def tf(word,doc):
    return doc.words.count(word)/len(doc.words)
def contains(word,docs):
    return sum(1 for doc in docs if word in doc.words)
def idf(word,docs):
    return math.log(len(docs)/(1+contains(word,docs)))
def tfidf(word,doc,docs):
    return tf(word,doc)+idf(word,docs)

#create a collection of documents as textblob
doc1=tb(txt)
doc2=tb(txt2)
doc3=tb(txt3)
docs=[doc1,doc2,doc3]

#Use TF-IDF to get the three most important words from each document
print("---------------------------------------------------------------")
for i,doc in enumerate(docs):
    print("Top words in document{}".format(i+1))
    scores={word:tfidf(word,doc,docs) for word in doc.words}
    sorted_words= sorted(scores.items(), key=lambda x:x[1],reverse=True)
    for word,score in sorted_words[:3]:
        print("\tWord:{},TF-IDF:{}".format(word,round(score,5)))
        
    




# In[ ]:





# In[ ]:




