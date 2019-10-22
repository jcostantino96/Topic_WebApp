#!/usr/bin/env python
# coding: utf-8

# ## Sprint 1 / Process a text document and extract sentences to generate topics
# ## Sprint 2 / Identify all sentences and store them into a JSON Object
# #### Team 3
# #### Reyes Ceballos
# #### Joseph A. Costantino
# #### Mallikarjunreddy Anireddy
# APPLIED ANALYTICS PROJECT <br>
# CUS-690 <br>
# Prof. Giancarlo Crocetti

# #### Choose file you will use

# In[1]:


file = 'Med_QA_aspirin.txt'
# file =  'VISASPIR.txt'
#file = input("Select text file ")
#file = 'insurance.txt'


# #### Read file

# In[2]:


infile = open(file,'r')
text = infile.read()
infile.close()


# ## Clean Text

# In[3]:


import nltk


# In[4]:


sentences = [i.strip().lower().replace('\t','').replace('\n','') for i in nltk.sent_tokenize(text)]
# Remove whitespace
# Make all letters lowercase
# remove \t and \n
# extract sentences


# In[5]:


import pandas

df = pandas.DataFrame({'Original_Sent': sentences})


# In[6]:


#sentences
# review sentences to see if theres anything else to clean


# In[7]:


punctuations = ".,!?'()-/©–—‘’“”:;#$%&*[]_~|``"
# Remove this punctuation


# In[8]:


sentences[0]


# In[9]:


sent = []

for i in sentences:
    s= []
    for j in i:
        if j not in punctuations:
            # Don't add punctuations
            s.append(j)
        else:
            s.append(' ')
            # add a space instead of punctuation
    sent.append(''.join(s))

sent = [i.strip() for i in sent]
# Remove extra spaces at the end of sentences


# In[10]:


import re
sent = [re.sub(r'[^\x00-\x7F]','',s.lower()) for s in sent]
# eliminate all UTF characters


# In[11]:


#sent
# Review sent to see if more cleaning is necessary


# In[12]:


#{j for i in sent for j in i}
# all characters. analyze for additional punctuation removal


# #### Tokenize and remove Stopwords

# In[13]:


from nltk.corpus import stopwords
# nltk.download() # download first
stopword = stopwords.words('english')

#stopword # words to be removed


# In[14]:


filtered = []

for i in sent:
    filt = []
    for j in nltk.word_tokenize(i):
         if j not in stopword:
            filt.append(''.join(j))
    filtered.append(' '.join(filt))


# In[15]:


#{j for i in filtered for j in i.split()}
# Review to see any other words should be removed
# May not be realistic since we don't know what documents we will be analyzing


# #### Lemmatization

# In[16]:


from nltk.stem import WordNetLemmatizer


# In[17]:


lemma = []

for i in filtered:
    l = []
    for j in i.split():
        l.append(WordNetLemmatizer().lemmatize(j))
    lemma.append(' '.join(l))


# In[18]:


lemma[0]


# In[19]:


df['lemma'] = lemma


# In[20]:


df.head()


# ## Start Topic discovery process

# ### LDA Topic Modeling

# In[21]:


import gensim.corpora as corpora
import gensim
from gensim.models.coherencemodel import CoherenceModel


# Creating dictionary

# In[22]:


id2word = corpora.Dictionary([i.split() for i in lemma])


# Creating corpus

# In[23]:


texts = [i.split() for i in lemma]
# list of sentences with tokenized words
corpus = [id2word.doc2bow(text) for text in texts]


# In[24]:


topicK = [i for i in range(2,6)] # number of topics


# In[25]:


topicDic = {} #holds all topics and the top words
models = {} #all models to later pick the best one
ModelEvaluation = [] # Coherence Score
#Model perplexity and topic coherence provide a convenient measure to judge how good a given topic model is. 
#topic coherence score has been more helpful.
#https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/


# Building the model & Analyzing the Model

# In[26]:


for k in topicK:
    iterations = 100 # 20-30 good for testing, 50-100 better for final
    topic_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=k, random_state=1, id2word = id2word, passes = iterations)

    topics = topic_model.print_topics(num_words = 7) # top 5 words
    models['k'+str(k)] = topics # add model to dict

    for i in topics:
        topicWords = [] #all words
        for j in i[1].split(' + '):
            topicWords.append(j.split('*')[1][1:-1]) #words
        topicDic['k'+str(k)+'t'+str(i[0])] = topicWords
        # gives K and Topic Number
    print(k)
    
    coherence_model_lda = CoherenceModel(model=topic_model, texts=texts, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    
    ModelEvaluation.append([str(k), coherence_lda]) # a measure of how good the model is.
    # perplexity - a measure of how good the model is. lower the better.
    # coherence score of each topic then aggregated. the higher the better
    # https://rare-technologies.com/what-is-topic-coherence/


# In[27]:


#topicDic.keys() # k's
#topicDic # words


# Choosing a K

# In[28]:


ModelEvaluation


# In[29]:


pick = '' # optimal K
score = 0 #highest coherence score

for i in ModelEvaluation:
    if i[-1] > score:
        pick = i[0]
        score = i[1]
    else:
        break


# In[30]:


words = set()
# top words in each topic (no duplicates)

for i in topicDic.keys():
    if int(i[1]) == int(pick):
        [words.add(j) for j in topicDic[str(i)]]


# #### Topic Words

# In[31]:


print('The topics for this document are: \n')
for i in words:
    print(i)


# In[32]:


words = list(words)


# #### Create Dictionary
# #### Keys = FileName, Text, Sentences

# In[33]:


data = {}


# In[34]:


data["FileName"] = file
data['Title'] = sent[0]
data["Text"] = " ".join(sent)
data["Sentences"] = sent
data["Topics"] = words


# In[35]:


data.keys()


# In[36]:


data["FileName"]


# In[37]:


#data["Text"]
# large output


# In[38]:


#data["Sentences"]
# large output


# In[39]:


data["Topics"]


# #### Convert to JSON Object

# In[40]:


import json


# In[41]:


json_data = json.dumps(data)


# In[42]:


with open(file[:-4]+".json","w") as f:
    json.dump(data,f)
# Saves a json file


# #### Test json object

# In[43]:


J =  open(file[:-4]+".json","r")
# opens json file
J = json.load(J)
# converts json file to python dict


# In[44]:


J.keys()


# #### References

# https://thepythonguru.com/reading-and-writing-json-in-python/

# In[ ]:




