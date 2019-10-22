{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Sprint 1 / Process a text document and extract sentences to generate topics\n",
    "## Sprint 2 / Identify all sentences and store them into a JSON Object\n",
    "#### Team 3\n",
    "#### Reyes Ceballos\n",
    "#### Joseph A. Costantino\n",
    "#### Mallikarjunreddy Anireddy\n",
    "APPLIED ANALYTICS PROJECT <br>\n",
    "CUS-690 <br>\n",
    "Prof. Giancarlo Crocetti"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Choose file you will use"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "file = 'Med_QA_aspirin.txt'\n",
    "# file =  'VISASPIR.txt'\n",
    "#file = input(\"Select text file \")\n",
    "#file = 'insurance.txt'"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Read file"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "infile = open(file,'r')\n",
    "text = infile.read()\n",
    "infile.close()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Clean Text"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "import nltk"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "sentences = [i.strip().lower().replace('\\t','').replace('\\n','') for i in nltk.sent_tokenize(text)]\n",
    "# Remove whitespace\n",
    "# Make all letters lowercase\n",
    "# remove \\t and \\n\n",
    "# extract sentences"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas\n",
    "\n",
    "df = pandas.DataFrame({'Original_Sent': sentences})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "#sentences\n",
    "# review sentences to see if theres anything else to clean"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "punctuations = \".,!?'()-/©–—‘’“”:;#$%&*[]_~|``\"\n",
    "# Remove this punctuation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'q&a 285.1is there evidence to support the use of enteric coated (ec) aspirin to reduce gastrointestinal side effects in cardiovascular patients?'"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sentences[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "sent = []\n",
    "\n",
    "for i in sentences:\n",
    "    s= []\n",
    "    for j in i:\n",
    "        if j not in punctuations:\n",
    "            # Don't add punctuations\n",
    "            s.append(j)\n",
    "        else:\n",
    "            s.append(' ')\n",
    "            # add a space instead of punctuation\n",
    "    sent.append(''.join(s))\n",
    "\n",
    "sent = [i.strip() for i in sent]\n",
    "# Remove extra spaces at the end of sentences"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "import re\n",
    "sent = [re.sub(r'[^\\x00-\\x7F]','',s.lower()) for s in sent]\n",
    "# eliminate all UTF characters"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "#sent\n",
    "# Review sent to see if more cleaning is necessary"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "#{j for i in sent for j in i}\n",
    "# all characters. analyze for additional punctuation removal"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Tokenize and remove Stopwords"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "from nltk.corpus import stopwords\n",
    "# nltk.download() # download first\n",
    "stopword = stopwords.words('english')\n",
    "\n",
    "#stopword # words to be removed"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "filtered = []\n",
    "\n",
    "for i in sent:\n",
    "    filt = []\n",
    "    for j in nltk.word_tokenize(i):\n",
    "         if j not in stopword:\n",
    "            filt.append(''.join(j))\n",
    "    filtered.append(' '.join(filt))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "#{j for i in filtered for j in i.split()}\n",
    "# Review to see any other words should be removed\n",
    "# May not be realistic since we don't know what documents we will be analyzing"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Lemmatization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "from nltk.stem import WordNetLemmatizer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "lemma = []\n",
    "\n",
    "for i in filtered:\n",
    "    l = []\n",
    "    for j in i.split():\n",
    "        l.append(WordNetLemmatizer().lemmatize(j))\n",
    "    lemma.append(' '.join(l))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'q 285 1is evidence support use enteric coated ec aspirin reduce gastrointestinal side effect cardiovascular patient'"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "lemma[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "df['lemma'] = lemma"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Original_Sent</th>\n",
       "      <th>lemma</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>q&amp;a 285.1is there evidence to support the use ...</td>\n",
       "      <td>q 285 1is evidence support use enteric coated ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>prepared by uk medicines information (ukmi) ph...</td>\n",
       "      <td>prepared uk medicine information ukmi pharmaci...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>aspirin irreversibly inhibits cyclo-oxygenase,...</td>\n",
       "      <td>aspirin irreversibly inhibits cyclo oxygenase ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>in platelets, this will reduce the formation o...</td>\n",
       "      <td>platelet reduce formation thromboxane a2 vasoc...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>it also reduces the formation of prostacyclin ...</td>\n",
       "      <td>also reduces formation prostacyclin vascular e...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                       Original_Sent  \\\n",
       "0  q&a 285.1is there evidence to support the use ...   \n",
       "1  prepared by uk medicines information (ukmi) ph...   \n",
       "2  aspirin irreversibly inhibits cyclo-oxygenase,...   \n",
       "3  in platelets, this will reduce the formation o...   \n",
       "4  it also reduces the formation of prostacyclin ...   \n",
       "\n",
       "                                               lemma  \n",
       "0  q 285 1is evidence support use enteric coated ...  \n",
       "1  prepared uk medicine information ukmi pharmaci...  \n",
       "2  aspirin irreversibly inhibits cyclo oxygenase ...  \n",
       "3  platelet reduce formation thromboxane a2 vasoc...  \n",
       "4  also reduces formation prostacyclin vascular e...  "
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Start Topic discovery process"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### LDA Topic Modeling"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Anaconda\\lib\\site-packages\\gensim\\utils.py:1197: UserWarning: detected Windows; aliasing chunkize to chunkize_serial\n",
      "  warnings.warn(\"detected Windows; aliasing chunkize to chunkize_serial\")\n"
     ]
    }
   ],
   "source": [
    "import gensim.corpora as corpora\n",
    "import gensim\n",
    "from gensim.models.coherencemodel import CoherenceModel"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Creating dictionary"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "id2word = corpora.Dictionary([i.split() for i in lemma])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Creating corpus"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "texts = [i.split() for i in lemma]\n",
    "# list of sentences with tokenized words\n",
    "corpus = [id2word.doc2bow(text) for text in texts]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [],
   "source": [
    "topicK = [i for i in range(2,6)] # number of topics"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "topicDic = {} #holds all topics and the top words\n",
    "models = {} #all models to later pick the best one\n",
    "ModelEvaluation = [] # Coherence Score\n",
    "#Model perplexity and topic coherence provide a convenient measure to judge how good a given topic model is. \n",
    "#topic coherence score has been more helpful.\n",
    "#https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Building the model & Analyzing the Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "2\n",
      "3\n",
      "4\n",
      "5\n"
     ]
    }
   ],
   "source": [
    "for k in topicK:\n",
    "    iterations = 100 # 20-30 good for testing, 50-100 better for final\n",
    "    topic_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=k, random_state=1, id2word = id2word, passes = iterations)\n",
    "\n",
    "    topics = topic_model.print_topics(num_words = 7) # top 5 words\n",
    "    models['k'+str(k)] = topics # add model to dict\n",
    "\n",
    "    for i in topics:\n",
    "        topicWords = [] #all words\n",
    "        for j in i[1].split(' + '):\n",
    "            topicWords.append(j.split('*')[1][1:-1]) #words\n",
    "        topicDic['k'+str(k)+'t'+str(i[0])] = topicWords\n",
    "        # gives K and Topic Number\n",
    "    print(k)\n",
    "    \n",
    "    coherence_model_lda = CoherenceModel(model=topic_model, texts=texts, dictionary=id2word, coherence='c_v')\n",
    "    coherence_lda = coherence_model_lda.get_coherence()\n",
    "    \n",
    "    ModelEvaluation.append([str(k), coherence_lda]) # a measure of how good the model is.\n",
    "    # perplexity - a measure of how good the model is. lower the better.\n",
    "    # coherence score of each topic then aggregated. the higher the better\n",
    "    # https://rare-technologies.com/what-is-topic-coherence/\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [],
   "source": [
    "#topicDic.keys() # k's\n",
    "#topicDic # words"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Choosing a K"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[['2', 0.3548039197675783],\n",
       " ['3', 0.4328877615990218],\n",
       " ['4', 0.38508323020932134],\n",
       " ['5', 0.4564743569005283]]"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ModelEvaluation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [],
   "source": [
    "pick = '' # optimal K\n",
    "score = 0 #highest coherence score\n",
    "\n",
    "for i in ModelEvaluation:\n",
    "    if i[-1] > score:\n",
    "        pick = i[0]\n",
    "        score = i[1]\n",
    "    else:\n",
    "        break"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [],
   "source": [
    "words = set()\n",
    "# top words in each topic (no duplicates)\n",
    "\n",
    "for i in topicDic.keys():\n",
    "    if int(i[1]) == int(pick):\n",
    "        [words.add(j) for j in topicDic[str(i)]]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Topic Words"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The topics for this document are: \n",
      "\n",
      "platelet\n",
      "et\n",
      "actual\n",
      "ec\n",
      "effect\n",
      "outcome\n",
      "www\n",
      "via\n",
      "uk\n",
      "online\n",
      "day\n",
      "study\n",
      "clinical\n",
      "coated\n",
      "accessed\n",
      "participant\n",
      "al\n",
      "enteric\n",
      "indication\n",
      "aspirin\n"
     ]
    }
   ],
   "source": [
    "print('The topics for this document are: \\n')\n",
    "for i in words:\n",
    "    print(i)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [],
   "source": [
    "words = list(words)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Create Dictionary\n",
    "#### Keys = FileName, Text, Sentences"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [],
   "source": [
    "data = {}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [],
   "source": [
    "data[\"FileName\"] = file\n",
    "data['Title'] = sent[0]\n",
    "data[\"Text\"] = \" \".join(sent)\n",
    "data[\"Sentences\"] = sent\n",
    "data[\"Topics\"] = words"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "dict_keys(['FileName', 'Title', 'Text', 'Sentences', 'Topics'])"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.keys()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'Med_QA_aspirin.txt'"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data[\"FileName\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [],
   "source": [
    "#data[\"Text\"]\n",
    "# large output"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [],
   "source": [
    "#data[\"Sentences\"]\n",
    "# large output"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['platelet',\n",
       " 'et',\n",
       " 'actual',\n",
       " 'ec',\n",
       " 'effect',\n",
       " 'outcome',\n",
       " 'www',\n",
       " 'via',\n",
       " 'uk',\n",
       " 'online',\n",
       " 'day',\n",
       " 'study',\n",
       " 'clinical',\n",
       " 'coated',\n",
       " 'accessed',\n",
       " 'participant',\n",
       " 'al',\n",
       " 'enteric',\n",
       " 'indication',\n",
       " 'aspirin']"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data[\"Topics\"]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Convert to JSON Object"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [],
   "source": [
    "import json"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [],
   "source": [
    "json_data = json.dumps(data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [],
   "source": [
    "with open(file[:-4]+\".json\",\"w\") as f:\n",
    "    json.dump(data,f)\n",
    "# Saves a json file"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Test json object"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [],
   "source": [
    "J =  open(file[:-4]+\".json\",\"r\")\n",
    "# opens json file\n",
    "J = json.load(J)\n",
    "# converts json file to python dict"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "dict_keys(['FileName', 'Title', 'Text', 'Sentences', 'Topics'])"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "J.keys()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### References"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "https://thepythonguru.com/reading-and-writing-json-in-python/"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
