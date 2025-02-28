import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem.porter import *
import string
import os
from nltk import FreqDist
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import nltk.classify.util
import spacy
import en_core_web_lg
from scipy import spatial
from spacy.language import Language
import json
import matplotlib.pyplot as plt

import time

import time
import subprocess

def run_and_hold(command):
    process = subprocess.Popen(command, shell=True)
    process.wait()

if __name__ == "__main__":
    print("Starting process...")
    run_and_hold("import.py") # Replace your_script.py with the actual script
    print("Process finished.")
    
# lightweight English language processing model optimized for CPU usage, provides comprehensive NLP capabilities while maintiant a small footprint 
nlp = en_core_web_lg.load()
stemmer = PorterStemmer()


###Making directory path
corpus_dir = 'Capstone'
if not os.path.exists(corpus_dir):
    os.makedirs(corpus_dir)


#Pulling in text into variable
texts = [
    (corpus_dir, '.*.txt')
]
corpus = PlaintextCorpusReader(corpus_dir, '.*.txt')

j=int(0)

fileid=corpus.fileids()
parastring=""

for id in fileid:
    file=fileid[j]
    #Variables to prepare document
    stop_words = list(stopwords.words('english'))
    stop_words.extend(['school', 'boise', 'get', 'teacher', 'kid','student','use','sarah'])

    words = corpus.words(fileids=[file])
    sents = corpus.sents()
    paras = corpus.paras()

    #Make object, pull in text and make all text lower case 
    normalizedword=[]
    organize=[]

    for word in words: 
        
        lowword=word.lower()
        tonum= re.sub(r'\d+', '', lowword)
        #Do not include in analysis if stop word, otherwise tokenize and append to object
        if tonum not in stop_words:
            wordy=word_tokenize(tonum)
            normalizedword.append(wordy)   
                                #Extract innerlist from object
    for inner_list in normalizedword:
        organize.extend(inner_list)
                                    #Remove punctuation and stem words
        cleaned_tokens = [re.sub(r'[^\w\s]', '', token) for token in organize if re.sub(r'[^\w\s]', '', token)]
        stemmed_tokens = [stemmer.stem(word) for word in cleaned_tokens]     
        # Find the frequency distribution in the given text
    frequency_distribution = FreqDist(stemmed_tokens)
    topten=(frequency_distribution.most_common(20))

    parastring1=" ".join(map(str,frequency_distribution))
    parastring2=" ".join(map(str,topten))
    j=j+1                                                           ##Printing Results
    print      
    name='NLP Analysis Document '+str(j)+'.txt'
    f=open(name,"w") # The with keyword automatically closes the file when you are done
    f.write("Keywords:\n" + parastring1 +"\n \n")
    f.write("***Top 20 Keywords***: \n" + parastring2+"\n \n \n")
    f.close()
                                        #Stringify for sentiment analysis and print analysis 
    sid = SentimentIntensityAnalyzer()
    parastring=" ".join(map(str,words))
    sentiment_scores = sid.polarity_scores(parastring)
    holder=json.dumps(sentiment_scores)
    
    f=open(name,"a")
    f.write("Sentiment Analysis \n Negative,     Neutral,     Positive,      Compound \n----------------------------------------------------------\n")
    f.write(holder)
    f.close()


    #-----------------------------------------------------------------------------
    #Similarity        #Word2Vec
    cosine_similarity = lambda x,y:1 - spatial.distance.cosine(x,y)
    mental = nlp.vocab['mental'].vector
    health = nlp.vocab['health'].vector
    classr = nlp.vocab['class'].vector
    rank = nlp.vocab['rank'].vector
    new_vector=mental-health+classr
    computed_similarities = []
    for word in nlp.vocab:
        if word.has_vector:
            if word.is_lower:
                if word.is_alpha:
                    similarity = cosine_similarity(new_vector, word.vector)
                    computed_similarities.append((word,similarity))
    computed_similarities = sorted(computed_similarities, key=lambda item:-item[1])

    similarity = cosine_similarity(new_vector,rank)
    str_similarity=str(similarity)
    w0text=([w[0].text for w in computed_similarities[:10]])
    w1text=([w[1] for w in computed_similarities[:10]])
    strw0=", ".join(str(x) for x in w0text)
    strw1=", ".join(str(x) for x in w1text)
    f=open(name,"a")
    f.write("\n \n \n Cosine similarity of relationship mental+health-class to *rank*\n----------------------------------------------------------------------------\n "+str_similarity+"\n \n")

  #======================================================================================================

    file1=fileid[1]
    file0=fileid[0]

   
    onereview=  " ".join(map(str,corpus.words(fileids=[file1])))
    zeroreview = " ".join(map(str,corpus.words(fileids=[file0])))
    review1=nlp(onereview)     
    review0=nlp(zeroreview)
    token1=nlp('rank')  
    reviews=[review1,review0]  
      
    compare= str(review1.similarity(review0))           
    f=open(name,"a")
    f.write("\n \n \n Similarity between texts\n----------------------------------------------------------------------------\n "+compare+"\n \n")
 
    f.close()
      
print ("successfully compiled")
