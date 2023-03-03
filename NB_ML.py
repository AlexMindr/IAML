import nltk
import pandas as pd
import numpy as np
from collections import Counter
import re
import random
from sklearn.naive_bayes import BernoulliNB
import time
from sklearn.metrics import fbeta_score

def scoate_url(text):
    '''Scoate url din text.
    '''
    return re.sub(r"http\S+", "", text)

def scoate_punctuatie(text):
    '''Scoate punctuatie din text.
    '''
    return re.sub(r'[^\w\s]','',text)

def scoate_cuv1si2cif(text):
    '''Scoate cuv de 1 si 2 cifre din text.
    '''
    return re.sub(r'\b\w{1,2}\b', '', text)

def scoate_cifre(text):
    '''Scoate cifre si numere din text.
    '''
    return re.sub(r"\d+", "", text)

def tokenize(text):
    '''Generic wrapper around different tokenization methods.
    '''
    text=scoate_url(text)
    text =scoate_punctuatie(text)
    text =scoate_cuv1si2cif(text)
    text =scoate_cifre(text)
    return nltk.tokenize.casual.TweetTokenizer(strip_handles=True, reduce_len=True,preserve_case=False).tokenize(text)

def get_representation(toate_cuvintele, how_many):
    '''Extract the first most common words from a vocabulary
    and return two dictionaries: word to index and index to word
           @  che  .   ,   di  e
    text0  0   1   0   2   0   1
    text1  1   2 ...
    ...
    textN  0   0   1   1   0   2
    '''
    most_comm = toate_cuvintele.most_common(how_many)
    wd2idx = {}
    idx2wd = {}
    for idx, itr in enumerate(most_comm):
        cuvant = itr[0]
        wd2idx[cuvant] = idx
        idx2wd[idx] = cuvant
    return wd2idx, idx2wd

def get_corpus_vocabulary(corpus):
    '''Write a function to return all the words in a corpus.
    '''
    counter = Counter()
    for text in corpus:
        tokens = tokenize(text)
        counter.update(tokens)
    return counter

def text_to_bow(text, wd2idx):
    '''Convert a text to a bag of words representation.
           @  che  .   ,   di  e
    text   0   1   0   2   0   1
    '''
    features = np.zeros(len(wd2idx))
    for token in tokenize(text):
        if token in wd2idx:
            features[wd2idx[token]] += 1
    return features

def corpus_to_bow(corpus, wd2idx):
    '''Convert a corpus to a bag of words representation.
           @  che  .   ,   di  e
    text0  0   1   0   2   0   1
    text1  1   2 ...
    ...
    textN  0   0   1   1   0   2
    '''

    all_features = []
    for text in corpus:
        all_features.append(text_to_bow(text, wd2idx))
    all_features = np.array(all_features)
    return all_features

def write_prediction(out_file, predictions):
    '''A function to write the predictions to a file.
    id,label
    5001,1
    5002,1
    5003,1
    ...
    '''
    with open(out_file, 'w') as fout:
        # aici e open in variabila 'fout'
        fout.write('id,label\n')
        start_id = 5001
        for i, pred in enumerate(predictions):
            linie = str(i + start_id) + ',' + str(pred) + '\n'
            fout.write(linie)
    # aici e fisierul closed

def cross_validate(k, data, labels):
    chunk_size = int(len(labels)/k)
    indici =np.arange(0, len(labels))
    random.shuffle(indici)
    for i in range(0,len(labels),chunk_size):
        valid_indici= indici[i:i+chunk_size]
        right_side=indici[i+chunk_size:]
        left_side= indici[0:i]
        train_indici=np.concatenate([left_side,right_side])
        train = data[train_indici]
        valid= data[valid_indici]
        y_train=labels[train_indici]
        y_valid= labels[valid_indici]
        yield train,valid,y_train,y_valid


train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
corpus = train_df['text']
text = train_df['text'][2]
toate_cuvintele = get_corpus_vocabulary(corpus)

#folosim primele 1400 cuv
wd2idx, idx2wd = get_representation(toate_cuvintele, 1400)

data = corpus_to_bow(corpus, wd2idx)
labels = train_df['label']
test_data = corpus_to_bow(test_df['text'], wd2idx)

#clasificatorul naive bayes
clf = BernoulliNB(alpha=0.5)

#pentru calcularea timpului
start =time.time()


#scrie predictia in .csv
#write_prediction('BernoulliNB_sub12.csv', predictii)



scoruri = [] #calcul cu fbeta_score
accpred = [] #calcul cu acc de la lab
c=np.zeros((2,2))
for train, valid, y_train, y_valid in cross_validate(10, data, labels):
        clf.fit(train, y_train)
        predictii = clf.predict(valid)
        scor = fbeta_score(y_valid, predictii, beta=1)
        scoruri.append(scor)
        acc = (predictii == y_valid).mean()
        accpred.append(acc)
        #matricea de confuzie pe setul de date curent (4500 train, 500 test) x10
        for pred, true_lbl in zip(predictii.astype(np.int), y_valid):
            c[true_lbl, pred] += 1

print(c)
print('-----')
print(np.mean(scoruri),
      'fbeta_score cu deviatia standard',
          np.std(scoruri))
print('-----')
print(np.mean(accpred),
      'acuratete cu deviatia standard',
            np.std(accpred))

end= time.time()
print('-----')
print(end-start,' secunde a durat antrenarea')


#aplicam 10 fold pentru o performanta constanta
scoruri10=[]
accpred10=[]
for il in range(1,10):
    scoruri =[]
    accpred=[]
    for train, valid, y_train, y_valid in cross_validate(10, data, labels):
        clf.fit(train, y_train)
        predictii=clf.predict(valid)
        scor=fbeta_score(y_valid, predictii, beta=1)
        #print(scor)
        scoruri.append(scor)
        acc=(predictii==y_valid).mean()
        accpred.append(acc)

    scoruri10.append(np.mean(scoruri))
    accpred10.append(np.mean(accpred))

print('----------')
print(np.mean(scoruri10),' ',np.std(scoruri10))
print(np.mean(accpred10),' ',np.std(accpred10))