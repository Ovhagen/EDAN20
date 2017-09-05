import regex as re
import pickle
import os
import sys
import math
from typing import Dict,List
from functools import reduce

N = -1
docs = []

def words(text):
    return re.finditer(r'\p{L}+', text)

def get_files(dir, suffix):
    """
    Returns all the files in a folder ending with suffix
    :param dir:
    :param suffix:
    :return: the list of file names
    """
    files = []
    for file in os.listdir(dir):
        if file.endswith(suffix):
            files.append(file)
    N = len(files)
    return files

def words_to_index(ws):
    word_idx = {}
    for match in ws:
        word = match.group()
        ind = match.start()
        if (word_idx.get(word)):
            word_idx[word].append(ind)
        else:
            word_idx[word] = [ind]
    return word_idx

def add(xs:Dict[str,int],ys:Dict[str,int]) -> Dict[str,int]:
    res = {}
    for x in xs:
        res[x] = xs[x]
    for y in ys:
        res[y] = res.get(y,0) + ys[y]
    return res

def worddicts(master_index):
    for w in master_index:
        yield dict(map(lambda t: (t[0], len(t[1])), master_index[w].items()))


def tf(master_index: Dict[str,Dict[str,List[int]]]):
    word_frq = {}
    s = reduce(add, worddicts(master_index))

    for w in master_index:
        for doc in master_index[w]:
            tf = len(master_index[w][doc])/s[doc]
            if w in word_frq:
                word_frq[w][doc] = tf
            else:
                word_frq[w] = {}
                word_frq[w][doc] = tf

    return word_frq

def idf(master_index):
    idfs = {}
    for w in master_index:
        M = len(master_index[w])
        idfs[w] = math.log10(N/M)
    return idfs

def docToVector(dic, doc):
    res = []
    for w in dic:
        res.append(tf[w][doc] * idf[w])
    return res

def norm(vector):
   return math.sqrt(sum(vector[i]*vector[i]) for i in vector.keys())

def cosineSim(x,y,dic):
    xwords = dict(map(lambda t : (t[0],tf[t[0]][x]*idf[t[0]]), map(lambda t : (t[0],t[1][x]), filter(lambda t : x in t[1], dic.items()))))
    ywords = dict(map(lambda t : (t[0],tf[t[0]][y]*idf[t[0]]), map(lambda t : (t[0],t[1][y]), filter(lambda t : y in t[1], dic.items()))))
    inter = set(xwords.keys()).intersection(set(ywords.keys()))
    dot = sum(xwords[key][x]*ywords[key][y] for key in inter )
    return dot/(norm(xwords)*norm(ywords))


def fileindex(dir):
    master_index = {}
    files = get_files(dir, ".txt")
    for file in files:
        text = open(dir+file).read().lower().strip()
        ws = words(text)
        idx = words_to_index(ws)
        for word in idx:
            if word in master_index:
                master_index[word][file] = idx[word]
            else:
                master_index[word] = {}
                master_index[word][file] = idx[word]

    pickle.dump(master_index, open("master_index" + ".idx", "wb")) #[:-4] to remove .txt part of string


#Main
arguments = sys.argv
if arguments[1] != '':
    print("Search in folder", arguments[1])
    docs = get_files(arguments[1], '.txt')
    N = len(docs)
    fileindex(arguments[1]+'/') #Starts generating indexed dictionary from files
else:
    raise ValueError("Invalid argument. Input should be the directory from where files should be read.")


#Tests
dic = pickle.load(open('master_index'+".idx", "rb"))
#print('gjord:', dic['gjord']) #Should be [8551, 183692, 220875]
#print('uppklarnande:', dic['uppklarnande']) #Should be [8567]
#print('stjärnor:', dic['stjärnor']) #Should be [8590]
tf = tf(dic)
idf = idf(dic)
#print(tf)
for w in dic:
    for doc in dic[w]:
        x = 0
        if(w == "et" or w == "gås"):
            print(w, doc, tf[w][doc]*idf[w])

for d1 in docs:
    for d2 in docs:
        print(cosineSim(d1,d2,dic))