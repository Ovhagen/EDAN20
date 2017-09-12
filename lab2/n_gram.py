import regex as re
from typing import List, Iterator
import math
from functools import reduce
#Regex (?s) = flags gXs active. global,
# disallow meaningless escapes and dot matches new line

def senteces(text: str) ->  Iterator:
    return re.finditer('\p{Lu}[^\.]+\.', text)

def tokenize(text):
    words = re.findall('\p{L}+', text)
    return words

def receiveSenteces(senteces: Iterator) -> List[str]:
    sent_idx = []
    for match in senteces:
        sent = match.group().replace("\n", ' ')
        sent = sent[:-1]
        sent = tokenize(sent.lower())
        sent = ["<s>"] + sent + ["</s>"]
        sent_idx.append(sent)
    return sent_idx


def count_unigrams(words):
    frequency = {}
    for word in words:
        if word in frequency:
            frequency[word] += 1
        else:
            frequency[word] = 1
    return frequency


def count_bigrams(words):
    bigrams = [tuple(words[inx:inx + 2])
               for inx in range(len(words) - 1)]
    frequencies = {}
    for bigram in bigrams:
        if bigram in frequencies:
            frequencies[bigram] += 1
        else:
            frequencies[bigram] = 1
    return frequencies


def mutual_info(words, freq_unigrams, freq_bigrams):
    mi = {}
    factor = len(words) * len(words) / (len(words) - 1)
    for bigram in freq_bigrams:
        mi[bigram] = (
            math.log(factor * freq_bigrams[bigram] /
                     (freq_unigrams[bigram[0]] *
                      freq_unigrams[bigram[1]]), 2))
    return mi



def uniP(word):
    try:
        return (unigrams[word] + 1)/len(allWords)
    except KeyError:
        return 1/len(allWords)


bar = "====================================================="

def mul(a,b):
    return a*b

def unigramProb(sentence):
    print("Unigram  model")
    print(bar)
    print("wi C(wi) #words P(wi)")
    print(bar)
    probs = []
    for w in sentence:
        prob = uniP(w)
        probs = probs + [prob]
        print(w,unigrams[w],len(allWords),prob)
    print(bar)
    totalProb = reduce(mul,probs)
    print("Prob. unigrams:",totalProb)
    print("Entropy rate:",-math.log2(totalProb)/len(sentence))


alpha = 1.0

def biP(tuple):
    try:
        return bigrams[tuple]/unigrams[tuple[0]]
    except KeyError:
        return alpha*uniP(tuple[1])


def CB(tuple):
    try:
        return bigrams[tuple]
    except KeyError:
        return 0

def bigramProb(sentence):
    print("Bigram model")
    print(bar)
    print("wi wi+1 Ci,i+1 C(i) P(wi+1|wi)")
    print(bar)
    bs = [tuple(sentence[inx:inx + 2])
     for inx in range(len(sentence) - 1)]
    for b in bs:
        print(b[0],b[1],CB(tuple),unigrams[b[0]],biP(b))


textSenteces = receiveSenteces(senteces(open("Selma.txt").read().strip()))
#print(textSenteces)
#print(len(textSenteces))
allWords = [w for s in textSenteces for w in s]
unigrams = count_unigrams(allWords)
bigrams = count_bigrams(allWords)
print("Antal unika ord:",len(unigrams))
print(len(unigrams)*len(unigrams))
print("Antal unika bigram: ",len(bigrams))
#print(mutual_info(allWords, unigrams, bigrams))
test = ["det","var","en","gång","en","katt","som","hette","nils","</s>"]
#unigramProb(test)
bigramProb(test)
#print(count_bigrams(allWords))
