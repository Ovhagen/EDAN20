import regex as re
import pickle
import codecs
from collections import Counter
from collections import defaultdict


def words(text):
    return re.finditer(r'[\wåäö]+', text.lower())


def fileindex(filename):
    with codecs.open("Selma/"+filename+".txt", 'r', encoding='utf8') as f:
        text = f.read()
    ws = words(text)
    dic = defaultdict()
    for match in ws:
        word = match.group()
        ind = match.start()
        if (dic.get(word)):
            dic[word].append(ind)
        else:
            dic[word] = [ind]
            
    pickle.dump(dic, open(filename+".idx", "wb"))


fileindex('bannlyst') #Starts generating dictionary
    
print(pickle.load(open('bannlyst'+".idx", "rb")))
