import regex as re
import pickle
import codecs
from collections import Counter
from collections import defaultdict


def words(text):
    return re.finditer(r'\p{L}+', text.lower())

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



fileindex('bannlyst') #Starts generating dictionary from bannlyst.txt

#Tests
dic = pickle.load(open('bannlyst'+".idx", "rb"))
print(dic)

print("gjord", dic.get('gjord'))            #Should be [8551, 183692, 220875]
print("uppklarnade",dic.get('uppklarnade')) #Should be [8567]
print("stjärnor",dic.get('stjärnor'))       #Should be [8590]
