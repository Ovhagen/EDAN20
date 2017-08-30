import re
from collections import Counter


def words(text):
    return re.findall(r'\w+', text.lower())


def indecies(word,text):
    i = re.finditer(r"\b{0}\b".format(word), text)
    return [m.start(0) for m in i]


def fileindex(filepath):
    text = open(filepath).read()
    ws = words(text)
    return dict({w, indecies(w, text)} for w in set(ws) )


print(fileindex('Selma/bannlyst.txt'))

WORDS = Counter(words(open('Selma/bannlyst.txt').read()))

print(WORDS)