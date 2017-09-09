import regex as re
from typing import List, Iterator

#Regex (?s) = flags gXs active. global,
# disallow meaningless escapes and dot matches new line

def senteces(text: str) ->  Iterator:
    return re.finditer('\p{Lu}[\p{L},\s]+\.?', text)

def receiveSenteces(senteces: Iterator) -> List[str]:
    sent_idx = []
    for match in senteces:
        sent = match.group().replace("\n", ' ')
        sent_idx.append(sent)
    return sent_idx

textSenteces = receiveSenteces(senteces(open("Selma.txt").read().strip()))
print(textSenteces)
print(len(textSenteces))