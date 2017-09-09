import regex as re

#Regex (?s) = flags gXs active. global,
# disallow meaningless escapes and dot matches new line

def senteces(text):
    return re.finditer('[A-Z][\p{L}1-9,;\s]+[.!?]', text)

def receiveSenteces(senteces):
    sent_idx = []
    for match in senteces:
        sent = match.group()
        sent_idx.append(sent)
    return sent_idx

textSenteces = receiveSenteces(senteces(open("Selma.txt").read().strip()))
print(textSenteces[5001])
print(len(textSenteces))