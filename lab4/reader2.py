"""
CoNLL-X and CoNLL-U file readers and writers
"""
__author__ = "Pierre Nugues"

import os
from functools import reduce


#Debugging

def contains(word, sentence):
    return len(list(filter(lambda w: w['form'] == word, sentence))) > 0


def containsAndHasRelation(word, sentence, relation):
    return len(list(filter(lambda w: w['form'] == word and w['deprel'] == relation, sentence))) > 0


def reformat(word):
    return [word['id'], word['form'], word['head'], word['deprel']]


def reformatSentence(sentence):
    return list(map(lambda w: reformat(w), sentence))


def reformatAll(sentences):
    return list(map(lambda s: reformatSentence(s), sentences))


def printSentence(sentence):
    print(reformatSentence(sentence))
    print()

#Code

def get_files(dir, suffix):
    """
    Returns all the files in a folder ending with suffix
    Recursive version
    :param dir:
    :param suffix:
    :return: the list of file names
    """
    files = []
    for file in os.listdir(dir):
        path = dir + '/' + file
        if os.path.isdir(path):
            files += get_files(path, suffix)
        elif os.path.isfile(path) and file.endswith(suffix):
            files.append(path)
    return files


def read_sentences(file):
    """
    Creates a list of sentences from the corpus
    Each sentence is a string
    :param file:
    :return:
    """
    f = open(file).read().strip()
    sentences = f.split('\n\n')
    return sentences


def split_rows(sentences, column_names):
    """
    Creates a list of sentence where each sentence is a list of lines
    Each line is a dictionary of columns
    :param sentences:
    :param column_names:
    :return:
    """
    new_sentences = []
    root_values = ['0', 'ROOT', 'ROOT', 'ROOT', 'ROOT', 'ROOT', '0', 'ROOT', '0', 'ROOT']
    start = [dict(zip(column_names, root_values))]
    for sentence in sentences:
        rows = sentence.split('\n')
        sentence = [dict(zip(column_names, row.split())) for row in rows if row[0] != '#']
        sentence = start + sentence
        new_sentences.append(sentence)
    return new_sentences


def save(file, formatted_corpus, column_names):
    f_out = open(file, 'w')
    for sentence in formatted_corpus:
        for row in sentence[1:]:
            # print(row, flush=True)
            for col in column_names[:-1]:
                if col in row:
                    f_out.write(row[col] + '\t')
                else:
                    f_out.write('_\t')
            col = column_names[-1]
            if col in row:
                f_out.write(row[col] + '\n')
            else:
                f_out.write('_\n')
        f_out.write('\n')
    f_out.close()


def lowerize(formatted_corpus):
    for s in formatted_corpus:
        for w in s:
            w['form'] = w['form'].lower()
    return formatted_corpus

def transformCorpus(formatted_corpus):
    res = []
    for sentence in formatted_corpus:
        dictSent = {}
        for word in sentence:
            dictSent[word['id']] = word
        res.append(dictSent)
    return res


def extractSubjectVerbPairs(formatted_corpus):
    svMap = {}
    for sentence in formatted_corpus:
        subjects = filter(lambda word: word['deprel'] == 'nsubj',list(sentence.values()))
        subjVerbs = map(lambda s: (s['form'], sentence[s['head']]['form']), subjects)
        for sv in subjVerbs:
            svMap[sv] = 1 + svMap.get(sv, 0)
    return svMap


def sortedSubjectVerbPairs(svMap):
    return sorted(svMap.items(), key=lambda p: p[1])


def pairCount(svMap):
    return reduce(lambda a,b: a+b, map(lambda k: svMap[k], svMap.keys()))


def extractSubjectVerbObjectTripples(formatted_corpus):
    SVOs = {}
    for sentence in formatted_corpus:
        subjects = list(filter(lambda word: word['deprel'] == 'nsubj', list(sentence.values())))
        subjVerbs = list(map(lambda s: (s, sentence[s['head']]), subjects))
        objects = list(filter(lambda word: word['deprel'] == 'obj', list(sentence.values())))
        for o in objects:
            for sv in subjVerbs:
                if sentence[o['head']]["id"] == sv[1]["id"]:
                    subject = sv[0]['form']
                    verb = sv[1]['form']
                    object = o['form']
                    SVOs[(subject, verb, object)] = 1 + SVOs.get((subject, verb, object), 0)
    return SVOs


if __name__ == '__main__':
    column_names_2006 = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats', 'head', 'deprel', 'phead', 'pdeprel']

    column_names_u = ['id', 'form', 'lemma', 'upostag', 'xpostag', 'feats', 'head', 'deprel', 'deps', 'misc']

    files = get_files('./ud-treebanks-v2.0/UD_Swedish', 'train.conllu')
    print(len(files))
    for train_file in files:
        sentences = read_sentences(train_file)
        formatted_corpus = lowerize(split_rows(sentences, column_names_u))
        corpus = transformCorpus(formatted_corpus)
        #print(corpus)
        svs = extractSubjectVerbPairs(corpus)
        svos = extractSubjectVerbObjectTripples(corpus)
        print(sortedSubjectVerbPairs(svs)[-5:])
        print(sortedSubjectVerbPairs(svos)[-5:])
