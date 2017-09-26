"""
CoNLL-X and CoNLL-U file readers and writers
"""
__author__ = "Pierre Nugues"

import os
from functools import reduce


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



def extractSubjectVerbPairs(formatted_corpus):
    for s in formatted_corpus:
        for w in s:
            w['form'] = w['form'].lower()
    svMap = {}
    for sentence in formatted_corpus:
        subjects = filter(lambda word: word['deprel'] == 'SS',sentence)
        subjVerbs = map(lambda s: (s['form'], sentence[int(s['head'])]['form']), subjects)
        for sv in subjVerbs:
            svMap[sv] = 1 + svMap.get(sv, 0)
    return svMap

def sortedSubjectVerbPairs(svMap):
    return sorted(svMap.items(), key=lambda p: p[1])


def pairCount(svMap):
    return reduce(lambda a,b: a+b, map(lambda k: svMap[k], svMap.keys()))

def extractSubjectVerbObjectTripples(formatted_corpus):
    for s in formatted_corpus:
        for w in s:
            w['form'] = w['form'].lower()
    SVOs = {}
    for sentence in formatted_corpus:
        subjects = filter(lambda word: word['deprel'] == 'SS', sentence)
        subjVerbs = map(lambda s: (s, sentence[int(s['head'])]), subjects)
        objects = filter(lambda word: word['deprel'] == 'OO', sentence)
        for o in objects:
            for sv in subjVerbs:
                if sentence[int(o['head'])] == sv[1]:
                    subject = sv[0]['form']
                    verb = sv[1]['form']
                    object = o['form']
                    SVOs[(subject, verb, object)] = 1 + SVOs.get((subject, verb, object), 0)
    return SVOs


if __name__ == '__main__':
    column_names_2006 = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats', 'head', 'deprel', 'phead', 'pdeprel']

    train_file = './swedish_talbanken05_train.conll'
    # train_file = 'test_x'
    test_file = './swedish_talbanken05_test.conll'

    sentences = read_sentences(train_file)
    formatted_corpus = split_rows(sentences, column_names_2006)
    #print(train_file, len(formatted_corpus))
    #print(formatted_corpus[0])

    svs = extractSubjectVerbPairs(formatted_corpus)
    svos = extractSubjectVerbObjectTripples(formatted_corpus)
    #print(formatted_corpus[2])
    #print(svos)
    #print(len(svos))
    #print(pairCount(svos))
    #print(sortedSubjectVerbPairs(svos))

    gifter = list(filter(lambda s: len(list(filter(lambda w: w['form'] == 'gifter', s))) > 0, formatted_corpus))
    print(extractSubjectVerbObjectTripples(gifter))

    column_names_u = ['id', 'form', 'lemma', 'upostag', 'xpostag', 'feats', 'head', 'deprel', 'deps', 'misc']

    files = get_files('./corpus/ud-treebanks-v1.3/', 'train.conll')
    for train_file in files:
        sentences = read_sentences(train_file)
        formatted_corpus = split_rows(sentences, column_names_u)
        print(train_file, len(formatted_corpus))
        print(formatted_corpus[0])
