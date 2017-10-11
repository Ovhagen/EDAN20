
import dparser
import transition
import conll
from sklearn.feature_extraction import DictVectorizer

import pickle


def extract(stack, queue, graph, feature_names, sentence, featSet):
    if(featSet == 1):
        return extract1(stack,queue,graph,feature_names,sentence)
    if(featSet == 2):
        return extract2(stack, queue, graph, feature_names, sentence)
    if (featSet == 3):
        return extract3(stack, queue, graph, feature_names, sentence)


def extract1(stack, queue, graph, feature_names, sentence):
    if stack:
        stackForm = stack[0]['form']
        stackPos = stack[0]['postag']
    else:
        stackForm = "nil"
        stackPos = "nil"

    if queue:
        queueForm = queue[0]['form']
        queuePos = queue[0]['postag']
    else:
        queueForm = "nil"
        queuePos = "nil"

    canRe = str(transition.can_reduce(stack, graph))
    canLa = str(transition.can_leftarc(stack, graph))

    feats = [stackPos, stackForm, queuePos, queueForm] + [canRe, canLa]
    features = dict(zip(feature_names, feats))
    return features


def extract2(stack, queue, graph, feature_names, sentence):
    feats = []
    if stack:
        stackForm = stack[0]['form']
        stackPos = stack[0]['postag']
    else:
        stackForm = "nil"
        stackPos = "nil"

    if len(stack) > 1:
        stackForm_2 = stack[1]['form']
        stackPos_2 = stack[1]['postag']
    else:
        stackForm_2 = "nil"
        stackPos_2 = "nil"

    if queue:
        queueForm = queue[0]['form']
        queuePos = queue[0]['postag']
    else:
        queueForm = "nil"
        queuePos = "nil"

    if len(queue) > 1:
        queueForm_2 = queue[1]['form']
        queuePos_2 = queue[1]['postag']
    else:
        queueForm_2 = "nil"
        queuePos_2 = "nil"

    canRe = str(transition.can_reduce(stack, graph))
    canLa = str(transition.can_leftarc(stack, graph))
    feats += [stackPos, stackPos_2, stackForm, stackForm_2,
              queuePos, queuePos_2, queueForm, queueForm_2] + [canRe, canLa]
    features = dict(zip(feature_names, feats))
    return features


def extract3(stack, queue, graph, feature_names, sentence):
    feats = []
    if stack:
        stackForm = stack[0]['form']
        stackPos = stack[0]['postag']
        if stack[0]['id'] in graph['heads'].keys() :
            stackHeadPos = sentence[int(graph['heads'][stack[0]['id']])]['postag']
            stackHeadForm = sentence[int(graph['heads'][stack[0]['id']])]['form']
        else :
            stackHeadPos = "nil"
            stackHeadForm = "nil"
        if int(stack[0]['id']) > 0 and len(sentence) > int(stack[0]['id']):
            stackNextPos = sentence[int(stack[0]['id']) + 1]['postag']
            stackNextForm = sentence[int(stack[0]['id']) + 1]['form']
        else :
            stackNextPos = "nil"
            stackNextForm = "nil"
    else:
        stackForm = "nil"
        stackPos = "nil"
        stackHeadPos = "nil"
        stackHeadForm = "nil"
        stackNextPos = "nil"
        stackNextForm = "nil"

    if len(stack) > 1:
        stackForm_2 = stack[1]['form']
        stackPos_2 = stack[1]['postag']
    else:
        stackForm_2 = "nil"
        stackPos_2 = "nil"

    if queue:
        queueForm = queue[0]['form']
        queuePos = queue[0]['postag']
    else:
        queueForm = "nil"
        queuePos = "nil"

    if len(queue) > 1:
        queueForm_2 = queue[1]['form']
        queuePos_2 = queue[1]['postag']
    else:
        queueForm_2 = "nil"
        queuePos_2 = "nil"

    canRe = str(transition.can_reduce(stack, graph))
    canLa = str(transition.can_leftarc(stack, graph))

    feats += [stackPos, stackPos_2, stackForm, stackForm_2,
              queuePos, queuePos_2, queueForm, queueForm_2,
              stackHeadPos, stackHeadForm, stackNextPos, stackNextForm] + [canRe, canLa]
    features = dict(zip(feature_names, feats))
    return features


def initialStructures(sentence):
    graph = {}
    graph['heads'] = {}
    graph['heads']['0'] = '0'
    graph['deprels'] = {}
    graph['deprels']['0'] = 'ROOT'
    stack = []
    queue = list(sentence)
    return stack, queue, graph


def createXY(sentences, feature_names, featSet = 3):
    X = []
    Y = []
    for sentence in sentences:
        (stack, queue, graph) = initialStructures(sentence)
        while queue:
            X.append(extract(stack, queue, graph, feature_names, sentence, featSet))
            (stack, queue, graph, action) = dparser.reference(stack, queue, graph)
            Y.append(action)
    return X, Y


def encode_classes(y_symbols):
    """
    Encode the classes as numbers
    :param y_symbols:
    :return: the y vector and the lookup dictionaries
    """
    # We extract the chunk names
    classes = sorted(list(set(y_symbols)))
    """
    Results in:
    ['B-ADJP', 'B-ADVP', 'B-CONJP', 'B-INTJ', 'B-LST', 'B-NP', 'B-PP',
    'B-PRT', 'B-SBAR', 'B-UCP', 'B-VP', 'I-ADJP', 'I-ADVP', 'I-CONJP',
    'I-INTJ', 'I-NP', 'I-PP', 'I-PRT', 'I-SBAR', 'I-UCP', 'I-VP', 'O']
    """
    # We assign each name a number
    dict_classes = dict(enumerate(classes))
    """
    Results in:
    {0: 'B-ADJP', 1: 'B-ADVP', 2: 'B-CONJP', 3: 'B-INTJ', 4: 'B-LST',
    5: 'B-NP', 6: 'B-PP', 7: 'B-PRT', 8: 'B-SBAR', 9: 'B-UCP', 10: 'B-VP',
    11: 'I-ADJP', 12: 'I-ADVP', 13: 'I-CONJP', 14: 'I-INTJ',
    15: 'I-NP', 16: 'I-PP', 17: 'I-PRT', 18: 'I-SBAR',
    19: 'I-UCP', 20: 'I-VP', 21: 'O'}
    """

    # We build an inverted dictionary
    inv_dict_classes = {v: k for k, v in dict_classes.items()}
    """
    Results in:
    {'B-SBAR': 8, 'I-NP': 15, 'B-PP': 6, 'I-SBAR': 18, 'I-PP': 16, 'I-ADVP': 12,
    'I-INTJ': 14, 'I-PRT': 17, 'I-CONJP': 13, 'B-ADJP': 0, 'O': 21,
    'B-VP': 10, 'B-PRT': 7, 'B-ADVP': 1, 'B-LST': 4, 'I-UCP': 19,
    'I-VP': 20, 'B-NP': 5, 'I-ADJP': 11, 'B-CONJP': 2, 'B-INTJ': 3, 'B-UCP': 9}
    """

    # We convert y_symbols into a numerical vector
    y = [inv_dict_classes[i] for i in y_symbols]
    return y, dict_classes, inv_dict_classes


def featNames(featSet):
    if featSet == 1:
        return ['stackPos', 'stackForm', 'queuePos', 'queueForm', 'can_re', 'can_la']
    if featSet == 2:
        return ['stackPos', 'stackPos_2', 'stackForm', 'stackForm_2', 'queuePos',
                      'queuePos_2', 'queueForm', 'queueForm_2', 'can_re', 'can_la']
    if featSet == 3:
        return ['stackPos', 'stackPos_2', 'stackForm', 'stackForm_2', 'queuePos',
                      'queuePos_2', 'queueForm', 'queueForm_2', 'stackHeadPos', 'stackHeadForm', 'stackNextPos',
                      'stackNextForm', 'can_re', 'can_la']



def getClassifier(featSet):
    try:
        classifier = pickle.load(open("clf" + featSet + ".sav", "rb"))
        return classifier
    except FileNotFoundError:
        print("ingen fil")


def getStuff(featSet):
    train_file = './swedish_talbanken05_train.conll'
    column_names_2006 = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats', 'head', 'deprel', 'phead', 'pdeprel']
    sentences = conll.read_sentences(train_file)
    formatted_corpus = conll.split_rows(sentences, column_names_2006)
    feature_names = featNames(featSet)
    X_dict, y_symbols = createXY(formatted_corpus, feature_names, featSet)
    vec = DictVectorizer(sparse=True)
    vec.fit_transform(X_dict)
    return vec, encode_classes(y_symbols)

