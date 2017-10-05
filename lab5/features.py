
import dparser
import conll
import time
from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model
from sklearn import metrics
import pickle

def extract1(stack, queue, graph, feature_names, sentence):
    print(stack)
    stackForm = stack[0]['form']
    stackPos = stack[0]['postag']
    queueForm = queue[0]['form']
    queuePos = queue[0]['postag']
    feats = [stackForm, stackPos, queueForm, queuePos]
    features = dict(zip(feature_names, feats))
    return features


def extract2(stack, queue, graph, feature_names, sentence):
    feats = []
    for i in range(2):
        stackForm = stack[i]['form']
        stackPos = stack[i]['postag']
        queueForm = queue[i]['form']
        queuePos = queue[i]['postag']
        feats += [stackForm, stackPos, queueForm, queuePos]
    features = dict(zip(feature_names, feats))
    return features


def extract3(stack, queue, graph, feature_names, sentence):
    feats = []
    for i in range(2):
        stackForm = stack[i]['form']
        stackPos = stack[i]['postag']
        queueForm = queue[i]['form']
        queuePos = queue[i]['postag']
        feats += [stackForm, stackPos, queueForm, queuePos]
    stackHeadPos = sentence[int(graph['heads'][stack[0]['id']])]['postag']
    stackPrevPos = sentence[int(stack[0]['id']) - 1]['postag']
    feats += [stackHeadPos, stackPrevPos]
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
    return (stack,queue,graph)


def createXY(sentences, feature_names):
    X = []
    Y = []
    for sentence in sentences:
        (stack, queue, graph) = initialStructures(sentence)
        while len(queue) > 0:
            (stack, queue, graph, action) = dparser.reference(stack, queue, graph)
            X.append(extract1(stack, queue, graph, feature_names, sentence))
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





if __name__ == "__main__":

    train_file = './swedish_talbanken05_train.conll'
    test_file = './swedish_talbanken05_test_blind.conll'
    column_names_2006 = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats', 'head', 'deprel', 'phead', 'pdeprel']
    column_names_2006_test = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats']

    sentences = conll.read_sentences(train_file)
    formatted_corpus = conll.split_rows(sentences, column_names_2006)

    feature_names = ['stackForm', 'stackPos', 'queueForm', 'queuePos']

    X_dict, y_symbols = createXY(sentences, feature_names)

    print("Encoding the features and classes...")
    # Vectorize the feature matrix and carry out a one-hot encoding
    vec = DictVectorizer(sparse=True)
    X = vec.fit_transform(X_dict)
    # The statement below will swallow a considerable memory
    # X = vec.fit_transform(X_dict).toarray()
    # print(vec.get_feature_names())

    y, dict_classes, inv_dict_classes = encode_classes(y_symbols)

    training_start_time = time.clock()
    print("Training the model...")
    classifier = linear_model.LogisticRegression(penalty='l2', dual=True, solver='liblinear')

    try:
        classifier = pickle.load(open("clf" + ".sav", "rb"))
    except FileNotFoundError:
        classifier.fit(X, y)
        pickle.dump(classifier, open("clf" + ".sav", "wb"))

    test_start_time = time.clock()
    # We apply the model to the test set
    test_sentences = conll.read_sentences(test_file)

    # Here we carry out a chunk tag prediction and we report the per tag error
    # This is done for the whole corpus without regard for the sentence structure
    print("Predicting the chunks in the test set...")
    X_test_dict, y_test_symbols = createXY(test_sentences, feature_names)
    # Vectorize the test set and one-hot encoding
    X_test = vec.transform(X_test_dict)  # Possible to add: .toarray()
    y_test = [inv_dict_classes[i] if i in y_symbols else 0 for i in y_test_symbols]
    y_test_predicted = classifier.predict(X_test)
    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(y_test, y_test_predicted)))

    # Here we tag the test set and we save it.
    # This prediction is redundant with the piece of code above,
    # but we need to predict one sentence at a time to have the same
    # corpus structure
    print("Predicting the test set...")
    f_out = open('out', 'w')
    #predict(test_sentences, feature_names, f_out)

    end_time = time.clock()
    print("Training time:", (test_start_time - training_start_time) / 60)
    print("Test time:", (end_time - test_start_time) / 60)
