import features
import transition
import conll


featSet = 3

classifier = features.getClassifier(str(featSet))
feature_names = features.featNames(featSet)
vec, (y, dict_classes, inv_dict_classes) = features.getStuff(featSet)

test_file = './swedish_talbanken05_test.conll'

column_names_2006 = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats', 'head', 'deprel', 'phead', 'pdeprel']
column_names_2006_test = ['id', 'form', 'lemma', 'cpostag', 'postag', 'feats']


def parse_ml(stack, queue, graph, trans):
    if stack and trans[:2] == 'ra':
        stack, queue, graph = transition.right_arc(stack, queue, graph, trans[3:])
        return stack, queue, graph, 'ra'
    if stack and trans[:2] == 'la':
        stack, queue, graph = transition.left_arc(stack, queue, graph, trans[3:])
        return stack, queue, graph, 'la'
    if trans == 're':
        stack, queue, graph = transition.reduce(stack, queue, graph)
        return stack, queue, graph, 're'
    if trans == 'sh':
        stack, queue, graph = transition.shift(stack, queue, graph)
        return stack, queue, graph, 'sh'
    print(trans, "is not a valid action")
    return None


def parse(sentence):
    (stack, queue, graph) = features.initialStructures(sentence)
    while queue:
        feats = features.extract(stack, queue, graph, feature_names, sentence, featSet)
        featVect = vec.transform(feats)
        trans_nr = classifier.predict(featVect)[0]
        trans = dict_classes[trans_nr]
        stack, queue, graph, trans = parse_ml(stack, queue, graph, trans)
    return graph


def forms(sentence):
    res = ""
    for word in sentence:
        res += (word['form'] + " ")
    return res



if __name__ == "__main__":


    test_sentences = conll.read_sentences(test_file)
    test_sentences = conll.split_rows(test_sentences, column_names_2006)

    for sentence in test_sentences:
        graph = parse(sentence)
        for word in sentence:
            if word['id'] in graph['heads'].keys():
                word['head'] = graph['heads'][word['id']]
                word['deprel'] = graph['deprels'][word['id']]
            else:
                word['head'] = '_'
                word['deprel'] = '_'

    conll.save("parsedTestSentences", test_sentences, column_names_2006)


