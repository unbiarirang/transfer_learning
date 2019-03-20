from konlpy.tag import Okt 
from pprint import pprint
import pickle
import nltk

pos_tagger = Okt()

def read_data(filename):
    with open(filename, 'r') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        data = data[1:] # remove header
    return data

def tokenize(doc):
    return ['/'.join(t) for t in pos_tagger.pos(doc, norm=True, stem=True)]

# Data preprocessing. Execute only once
def import_data():
    train_data = read_data('ratings_train.txt')
    test_data = read_data('ratings_test.txt')

    train_docs = [(tokenize(row[1]), row[2]) for row in train_data]
    test_docs = [(tokenize(row[1]), row[2]) for row in test_data]

    save_obj(train_data, 'train_docs.obj')
    save_obj(test_data, 'test_docs.obj')

def save_obj(obj, filename):
    pickle.dump(obj, open(filename, 'wb'))

def read_obj(filename):
    return pickle.load(open(filename, 'rb'))

def term_exists(doc):
    return {'exists({})'.format(word): (word in set(doc)) for word in selected_words}

train_docs = read_obj('train_docs.obj')
train_docs = train_docs[:10000]
test_docs = read_obj('test_docs.obj')

pprint(train_docs[0])

tokens = [t for d in train_docs for t in d[0]]
print("tokens length: ", len(tokens))

text = nltk.Text(tokens, name='NMSC')
selected_words =  [f[0] for f in text.vocab().most_common(2000)]

# Exclude top 50 most common words
#selected_words = selected_words[50:]

train_xy = [(term_exists(d), c) for d, c in train_docs]
test_xy = [(term_exists(d), c) for d, c in test_docs]

classifier = nltk.NaiveBayesClassifier.train(train_xy)
print(nltk.classify.accuracy(classifier, test_xy))
classifier.show_most_informative_features(10)
